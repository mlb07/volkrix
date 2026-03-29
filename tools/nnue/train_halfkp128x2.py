#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


EXAMPLES_MAGIC = "VOLKRIX_EXAMPLES"
EXAMPLES_VERSION = 1
CHECKPOINT_MAGIC = "VOLKRIX_HALFKP128X2_CHECKPOINT"
CHECKPOINT_VERSION = 1
TRAINER_VERSION = "phase13-v1"
TOPOLOGY_NAME = "HalfKP128x2"
TOPOLOGY_ID = 1
FEATURE_COUNT = 40960
HIDDEN_SIZE = 128
OUTPUT_INPUTS = 256
OUTPUT_SCALE = 64
SEED = 13
OPTIMIZER = "AdamW"
LOSS = "SmoothL1Loss"
NORMALIZED_FEN_RULE = (
    "normalized FEN is Position::from_fen(input)?.to_fen(), preserving all 6 canonical fields"
)
SPLIT_RULE = "fnv1a64(normalized_fen_utf8) % 10 == 0 => validation"
INPUT_WEIGHTS_FILE = "input_weights.f32le"
HIDDEN_BIASES_FILE = "hidden_biases.f32le"
OUTPUT_WEIGHTS_FILE = "output_weights.f32le"
OUTPUT_BIAS_FILE = "output_bias.f32le"
EPOCHS = 4
LEARNING_RATE = 0.0025
WEIGHT_DECAY = 0.0001
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8


@dataclass
class Example:
    normalized_fen: str
    target_cp: float
    active_features: np.ndarray
    passive_features: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Volkrix HalfKP128x2 checkpoint")
    parser.add_argument("--examples", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    return parser.parse_args()


def fnv1a64(text: str) -> int:
    hash_value = 0xCBF29CE484222325
    for byte in text.encode("utf-8"):
        hash_value ^= byte
        hash_value = (hash_value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return hash_value


def split_for_normalized_fen(normalized_fen: str) -> str:
    return "validation" if fnv1a64(normalized_fen) % 10 == 0 else "train"


def parse_features(text: str) -> np.ndarray:
    if not text:
        return np.empty((0,), dtype=np.int32)
    return np.fromiter((int(part) for part in text.split(",")), dtype=np.int32)


def read_examples(path: Path) -> tuple[dict, list[Example]]:
    with path.open("r", encoding="utf-8") as handle:
        magic_line = handle.readline().rstrip("\n")
        if magic_line != f"{EXAMPLES_MAGIC}\t{EXAMPLES_VERSION}":
            raise ValueError(f"unsupported examples header: {magic_line!r}")

        manifest_line = handle.readline().rstrip("\n")
        if not manifest_line.startswith("# "):
            raise ValueError("examples manifest line must start with '# '")
        manifest = json.loads(manifest_line[2:])

        columns = handle.readline().rstrip("\n")
        expected_columns = (
            "fen\tnormalized_fen\tside_to_move\traw_score_cp\ttarget_cp\tactive_features\tpassive_features"
        )
        if columns != expected_columns:
            raise ValueError("unsupported examples column layout")

        examples: list[Example] = []
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            columns = line.split("\t")
            if len(columns) != 7:
                raise ValueError(f"expected 7 columns, found {len(columns)}")
            examples.append(
                Example(
                    normalized_fen=columns[1],
                    target_cp=float(columns[4]),
                    active_features=parse_features(columns[5]),
                    passive_features=parse_features(columns[6]),
                )
            )
    return manifest, examples


def smooth_l1_grad(diff: float) -> float:
    if abs(diff) < 1.0:
        return diff
    return 1.0 if diff > 0 else -1.0


def forward(
    input_weights: np.ndarray,
    hidden_biases: np.ndarray,
    output_weights: np.ndarray,
    output_bias: np.ndarray,
    example: Example,
) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    active_hidden = hidden_biases.copy()
    if example.active_features.size:
        active_hidden += input_weights[example.active_features].sum(axis=0)

    passive_hidden = hidden_biases.copy()
    if example.passive_features.size:
        passive_hidden += input_weights[example.passive_features].sum(axis=0)

    active_clipped = np.clip(active_hidden, 0.0, 255.0)
    passive_clipped = np.clip(passive_hidden, 0.0, 255.0)
    prediction = (
        output_bias[0]
        + float(np.dot(active_clipped, output_weights[:HIDDEN_SIZE]))
        + float(np.dot(passive_clipped, output_weights[HIDDEN_SIZE:]))
    ) / OUTPUT_SCALE
    return prediction, (active_hidden, passive_hidden, active_clipped, passive_clipped)


def adamw_update_dense(
    params: np.ndarray,
    grad: np.ndarray,
    moment1: np.ndarray,
    moment2: np.ndarray,
    step: int,
) -> None:
    params -= LEARNING_RATE * WEIGHT_DECAY * params
    moment1 *= BETA1
    moment1 += (1.0 - BETA1) * grad
    moment2 *= BETA2
    moment2 += (1.0 - BETA2) * (grad * grad)
    m_hat = moment1 / (1.0 - BETA1**step)
    v_hat = moment2 / (1.0 - BETA2**step)
    params -= LEARNING_RATE * m_hat / (np.sqrt(v_hat) + EPS)


def adamw_update_rows(
    params: np.ndarray,
    grad_rows: np.ndarray,
    row_indices: np.ndarray,
    moment1: np.ndarray,
    moment2: np.ndarray,
    step: int,
) -> None:
    if row_indices.size == 0:
        return
    params[row_indices] -= LEARNING_RATE * WEIGHT_DECAY * params[row_indices]
    moment1[row_indices] *= BETA1
    moment1[row_indices] += (1.0 - BETA1) * grad_rows
    moment2[row_indices] *= BETA2
    moment2[row_indices] += (1.0 - BETA2) * (grad_rows * grad_rows)
    m_hat = moment1[row_indices] / (1.0 - BETA1**step)
    v_hat = moment2[row_indices] / (1.0 - BETA2**step)
    params[row_indices] -= LEARNING_RATE * m_hat / (np.sqrt(v_hat) + EPS)


def train(
    examples: list[Example],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
]:
    rng = np.random.default_rng(SEED)
    input_weights = rng.normal(0.0, 0.05, size=(FEATURE_COUNT, HIDDEN_SIZE)).astype(np.float32)
    hidden_biases = np.full((HIDDEN_SIZE,), 16.0, dtype=np.float32)
    output_weights = rng.normal(0.0, 0.05, size=(OUTPUT_INPUTS,)).astype(np.float32)
    output_bias = np.zeros((1,), dtype=np.float32)

    input_m1 = np.zeros_like(input_weights)
    input_m2 = np.zeros_like(input_weights)
    hidden_m1 = np.zeros_like(hidden_biases)
    hidden_m2 = np.zeros_like(hidden_biases)
    output_m1 = np.zeros_like(output_weights)
    output_m2 = np.zeros_like(output_weights)
    bias_m1 = np.zeros_like(output_bias)
    bias_m2 = np.zeros_like(output_bias)

    train_examples = [example for example in examples if split_for_normalized_fen(example.normalized_fen) == "train"]
    validation_examples = [
        example for example in examples if split_for_normalized_fen(example.normalized_fen) == "validation"
    ]
    if not train_examples:
        raise ValueError("training split is empty; provide a larger or differently distributed corpus")

    step = 0
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for example in train_examples:
            step += 1
            prediction, cache = forward(
                input_weights, hidden_biases, output_weights, output_bias, example
            )
            active_hidden, passive_hidden, active_clipped, passive_clipped = cache
            diff = prediction - example.target_cp
            total_loss += abs(diff) - 0.5 if abs(diff) >= 1.0 else 0.5 * diff * diff

            grad_prediction = smooth_l1_grad(diff)
            grad_scale = grad_prediction / OUTPUT_SCALE

            grad_output_weights = np.empty((OUTPUT_INPUTS,), dtype=np.float32)
            grad_output_weights[:HIDDEN_SIZE] = active_clipped * grad_scale
            grad_output_weights[HIDDEN_SIZE:] = passive_clipped * grad_scale
            grad_output_bias = np.array([grad_prediction / OUTPUT_SCALE], dtype=np.float32)

            active_gate = ((active_hidden > 0.0) & (active_hidden < 255.0)).astype(np.float32)
            passive_gate = ((passive_hidden > 0.0) & (passive_hidden < 255.0)).astype(np.float32)
            grad_active_hidden = (
                output_weights[:HIDDEN_SIZE] * grad_scale * active_gate
            ).astype(np.float32)
            grad_passive_hidden = (
                output_weights[HIDDEN_SIZE:] * grad_scale * passive_gate
            ).astype(np.float32)
            grad_hidden_biases = grad_active_hidden + grad_passive_hidden

            row_indices = sorted(
                set(int(index) for index in example.active_features)
                | set(int(index) for index in example.passive_features)
            )
            row_index_array = np.array(row_indices, dtype=np.int32)
            grad_input_rows = np.zeros((row_index_array.size, HIDDEN_SIZE), dtype=np.float32)
            row_lookup = {feature: row for row, feature in enumerate(row_indices)}
            for feature in example.active_features:
                grad_input_rows[row_lookup[int(feature)]] += grad_active_hidden
            for feature in example.passive_features:
                grad_input_rows[row_lookup[int(feature)]] += grad_passive_hidden

            adamw_update_rows(input_weights, grad_input_rows, row_index_array, input_m1, input_m2, step)
            adamw_update_dense(hidden_biases, grad_hidden_biases, hidden_m1, hidden_m2, step)
            adamw_update_dense(output_weights, grad_output_weights, output_m1, output_m2, step)
            adamw_update_dense(output_bias, grad_output_bias, bias_m1, bias_m2, step)

        validation_loss = evaluate(validation_examples, input_weights, hidden_biases, output_weights, output_bias)
        print(
            f"epoch {epoch + 1}/{EPOCHS}: train_loss {total_loss / len(train_examples):.4f} "
            f"val_loss {validation_loss:.4f} train_examples {len(train_examples)} "
            f"validation_examples {len(validation_examples)}"
        )

    return (
        input_weights,
        hidden_biases,
        output_weights,
        output_bias,
        len(train_examples),
        len(validation_examples),
    )


def evaluate(
    examples: list[Example],
    input_weights: np.ndarray,
    hidden_biases: np.ndarray,
    output_weights: np.ndarray,
    output_bias: np.ndarray,
) -> float:
    if not examples:
        return 0.0
    total = 0.0
    for example in examples:
        prediction, _ = forward(input_weights, hidden_biases, output_weights, output_bias, example)
        diff = prediction - example.target_cp
        total += abs(diff) - 0.5 if abs(diff) >= 1.0 else 0.5 * diff * diff
    return total / len(examples)


def write_checkpoint(
    checkpoint_dir: Path,
    manifest: dict,
    input_weights: np.ndarray,
    hidden_biases: np.ndarray,
    output_weights: np.ndarray,
    output_bias: np.ndarray,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    input_weights.astype("<f4", copy=False).tofile(checkpoint_dir / INPUT_WEIGHTS_FILE)
    hidden_biases.astype("<f4", copy=False).tofile(checkpoint_dir / HIDDEN_BIASES_FILE)
    output_weights.astype("<f4", copy=False).tofile(checkpoint_dir / OUTPUT_WEIGHTS_FILE)
    output_bias.astype("<f4", copy=False).tofile(checkpoint_dir / OUTPUT_BIAS_FILE)


def main() -> int:
    args = parse_args()
    examples_path = Path(args.examples)
    checkpoint_dir = Path(args.checkpoint_dir)

    example_manifest, examples = read_examples(examples_path)
    (
        input_weights,
        hidden_biases,
        output_weights,
        output_bias,
        train_examples,
        validation_examples,
    ) = train(examples)

    manifest = {
        "magic": CHECKPOINT_MAGIC,
        "version": CHECKPOINT_VERSION,
        "trainer_version": TRAINER_VERSION,
        "source_engine_commit": example_manifest["source_engine_commit"],
        "topology_name": TOPOLOGY_NAME,
        "topology_id": TOPOLOGY_ID,
        "feature_count": FEATURE_COUNT,
        "hidden_size": HIDDEN_SIZE,
        "output_inputs": OUTPUT_INPUTS,
        "output_scale": OUTPUT_SCALE,
        "seed": SEED,
        "optimizer": OPTIMIZER,
        "loss": LOSS,
        "split_rule": SPLIT_RULE,
        "normalized_fen_rule": NORMALIZED_FEN_RULE,
        "examples_path": str(examples_path),
        "example_manifest": example_manifest,
        "train_examples": train_examples,
        "validation_examples": validation_examples,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "input_weights_file": INPUT_WEIGHTS_FILE,
        "hidden_biases_file": HIDDEN_BIASES_FILE,
        "output_weights_file": OUTPUT_WEIGHTS_FILE,
        "output_bias_file": OUTPUT_BIAS_FILE,
    }
    write_checkpoint(
        checkpoint_dir,
        manifest,
        input_weights,
        hidden_biases,
        output_weights,
        output_bias,
    )
    print(
        f"wrote checkpoint to {checkpoint_dir} with train_examples {train_examples} "
        f"validation_examples {validation_examples}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
