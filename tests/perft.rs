use std::{fs, path::Path};

use volkrix::core::{Position, divide, perft};

struct PerftCase {
    name: &'static str,
    fen: &'static str,
    depths: &'static [(u8, u64)],
}

const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
const POSITION_3: &str = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
const POSITION_4: &str = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";
const POSITION_5: &str = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";
const POSITION_6: &str = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10";

const FAST_CASES: &[PerftCase] = &[
    PerftCase {
        name: "startpos",
        fen: STARTPOS,
        depths: &[(1, 20), (2, 400), (3, 8902), (4, 197281)],
    },
    PerftCase {
        name: "kiwipete",
        fen: KIWIPETE,
        depths: &[(1, 48), (2, 2039), (3, 97862), (4, 4085603)],
    },
    PerftCase {
        name: "position3",
        fen: POSITION_3,
        depths: &[(1, 14), (2, 191), (3, 2812), (4, 43238)],
    },
    PerftCase {
        name: "position4",
        fen: POSITION_4,
        depths: &[(1, 6), (2, 264), (3, 9467), (4, 422333)],
    },
    PerftCase {
        name: "position5",
        fen: POSITION_5,
        depths: &[(1, 44), (2, 1486), (3, 62379), (4, 2103487)],
    },
    PerftCase {
        name: "position6",
        fen: POSITION_6,
        depths: &[(1, 46), (2, 2079), (3, 89890), (4, 3894594)],
    },
];

const LONG_CASES: &[PerftCase] = &[
    PerftCase {
        name: "startpos",
        fen: STARTPOS,
        depths: &[(5, 4865609), (6, 119060324)],
    },
    PerftCase {
        name: "kiwipete",
        fen: KIWIPETE,
        depths: &[(5, 193690690)],
    },
    PerftCase {
        name: "position3",
        fen: POSITION_3,
        depths: &[(5, 674624), (6, 11030083)],
    },
    PerftCase {
        name: "position4",
        fen: POSITION_4,
        depths: &[(5, 15833292)],
    },
];

#[test]
fn perft_reference_fast_suite() {
    run_cases(FAST_CASES);
}

#[test]
#[ignore = "long reference suite; run with cargo test --release --test perft perft_reference_long_suite -- --ignored"]
fn perft_reference_long_suite() {
    run_cases(LONG_CASES);
}

#[test]
fn divide_artifacts_match_verified_outputs() {
    assert_divide_fixture_matches(STARTPOS, 3, "tests/fixtures/divide/startpos_d3.txt");
    assert_divide_fixture_matches(KIWIPETE, 3, "tests/fixtures/divide/kiwipete_d3.txt");
}

fn run_cases(cases: &[PerftCase]) {
    for case in cases {
        let mut position = Position::from_fen(case.fen).expect("FEN parse must succeed");
        for &(depth, expected) in case.depths {
            let actual = perft(&mut position, depth);
            println!("{} depth {} => {}", case.name, depth, actual);
            assert_eq!(
                actual, expected,
                "perft mismatch for {} at depth {}",
                case.name, depth
            );
        }
    }
}

fn assert_divide_fixture_matches(fen: &str, depth: u8, fixture_path: &str) {
    let mut position = Position::from_fen(fen).expect("FEN parse must succeed");
    let actual = divide(&mut position, depth);
    let expected = parse_divide_fixture(Path::new(fixture_path));

    let actual_as_text: Vec<(String, u64)> = actual
        .into_iter()
        .map(|(mv, nodes)| (mv.to_string(), nodes))
        .collect();

    assert_eq!(actual_as_text, expected);
}

fn parse_divide_fixture(path: &Path) -> Vec<(String, u64)> {
    fs::read_to_string(path)
        .expect("fixture file must exist")
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let (move_text, count_text) = line
                .split_once(' ')
                .expect("fixture lines must contain move and count");
            assert!(
                (4..=5).contains(&move_text.len()),
                "fixture move must be a valid UCI move token: {move_text}"
            );
            let count = count_text
                .parse::<u64>()
                .expect("fixture count must be a valid integer");
            (move_text.to_owned(), count)
        })
        .collect()
}
