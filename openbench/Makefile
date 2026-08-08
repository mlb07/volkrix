EXE ?= Volkrix

ifneq ($(strip $(EVALFILE)),)
$(error Volkrix does not yet embed OpenBench EVALFILE networks; use the classical template)
endif

ifeq ($(OS),Windows_NT)
BIN_EXT := .exe
endif

.PHONY: all clean

all:
	cargo build --manifest-path ../Cargo.toml --locked --release --package volkrix --bin volkrix
	cp "../target/release/volkrix$(BIN_EXT)" "$(EXE)$(BIN_EXT)"

clean:
	cargo clean --manifest-path ../Cargo.toml
