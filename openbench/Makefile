EXE ?= Volkrix
# OpenBench uses the same binary for SPRT and SPSA. Tunables preserve the
# production defaults exactly; set TUNE=0 only for an isolated release smoke.
TUNE ?= 1
ifeq ($(strip $(TUNE)),)
$(error TUNE must be exactly 0 or 1)
endif
ifneq ($(words $(strip $(TUNE))),1)
$(error TUNE must be exactly 0 or 1)
endif
ifeq ($(filter 0 1,$(strip $(TUNE))),)
$(error TUNE must be exactly 0 or 1)
endif
empty :=
space := $(empty) $(empty)
comma := ,

FEATURES :=
EMBED_CHECK := :
ifneq ($(strip $(EVALFILE)),)
FEATURES += embedded-nnue
EMBED_ENV := VOLKRIX_EMBEDDED_NNUE="$(abspath $(EVALFILE))"
EMBED_CHECK := test -f "$(abspath $(EVALFILE))" -a -r "$(abspath $(EVALFILE))" || { echo "EVALFILE must name a readable file: $(abspath $(EVALFILE))" >&2; exit 2; }
endif
ifeq ($(strip $(TUNE)),1)
FEATURES += spsa-tuning
endif

FEATURE_ARGS := $(if $(strip $(FEATURES)),--features $(subst $(space),$(comma),$(strip $(FEATURES))),)

ifeq ($(OS),Windows_NT)
BIN_EXT := .exe
endif

ifeq ($(filter %.exe %.EXE,$(EXE)),)
OUTPUT_EXE := $(EXE)$(BIN_EXT)
else
OUTPUT_EXE := $(EXE)
endif

.PHONY: all clean

all:
	@$(EMBED_CHECK)
	$(EMBED_ENV) cargo build --manifest-path ../Cargo.toml --locked --release --package volkrix --bin volkrix $(FEATURE_ARGS)
	cp "../target/release/volkrix$(BIN_EXT)" "$(OUTPUT_EXE)"

clean:
	cargo clean --manifest-path ../Cargo.toml
