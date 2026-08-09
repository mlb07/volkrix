#[path = "../build_support.rs"]
mod build_support;

use std::path::Path;

#[test]
fn cargo_directive_paths_reject_line_injection() {
    assert_eq!(
        build_support::cargo_directive_path(Path::new("network.nnue")),
        Ok("network.nnue")
    );
    assert_eq!(
        build_support::cargo_directive_path(Path::new("bad\nnetwork.nnue")),
        Err("path contains a forbidden newline")
    );
    assert_eq!(
        build_support::cargo_directive_path(Path::new("bad\rnetwork.nnue")),
        Err("path contains a forbidden newline")
    );
}

#[test]
fn source_identity_is_bounded_and_line_safe() {
    assert_eq!(
        build_support::source_id("52ac16d-dirty-0123456789ab"),
        Ok("52ac16d-dirty-0123456789ab")
    );
    assert_eq!(
        build_support::source_id(""),
        Err("source identity is empty")
    );
    assert_eq!(
        build_support::source_id("commit\nuciok"),
        Err("source identity contains a character outside [A-Za-z0-9._+-]")
    );
    assert_eq!(
        build_support::source_id("commit with spaces"),
        Err("source identity contains a character outside [A-Za-z0-9._+-]")
    );
    assert_eq!(
        build_support::source_id(&"x".repeat(129)),
        Err("source identity is longer than 128 bytes")
    );
}
