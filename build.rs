use sha2::{Digest, Sha256};

mod build_support;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(volkrix_embedded_nnue)");
    println!("cargo:rerun-if-changed=.git/HEAD");
    configure_git_reruns();
    println!("cargo:rerun-if-env-changed=VOLKRIX_SOURCE_ID");
    println!("cargo:rerun-if-changed=vendor/fathom/src/tbprobe.c");
    println!("cargo:rerun-if-changed=vendor/fathom/src/tbprobe.h");
    println!("cargo:rerun-if-changed=vendor/fathom/src/tbconfig.h");
    println!("cargo:rerun-if-changed=vendor/fathom/src/stdendian.h");
    println!("cargo:rerun-if-changed=vendor/fathom/src/tbchess.c");
    let source_id = configured_source_id()
        .or_else(source_commit)
        .unwrap_or_else(|| "unknown".to_owned());
    println!("cargo:rustc-env=VOLKRIX_SOURCE_COMMIT={source_id}");
    configure_embedded_nnue();

    let mut build = cc::Build::new();
    build
        .file("vendor/fathom/src/tbprobe.c")
        .include("vendor/fathom/src")
        .warnings(false);

    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        build.flag_if_supported("/std:c11");
        build.flag_if_supported("/experimental:c11atomics");
    } else {
        build.flag_if_supported("-std=gnu11");
    }

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        build.flag("-mmacosx-version-min=11.0.0");
    }

    build.compile("fathom");
}

fn configure_git_reruns() {
    // `.git/HEAD` normally contains only a stable symbolic-ref name. Watching
    // the resolved loose ref prevents incremental builds after a commit from
    // retaining the previous commit identity.
    let Ok(symbolic_ref) = std::process::Command::new("git")
        .args(["symbolic-ref", "--quiet", "HEAD"])
        .output()
    else {
        return;
    };
    if !symbolic_ref.status.success() {
        return;
    }
    let Ok(reference) = String::from_utf8(symbolic_ref.stdout) else {
        return;
    };
    let reference = reference.trim();
    if reference.is_empty() {
        return;
    }
    let Ok(path) = std::process::Command::new("git")
        .args(["rev-parse", "--git-path", reference])
        .output()
    else {
        return;
    };
    if !path.status.success() {
        return;
    }
    let Ok(path) = String::from_utf8(path.stdout) else {
        return;
    };
    let path = path.trim();
    if !path.is_empty() {
        let path = build_support::cargo_directive_path(std::path::Path::new(path)).unwrap_or_else(
            |reason| panic!("Git ref path is unsafe for Cargo directives: {reason}"),
        );
        println!("cargo:rerun-if-changed={path}");
    }
}

fn configured_source_id() -> Option<String> {
    let value = match std::env::var("VOLKRIX_SOURCE_ID") {
        Ok(value) => value,
        Err(std::env::VarError::NotPresent) => return None,
        Err(std::env::VarError::NotUnicode(_)) => {
            panic!("VOLKRIX_SOURCE_ID must be valid UTF-8")
        }
    };
    Some(
        build_support::source_id(&value)
            .unwrap_or_else(|reason| panic!("VOLKRIX_SOURCE_ID is invalid: {reason}"))
            .to_owned(),
    )
}

fn configure_embedded_nnue() {
    println!("cargo:rerun-if-env-changed=VOLKRIX_EMBEDDED_NNUE");
    println!("cargo:rerun-if-env-changed=VOLKRIX_EMBEDDED_NNUE_SHA256");

    if std::env::var_os("CARGO_FEATURE_EMBEDDED_NNUE").is_none() {
        return;
    }
    let Some(path) = std::env::var_os("VOLKRIX_EMBEDDED_NNUE") else {
        // The feature denotes build capability. Keeping all-features builds
        // usable without silently embedding an arbitrary local file is useful
        // for CI; the OpenBench Makefile requires EVALFILE separately.
        return;
    };
    let canonical = std::fs::canonicalize(&path).unwrap_or_else(|error| {
        panic!(
            "VOLKRIX_EMBEDDED_NNUE '{}' could not be resolved: {error}",
            std::path::Path::new(&path).display()
        )
    });
    let canonical_text = build_support::cargo_directive_path(&canonical).unwrap_or_else(|reason| {
        panic!(
            "VOLKRIX_EMBEDDED_NNUE path is unsafe for Cargo directives ({reason}): '{}'",
            canonical.display(),
        )
    });
    let bytes = std::fs::read(&canonical).unwrap_or_else(|error| {
        panic!(
            "VOLKRIX_EMBEDDED_NNUE '{}' could not be read: {error}",
            canonical.display()
        )
    });
    assert!(!bytes.is_empty(), "VOLKRIX_EMBEDDED_NNUE must not be empty");
    let digest = format!("{:x}", Sha256::digest(&bytes));
    let expected = std::env::var_os("VOLKRIX_EMBEDDED_NNUE_SHA256").or_else(|| {
        let name = canonical.file_name()?.to_str()?;
        (name.len() == 64 && name.bytes().all(|byte| byte.is_ascii_hexdigit())).then(|| name.into())
    });
    if let Some(expected) = expected {
        let expected = expected.to_string_lossy().trim().to_ascii_lowercase();
        assert_eq!(
            digest,
            expected,
            "VOLKRIX_EMBEDDED_NNUE_SHA256 did not match '{}'",
            canonical.display()
        );
    }

    println!("cargo:rerun-if-changed={canonical_text}");
    println!("cargo:rustc-cfg=volkrix_embedded_nnue");
    println!("cargo:rustc-env=VOLKRIX_EMBEDDED_NNUE={}", canonical_text);
    println!("cargo:rustc-env=VOLKRIX_EMBEDDED_NNUE_SHA256={digest}");
    println!("cargo:rustc-env=VOLKRIX_EMBEDDED_NNUE_SIZE={}", bytes.len());
}

fn source_commit() -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let commit = String::from_utf8(output.stdout).ok()?;
    let commit = commit.trim();
    if commit.is_empty() {
        None
    } else {
        Some(commit.to_owned())
    }
}
