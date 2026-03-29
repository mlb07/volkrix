fn main() {
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=vendor/fathom/src/tbprobe.c");
    println!("cargo:rerun-if-changed=vendor/fathom/src/tbprobe.h");
    println!("cargo:rerun-if-changed=vendor/fathom/src/tbconfig.h");
    println!("cargo:rerun-if-changed=vendor/fathom/src/stdendian.h");
    println!("cargo:rerun-if-changed=vendor/fathom/src/tbchess.c");
    println!(
        "cargo:rustc-env=VOLKRIX_SOURCE_COMMIT={}",
        source_commit().unwrap_or_else(|| "unknown".to_owned())
    );

    let mut build = cc::Build::new();
    build
        .file("vendor/fathom/src/tbprobe.c")
        .include("vendor/fathom/src")
        .flag_if_supported("-std=gnu99")
        .warnings(false);

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        build.flag("-mmacosx-version-min=11.0.0");
    }

    build.compile("fathom");
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
