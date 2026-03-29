fn main() {
    println!("cargo:rerun-if-changed=vendor/fathom/src/tbprobe.c");
    println!("cargo:rerun-if-changed=vendor/fathom/src/tbprobe.h");
    println!("cargo:rerun-if-changed=vendor/fathom/src/tbconfig.h");
    println!("cargo:rerun-if-changed=vendor/fathom/src/stdendian.h");
    println!("cargo:rerun-if-changed=vendor/fathom/src/tbchess.c");

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
