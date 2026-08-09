use std::path::Path;

/// Validates an operator-supplied identity before a build script places it in
/// both a Cargo directive and the engine's line-oriented UCI identification.
pub fn source_id(value: &str) -> Result<&str, &'static str> {
    if value.is_empty() {
        return Err("source identity is empty");
    }
    if value.len() > 128 {
        return Err("source identity is longer than 128 bytes");
    }
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || b"-._+".contains(&byte))
    {
        return Err("source identity contains a character outside [A-Za-z0-9._+-]");
    }
    Ok(value)
}

/// Converts a path to text that is safe to interpolate into a Cargo directive.
///
/// Cargo build-script directives are line-oriented, so accepting CR or LF here
/// would allow a crafted path to emit an additional directive.
pub fn cargo_directive_path(path: &Path) -> Result<&str, &'static str> {
    let text = path.to_str().ok_or("path is not valid UTF-8")?;
    if text.contains(['\r', '\n']) {
        return Err("path contains a forbidden newline");
    }
    Ok(text)
}
