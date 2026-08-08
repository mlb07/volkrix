use nnue_rs::{Error, Network};

const SFNNV16_PP_3WIDE_VERSION: u32 = 0x6A448AFA;

fn header(version: u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&version.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes()); // architecture hash
    bytes.extend_from_slice(&0u32.to_le_bytes()); // description length
    bytes
}

#[test]
fn identifies_sfnnv16_pp_3wide_instead_of_guessing_its_layout() {
    let error = match Network::from_bytes(&header(SFNNV16_PP_3WIDE_VERSION)) {
        Ok(_) => panic!("SFNNv16 PP_3Wide must not be loaded as an older architecture"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        Error::UnsupportedPp3Wide(SFNNV16_PP_3WIDE_VERSION)
    ));
    let message = error.to_string();
    assert!(message.contains("SFNNv16/current-development PP_3Wide"));
    assert!(message.contains("stable SFNNv10 networks remain supported"));
}

#[test]
fn unrelated_unknown_versions_keep_the_generic_diagnostic() {
    let version = 0xDEADBEEF;
    let error = match Network::from_bytes(&header(version)) {
        Ok(_) => panic!("unknown network version must be rejected"),
        Err(error) => error,
    };

    assert!(matches!(error, Error::UnsupportedVersion(0xDEADBEEF)));
    assert_eq!(error.to_string(), "unsupported NNUE version: 0xdeadbeef");
}
