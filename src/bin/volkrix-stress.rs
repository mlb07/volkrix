use volkrix::stress::{StressConfig, run};

fn main() {
    let config = parse_args(std::env::args().skip(1)).unwrap_or_else(|error| {
        eprintln!("error: {error}");
        eprintln!(
            "usage: volkrix-stress [--seed U64|0xHEX] [--walks N] [--plies N] [--parser-cases N]"
        );
        std::process::exit(2);
    });
    match run(config) {
        Ok(report) => println!("{report}"),
        Err(error) => {
            eprintln!("stress failure: {error}");
            std::process::exit(1);
        }
    }
}

fn parse_args(mut args: impl Iterator<Item = String>) -> Result<StressConfig, String> {
    let mut config = StressConfig::default();
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--seed" => config.seed = parse_u64(required_value(&mut args, "--seed")?, "--seed")?,
            "--walks" => {
                config.walks = parse_u32(required_value(&mut args, "--walks")?, "--walks")?
            }
            "--plies" => {
                config.plies = parse_u32(required_value(&mut args, "--plies")?, "--plies")?
            }
            "--parser-cases" => {
                config.parser_cases = parse_u32(
                    required_value(&mut args, "--parser-cases")?,
                    "--parser-cases",
                )?
            }
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    Ok(config)
}

fn required_value(args: &mut impl Iterator<Item = String>, option: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value for {option}"))
}

fn parse_u32(value: String, option: &str) -> Result<u32, String> {
    value
        .parse::<u32>()
        .map_err(|_| format!("invalid {option} value '{value}'"))
}

fn parse_u64(value: String, option: &str) -> Result<u64, String> {
    let parsed = value
        .strip_prefix("0x")
        .map_or_else(|| value.parse::<u64>(), |hex| u64::from_str_radix(hex, 16));
    parsed.map_err(|_| format!("invalid {option} value '{value}'"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arguments_accept_decimal_and_hex_seeds() {
        let decimal = parse_args(["--seed", "42"].into_iter().map(str::to_owned))
            .expect("decimal seed must parse");
        let hexadecimal = parse_args(["--seed", "0x2a"].into_iter().map(str::to_owned))
            .expect("hexadecimal seed must parse");
        assert_eq!(decimal.seed, 42);
        assert_eq!(hexadecimal.seed, 42);
    }

    #[test]
    fn arguments_reject_unknown_or_missing_values() {
        assert!(parse_args(["--unknown"].into_iter().map(str::to_owned)).is_err());
        assert!(parse_args(["--walks"].into_iter().map(str::to_owned)).is_err());
    }
}
