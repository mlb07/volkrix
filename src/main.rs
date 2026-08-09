fn main() -> std::io::Result<()> {
    eprintln!("{} {}", volkrix::ENGINE_NAME, volkrix::VERSION);

    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        None => volkrix::uci::run_stdio(),
        Some("bench") => {
            let config = parse_bench_args(args).unwrap_or_else(|error| {
                eprintln!("error: {error}");
                eprintln!(
                    "usage: volkrix bench [--depth N] [--threads N] [--hash-mb N]\n\
                     [--evalfile classical|/absolute/path/to/network.nnue]\n\
                     [--small-evalfile /absolute/path/to/small.nnue]\n\
                     [--dual-policy off|small-fallback] [--dual-threshold CP] [--no-tt]"
                );
                std::process::exit(2);
            });
            for line in volkrix::search::run_bench(config).render_lines() {
                println!("{line}");
            }
            Ok(())
        }
        Some(other) => {
            eprintln!("unsupported command '{other}'");
            std::process::exit(1);
        }
    }
}

fn parse_bench_args(
    mut args: impl Iterator<Item = String>,
) -> Result<volkrix::search::BenchConfig, String> {
    let mut config = volkrix::search::BenchConfig::default().with_classical_eval();
    #[cfg(volkrix_embedded_nnue)]
    {
        config = config.with_discovered_eval();
    }
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--no-tt" => config = config.without_tt(),
            "--depth" => {
                let value = required_bench_value(&mut args, "--depth")?;
                let depth = value
                    .parse::<u8>()
                    .map_err(|_| format!("invalid --depth value '{value}'"))?;
                if depth == 0 {
                    return Err("--depth must be at least 1".to_owned());
                }
                config.depth = depth;
            }
            "--threads" => {
                let value = required_bench_value(&mut args, "--threads")?;
                let threads = value
                    .parse::<usize>()
                    .map_err(|_| format!("invalid --threads value '{value}'"))?;
                if threads == 0 {
                    return Err("--threads must be at least 1".to_owned());
                }
                config = config.with_threads(threads);
            }
            "--hash-mb" => {
                let value = required_bench_value(&mut args, "--hash-mb")?;
                let hash_mb = value
                    .parse::<usize>()
                    .map_err(|_| format!("invalid --hash-mb value '{value}'"))?;
                if hash_mb == 0 {
                    return Err("--hash-mb must be at least 1".to_owned());
                }
                config = config.with_hash_mb(hash_mb);
            }
            "--evalfile" => {
                let value = required_bench_value(&mut args, "--evalfile")?;
                if value == "classical" {
                    config = config.with_classical_eval();
                } else if value == "embedded" {
                    #[cfg(volkrix_embedded_nnue)]
                    {
                        config = config.with_discovered_eval();
                    }
                    #[cfg(not(volkrix_embedded_nnue))]
                    return Err("--evalfile embedded requires an embedded-network build".to_owned());
                } else {
                    let path = std::path::Path::new(&value);
                    if !path.is_absolute() {
                        return Err(format!(
                            "--evalfile must be 'classical' or an absolute path, got '{value}'"
                        ));
                    }
                    if !path.is_file() {
                        return Err(format!("--evalfile does not exist: '{value}'"));
                    }
                    config = config.with_eval_file(value);
                }
            }
            "--small-evalfile" => {
                let value = required_bench_value(&mut args, "--small-evalfile")?;
                let path = std::path::Path::new(&value);
                if !path.is_absolute() || !path.is_file() {
                    return Err(format!(
                        "--small-evalfile must be an existing absolute path, got '{value}'"
                    ));
                }
                config = config.with_small_eval_file(value);
            }
            "--dual-policy" => {
                let value = required_bench_value(&mut args, "--dual-policy")?;
                match value.as_str() {
                    "off" => {}
                    "small-fallback" => {
                        config = config.enable_dual_small_fallback();
                    }
                    _ => {
                        return Err(format!(
                            "invalid --dual-policy '{value}'; expected off or small-fallback"
                        ));
                    }
                }
            }
            "--dual-threshold" => {
                let value = required_bench_value(&mut args, "--dual-threshold")?;
                let threshold = value
                    .parse::<i32>()
                    .map_err(|_| format!("invalid --dual-threshold value '{value}'"))?;
                if !(0..=volkrix::search::MAX_DUAL_EVAL_THRESHOLD).contains(&threshold) {
                    return Err(format!(
                        "--dual-threshold must be between 0 and {}",
                        volkrix::search::MAX_DUAL_EVAL_THRESHOLD
                    ));
                }
                config = config.with_dual_eval_threshold(threshold);
            }
            _ => return Err(format!("unknown bench argument '{argument}'")),
        }
    }
    Ok(config)
}

fn required_bench_value(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value for {option}"))
}
