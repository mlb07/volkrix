fn main() -> std::io::Result<()> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        None => volkrix::uci::run_stdio(),
        Some("bench") => {
            let config = if matches!(args.next().as_deref(), Some("--no-tt")) {
                volkrix::search::BenchConfig::default().without_tt()
            } else {
                volkrix::search::BenchConfig::default()
            };
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
