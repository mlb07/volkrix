use std::io::{self, BufRead, Write};

use crate::{ENGINE_AUTHOR, ENGINE_NAME, core::Position};

pub struct UciResponse {
    pub lines: Vec<String>,
    pub should_quit: bool,
}

pub struct UciEngine {
    position: Position,
}

impl UciEngine {
    pub fn new() -> Self {
        Self {
            position: Position::startpos(),
        }
    }

    pub fn position(&self) -> &Position {
        &self.position
    }

    pub fn handle_line(&mut self, line: &str) -> UciResponse {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return UciResponse {
                lines: Vec::new(),
                should_quit: false,
            };
        }

        let tokens: Vec<&str> = trimmed.split_whitespace().collect();
        match tokens[0] {
            "uci" => UciResponse {
                lines: vec![
                    format!("id name {ENGINE_NAME}"),
                    format!("id author {ENGINE_AUTHOR}"),
                    "uciok".to_owned(),
                ],
                should_quit: false,
            },
            "isready" => UciResponse {
                lines: vec!["readyok".to_owned()],
                should_quit: false,
            },
            "ucinewgame" => {
                self.position = Position::startpos();
                UciResponse {
                    lines: Vec::new(),
                    should_quit: false,
                }
            }
            "position" => UciResponse {
                lines: self.handle_position(&tokens),
                should_quit: false,
            },
            "go" => UciResponse {
                lines: self.handle_go(&tokens),
                should_quit: false,
            },
            "stop" => UciResponse {
                lines: Vec::new(),
                should_quit: false,
            },
            "quit" => UciResponse {
                lines: Vec::new(),
                should_quit: true,
            },
            _ => UciResponse {
                lines: vec![format!(
                    "info string error: unsupported command '{trimmed}'"
                )],
                should_quit: false,
            },
        }
    }

    fn handle_position(&mut self, tokens: &[&str]) -> Vec<String> {
        if tokens.len() < 2 {
            return vec!["info string error: position requires startpos or fen".to_owned()];
        }

        let mut cursor = 1usize;
        let mut next_position = match tokens[cursor] {
            "startpos" => {
                cursor += 1;
                Position::startpos()
            }
            "fen" => {
                cursor += 1;
                if tokens.len() < cursor + 6 {
                    return vec![
                        "info string error: position fen requires 6 FEN fields".to_owned(),
                    ];
                }
                let fen = tokens[cursor..cursor + 6].join(" ");
                cursor += 6;
                match Position::from_fen(&fen) {
                    Ok(position) => position,
                    Err(error) => {
                        return vec![format!("info string error: {error}")];
                    }
                }
            }
            other => {
                return vec![format!(
                    "info string error: unsupported position source '{other}'"
                )];
            }
        };

        if cursor < tokens.len() {
            if tokens[cursor] != "moves" {
                return vec![format!(
                    "info string error: expected 'moves' after position source, found '{}'",
                    tokens[cursor]
                )];
            }
            cursor += 1;
            for move_text in &tokens[cursor..] {
                if let Err(error) = next_position.apply_uci_move(move_text) {
                    return vec![format!("info string error: {error}")];
                }
            }
        }

        self.position = next_position;
        Vec::new()
    }

    fn handle_go(&mut self, tokens: &[&str]) -> Vec<String> {
        let mut errors = Vec::new();
        let mut index = 1usize;
        while index < tokens.len() {
            if tokens[index] == "depth" {
                if index + 1 >= tokens.len() {
                    errors.push("info string error: go depth requires a value".to_owned());
                    break;
                }
                if tokens[index + 1].parse::<u32>().is_err() {
                    errors.push(format!(
                        "info string error: invalid go depth value '{}'",
                        tokens[index + 1]
                    ));
                }
                index += 2;
                continue;
            }
            index += 1;
        }

        let bestmove = self
            .position
            .select_placeholder_bestmove()
            .map_or_else(|| "0000".to_owned(), |mv| mv.to_string());
        errors.push(format!("bestmove {bestmove}"));
        errors
    }
}

impl Default for UciEngine {
    fn default() -> Self {
        Self::new()
    }
}

pub fn run_stdio() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut engine = UciEngine::new();
    let mut output = io::BufWriter::new(stdout.lock());

    for line_result in stdin.lock().lines() {
        let line = line_result?;
        let response = engine.handle_line(&line);
        for output_line in response.lines {
            writeln!(output, "{output_line}")?;
        }
        output.flush()?;
        if response.should_quit {
            break;
        }
    }

    Ok(())
}
