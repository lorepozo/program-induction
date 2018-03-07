use std::collections::VecDeque;
use std::{error, fmt};
use super::{Expression, Language};

#[derive(Debug, Clone)]
pub struct ParseError {
    /// The location of the original string where parsing failed.
    pub location: usize,
    /// A message associated with the parse error.
    pub msg: &'static str,
}
impl ParseError {
    fn new(location: usize, msg: &'static str) -> Self {
        Self { location, msg }
    }
}
impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "{} at index {}", self.msg, self.location)
    }
}
impl error::Error for ParseError {
    fn description(&self) -> &str {
        "could not parse expression"
    }
}

pub fn parse(dsl: &Language, inp: &str) -> Result<Expression, ParseError> {
    let s = inp.trim_left();
    let offset = inp.len() - s.len();
    streaming_parse(dsl, s, offset).and_then(move |(di, expr)| {
        if s[di..].chars().all(char::is_whitespace) {
            Ok(expr)
        } else {
            Err(ParseError::new(
                offset + di,
                "expected end of expression, found more tokens",
            ))
        }
    })
}

/// inp must not have leading whitespace. Does not invent.
fn streaming_parse(
    dsl: &Language,
    inp: &str,
    offset: usize, // for good error messages
) -> Result<(usize, Expression), ParseError> {
    let init: Option<Result<(usize, Expression), ParseError>> = None;

    let abstraction = || {
        inp.find('(')
            .and_then(|i| {
                if inp[..i].chars().all(char::is_whitespace) {
                    Some(i + 1)
                } else {
                    None
                }
            })
            .and_then(|di| match inp[di..].find(char::is_whitespace) {
                Some(ndi) if &inp[di..di + ndi] == "lambda" || &inp[di..di + ndi] == "λ" => {
                    Some(di + ndi)
                }
                _ => None,
            })
            .map(|mut di| {
                // skip spaces
                di += inp[di..].chars().take_while(|c| c.is_whitespace()).count();
                // parse body
                let (ndi, body) = streaming_parse(dsl, &inp[di..], offset + di)?;
                di += ndi;
                // check if complete
                inp[di..]
                    .chars()
                    .nth(0)
                    .and_then(|c| if c == ')' { Some(di + 1) } else { None })
                    .ok_or_else(|| ParseError::new(offset + di, "incomplete application"))
                    .map(|di| (di, Expression::Abstraction(Box::new(body))))
            })
    };
    let application = || {
        inp.find('(')
            .and_then(|i| {
                if inp[..i].chars().all(char::is_whitespace) {
                    Some(i + 1)
                } else {
                    None
                }
            })
            .map(|mut di| {
                let mut items = VecDeque::new();
                loop {
                    // parse expr
                    let (ndi, expr) = streaming_parse(dsl, &inp[di..], offset + di)?;
                    items.push_back(expr);
                    di += ndi;
                    // skip spaces
                    di += inp[di..].chars().take_while(|c| c.is_whitespace()).count();
                    // check if complete
                    match inp[di..].chars().nth(0) {
                        None => break Err(ParseError::new(offset + di, "incomplete application")),
                        Some(')') => {
                            di += 1;
                            break if let Some(init) = items.pop_front() {
                                let app = items.into_iter().fold(init, |a, v| {
                                    Expression::Application(Box::new(a), Box::new(v))
                                });
                                Ok((di, app))
                            } else {
                                Err(ParseError::new(offset + di, "empty application"))
                            };
                        }
                        _ => (),
                    }
                }
            })
    };
    let index = || {
        if inp.chars().nth(0) == Some('$') && inp.len() > 1 {
            inp[1..]
                .find(|c: char| c.is_whitespace() || c == ')')
                .and_then(|i| inp[1..1 + i].parse::<usize>().ok().map(|num| (1 + i, num)))
                .map(|(di, num)| Ok((di, Expression::Index(num))))
        } else {
            None
        }
    };
    let invented = || {
        if inp.chars().take(2).collect::<String>() == "#(" {
            Some(1)
        } else {
            None
        }.map(|mut di| {
            let (ndi, expr) = streaming_parse(dsl, &inp[di..], offset + di)?;
            di += ndi;
            if let Some(num) = dsl.invented.iter().position(|&(ref x, _, _)| x == &expr) {
                Ok((di, Expression::Invented(num)))
            } else {
                Err(ParseError::new(
                    offset + di,
                    "invented expr is unfamiliar to context",
                ))
            }
        })
    };
    let primitive = || {
        match inp.find(|c: char| c.is_whitespace() || c == ')') {
            None if !inp.is_empty() => Some(inp.len()),
            Some(next) if next > 0 => Some(next),
            _ => None,
        }.map(|di| {
            if let Some(num) = dsl.primitives
                .iter()
                .position(|&(ref name, _, _)| name == &inp[..di])
            {
                Ok((di, Expression::Primitive(num)))
            } else {
                Err(ParseError::new(offset + di, "unexpected end of expression"))
            }
        })
    };
    // These parsers return None if the expr isn't applicable
    // or Some(Err(..)) if the expr applied but was invalid.
    // Ordering is intentional.
    init.or_else(abstraction)
        .or_else(application)
        .or_else(index)
        .or_else(invented)
        .or_else(primitive)
        .unwrap_or_else(|| {
            Err(ParseError::new(
                offset,
                "could not parse any expression variant",
            ))
        })
}
