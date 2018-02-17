use polytype::{self, Context, Type};

use std::collections::{HashMap, VecDeque};
use std::fmt;

/// A DSL is effectively a registry for primitive and invented expressions.
#[derive(Debug, Clone, PartialEq)]
pub struct DSL {
    primitives: Vec<(String, Type)>,
    invented: Vec<(Expression, Type)>,
}
impl DSL {
    pub fn new(primitives: Vec<(String, Type)>, invented: Vec<(Expression, Type)>) -> Self {
        DSL {
            primitives,
            invented,
        }
    }
    pub fn invent(&mut self, expr: Expression) -> Result<usize, InferenceError> {
        let mut ctx = Context::default();
        let env = VecDeque::new();
        let mut indices = HashMap::new();
        let tp = expr.infer_internal(&self, &mut ctx, &env, &mut indices)?;
        self.invented.push((expr, tp));
        Ok(self.invented.len() - 1)
    }
    pub fn parse(&self, inp: &str) -> Result<Expression, ParseError> {
        let s = inp.trim_left();
        let offset = inp.len() - s.len();
        Expression::parse(self, s, offset).and_then(move |(di, expr)| {
            if s[di..].chars().all(char::is_whitespace) {
                Ok(expr)
            } else {
                Err(ParseError(
                    offset + di,
                    "expected end of expression, found more tokens",
                ))
            }
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    Primitive(usize),
    Application(Box<Expression>, Box<Expression>),
    Abstraction(Box<Expression>),
    Index(u64),

    Invented(usize),
}
impl Expression {
    pub fn infer(&self, dsl: &DSL) -> Result<Type, InferenceError> {
        let mut ctx = Context::default();
        let env = VecDeque::new();
        let mut indices = HashMap::new();
        self.infer_internal(dsl, &mut ctx, &env, &mut indices)
    }
    fn infer_internal(
        &self,
        dsl: &DSL,
        mut ctx: &mut Context,
        env: &VecDeque<Type>,
        indices: &mut HashMap<u64, Type>,
    ) -> Result<Type, InferenceError> {
        match self {
            &Expression::Primitive(num) => if let Some(prim) = dsl.primitives.get(num as usize) {
                Ok(prim.1.instantiate_indep(ctx))
            } else {
                Err(InferenceError::BadPrimitive(num))
            },
            &Expression::Application(ref f, ref x) => {
                let f_tp = f.infer_internal(dsl, &mut ctx, env, indices)?;
                let x_tp = x.infer_internal(dsl, &mut ctx, env, indices)?;
                let ret_tp = ctx.new_variable();
                ctx.unify(&f_tp, &arrow![x_tp, ret_tp.clone()])?;
                Ok(ret_tp.apply(ctx))
            }
            &Expression::Abstraction(ref body) => {
                let arg_tp = ctx.new_variable();
                let mut env = env.clone();
                env.push_front(arg_tp.clone());
                let ret_tp = body.infer_internal(dsl, &mut ctx, &env, indices)?;
                Ok(arrow![arg_tp, ret_tp].apply(ctx))
            }
            &Expression::Index(i) => {
                if (i as usize) < env.len() {
                    Ok(env[i as usize].apply(ctx))
                } else {
                    Ok(indices
                        .entry(i - (env.len() as u64))
                        .or_insert_with(|| ctx.new_variable())
                        .apply(ctx))
                }
            }
            &Expression::Invented(num) => if let Some(inv) = dsl.invented.get(num as usize) {
                Ok(inv.1.instantiate_indep(ctx))
            } else {
                Err(InferenceError::BadInvented(num))
            },
        }
    }
    pub fn to_string(&self, dsl: &DSL) -> String {
        self.show(dsl, false)
    }
    fn show(&self, dsl: &DSL, is_function: bool) -> String {
        match self {
            &Expression::Primitive(num) => dsl.primitives[num as usize].0.clone(),
            &Expression::Application(ref f, ref x) => if is_function {
                format!("{} {}", f.show(dsl, true), x.show(dsl, false))
            } else {
                format!("({} {})", f.show(dsl, true), x.show(dsl, false))
            },
            &Expression::Abstraction(ref body) => format!("(λ {})", body.show(dsl, false)),
            &Expression::Index(i) => format!("${}", i),
            &Expression::Invented(num) => {
                format!("#{}", dsl.invented[num as usize].0.show(dsl, false))
            }
        }
    }
    /// inp must not have leading whitespace. Does not invent.
    fn parse(
        dsl: &DSL,
        inp: &str,
        offset: usize, // for good error messages
    ) -> Result<(usize, Expression), ParseError> {
        let init: Option<Result<(usize, Expression), ParseError>> = None;

        let primitive = || {
            match inp.find(|c: char| c.is_whitespace() || c == ')') {
                None if inp.len() > 0 => Some(inp.len()),
                Some(next) if next > 0 => Some(next),
                _ => None,
            }.map(|di| {
                if let Some(num) = dsl.primitives
                    .iter()
                    .position(|&(ref name, _)| name == &inp[..di])
                {
                    Ok((di, Expression::Primitive(num)))
                } else {
                    Err(ParseError(offset + di, "unexpected end of expression"))
                }
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
                        let (ndi, expr) = Expression::parse(dsl, &inp[di..], offset + di)?;
                        items.push_back(expr);
                        di += ndi;
                        // skip spaces
                        di += inp[di..].chars().take_while(|c| c.is_whitespace()).count();
                        // check if complete
                        match inp[di..].chars().nth(0) {
                            None => break Err(ParseError(offset + di, "incomplete application")),
                            Some(')') => {
                                di += 1;
                                break if let Some(init) = items.pop_front() {
                                    let app = items.into_iter().fold(init, |a, v| {
                                        Expression::Application(Box::new(a), Box::new(v))
                                    });
                                    Ok((di, app))
                                } else {
                                    Err(ParseError(offset + di, "empty application"))
                                };
                            }
                            _ => (),
                        }
                    }
                })
        };
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
                    let (ndi, body) = Expression::parse(dsl, &inp[di..], offset + di)?;
                    di += ndi;
                    // check if complete
                    inp[di..]
                        .chars()
                        .nth(0)
                        .and_then(|c| if c == ')' { Some(di + 1) } else { None })
                        .ok_or(ParseError(offset + di, "incomplete application"))
                        .map(|di| (di, Expression::Abstraction(Box::new(body))))
                })
        };
        let index = || {
            if inp.chars().nth(0) == Some('$') && inp.len() > 1 {
                inp[1..]
                    .find(|c: char| c.is_whitespace() || c == ')')
                    .and_then(|i| inp[1..1 + i].parse::<u64>().ok().map(|num| (1 + i, num)))
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
                let (ndi, expr) = Expression::parse(dsl, &inp[di..], offset + di)?;
                di += ndi;
                if let Some(num) = dsl.invented.iter().position(|&(ref x, _)| x == &expr) {
                    Ok((di, Expression::Invented(num)))
                } else {
                    Err(ParseError(
                        offset + di,
                        "invented expr is unfamiliar to context",
                    ))
                }
            })
        };
        // These parsers return None if the expr isn't applicable
        // or Some(Err(..)) if the expr applied but was invalid.
        // It is imperative that primitive comes last.
        init.or_else(abstraction)
            .or_else(application)
            .or_else(index)
            .or_else(invented)
            .or_else(primitive)
            .unwrap_or(Err(ParseError(
                offset,
                "could not parse any expression variant",
            )))
    }
}

#[derive(Debug, Clone)]
pub enum InferenceError {
    BadPrimitive(usize),
    BadInvented(usize),
    Unify(polytype::UnificationError),
}
impl From<polytype::UnificationError> for InferenceError {
    fn from(err: polytype::UnificationError) -> Self {
        InferenceError::Unify(err)
    }
}
impl fmt::Display for InferenceError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            &InferenceError::BadPrimitive(i) => write!(f, "invalid primitive: '{}'", i),
            &InferenceError::BadInvented(i) => write!(f, "invalid invented: '{}'", i),
            &InferenceError::Unify(ref err) => write!(f, "could not unify to infer type: {}", err),
        }
    }
}
impl ::std::error::Error for InferenceError {
    fn description(&self) -> &str {
        "could not infer type"
    }
}

#[derive(Debug, Clone)]
pub struct ParseError(usize, &'static str);
impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "{} at index {}", self.1, self.0)
    }
}
impl ::std::error::Error for ParseError {
    fn description(&self) -> &str {
        "could not parse expression"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_primitive() {
        let dsl = DSL::new(
            vec![
                (String::from("singleton"), arrow![tp!(0), tp!(list(tp!(0)))]),
            ],
            vec![],
        );
        let expr = dsl.parse("singleton").unwrap();
        assert_eq!(expr, Expression::Primitive(0));

        assert!(dsl.parse("something_else").is_err());
        assert!(dsl.parse("singleton singleton").is_err());
    }

    #[test]
    fn test_parse_application() {
        let dsl = DSL::new(
            vec![
                (String::from("singleton"), arrow![tp!(0), tp!(list(tp!(0)))]),
                (String::from("thing"), arrow![tp!(int), tp!(int)]),
            ],
            vec![],
        );
        assert_eq!(
            dsl.parse("(singleton singleton)").unwrap(),
            Expression::Application(
                Box::new(Expression::Primitive(0)),
                Box::new(Expression::Primitive(0)),
            )
        );

        // not a valid type, but that's not a guarantee the parser makes.
        assert_eq!(
            dsl.parse("(singleton thing singleton (singleton thing))")
                .unwrap(),
            Expression::Application(
                Box::new(Expression::Application(
                    Box::new(Expression::Application(
                        Box::new(Expression::Primitive(0)),
                        Box::new(Expression::Primitive(1)),
                    )),
                    Box::new(Expression::Primitive(0)),
                )),
                Box::new(Expression::Application(
                    Box::new(Expression::Primitive(0)),
                    Box::new(Expression::Primitive(1)),
                )),
            )
        );

        assert!(dsl.parse("()").is_err());
    }

    #[test]
    fn test_parse_index() {
        let dsl = DSL::new(
            vec![
                (String::from("singleton"), arrow![tp!(0), tp!(list(tp!(0)))]),
            ],
            vec![],
        );
        assert_eq!(
            dsl.parse("(singleton $0)").unwrap(),
            Expression::Application(
                Box::new(Expression::Primitive(0)),
                Box::new(Expression::Index(0))
            )
        );

        /// an index never makes sense outside of an application or lambda body.
        assert!(dsl.parse("$0").is_err());
    }

    #[test]
    fn test_parse_invented() {
        let dsl = DSL::new(
            vec![
                (String::from("+"), arrow![tp!(int), tp!(int), tp!(int)]),
                (String::from("1"), tp!(int)),
            ],
            vec![
                (
                    Expression::Application(
                        Box::new(Expression::Primitive(0)),
                        Box::new(Expression::Primitive(1)),
                    ),
                    arrow![tp!(int), tp!(int)],
                ),
            ],
        );
        assert_eq!(
            dsl.parse("(#(+ 1) 1)").unwrap(),
            Expression::Application(
                Box::new(Expression::Invented(0)),
                Box::new(Expression::Primitive(1)),
            )
        );
        assert!(dsl.parse("(#(+ 1 1) 1)").is_err());
    }

    #[test]
    fn test_parse_abstraction() {
        let dsl = DSL::new(
            vec![
                (String::from("+"), arrow![tp!(int), tp!(int), tp!(int)]),
                (String::from("1"), tp!(int)),
            ],
            vec![
                (
                    Expression::Abstraction(Box::new(Expression::Application(
                        Box::new(Expression::Application(
                            Box::new(Expression::Primitive(0)),
                            Box::new(Expression::Application(
                                Box::new(Expression::Application(
                                    Box::new(Expression::Primitive(0)),
                                    Box::new(Expression::Primitive(1)),
                                )),
                                Box::new(Expression::Primitive(1)),
                            )),
                        )),
                        Box::new(Expression::Index(0)),
                    ))),
                    arrow![tp!(int), tp!(int)],
                ),
            ],
        );
        let expr = dsl.parse("(λ (+ $0))").unwrap();
        assert_eq!(
            expr,
            Expression::Abstraction(Box::new(Expression::Application(
                Box::new(Expression::Primitive(0)),
                Box::new(Expression::Index(0)),
            )))
        );
        assert_eq!(expr.to_string(&dsl), "(λ (+ $0))");
        let expr = dsl.parse("(#(lambda (+ (+ 1 1) $0)) ((lambda (+ $0 1)) 1))")
            .unwrap();
        assert_eq!(
            expr,
            Expression::Application(
                Box::new(Expression::Invented(0)),
                Box::new(Expression::Application(
                    Box::new(Expression::Abstraction(Box::new(Expression::Application(
                        Box::new(Expression::Application(
                            Box::new(Expression::Primitive(0)),
                            Box::new(Expression::Index(0)),
                        )),
                        Box::new(Expression::Primitive(1)),
                    )))),
                    Box::new(Expression::Primitive(1)),
                )),
            ),
        );
        assert_eq!(
            expr.to_string(&dsl),
            "(#(λ (+ (+ 1 1) $0)) ((λ (+ $0 1)) 1))"
        );
        let expr = dsl.parse("(lambda $0)").unwrap();
        assert_eq!(
            expr,
            Expression::Abstraction(Box::new(Expression::Index(0)))
        );
        assert_eq!(expr.to_string(&dsl), "(λ $0)");
    }

    #[test]
    fn test_infer() {
        let dsl = DSL::new(
            vec![
                (String::from("singleton"), arrow![tp!(0), tp!(list(tp!(0)))]),
                (String::from(">="), arrow![tp!(int), tp!(int), tp!(bool)]),
                (String::from("+"), arrow![tp!(int), tp!(int), tp!(int)]),
                (String::from("0"), tp!(int)),
                (String::from("1"), tp!(int)),
            ],
            vec![
                (
                    Expression::Application(
                        Box::new(Expression::Primitive(2)),
                        Box::new(Expression::Primitive(4)),
                    ),
                    arrow![tp!(int), tp!(int)],
                ),
            ],
        );
        let expr = Expression::Application(
            Box::new(Expression::Primitive(0)),
            Box::new(Expression::Application(
                Box::new(Expression::Abstraction(Box::new(Expression::Application(
                    Box::new(Expression::Application(
                        Box::new(Expression::Primitive(1)),
                        Box::new(Expression::Index(0)),
                    )),
                    Box::new(Expression::Primitive(4)),
                )))),
                Box::new(Expression::Application(
                    Box::new(Expression::Invented(0)),
                    Box::new(Expression::Primitive(3)),
                )),
            )),
        );
        assert_eq!(expr.infer(&dsl).unwrap(), tp!(list(tp!(bool))));
        assert_eq!(
            expr.to_string(&dsl),
            "(singleton ((λ (>= $0 1)) (#(+ 1) 0)))"
        );
    }
}
