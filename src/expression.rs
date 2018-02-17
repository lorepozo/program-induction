use polytype::{self, Type};

use std::collections::{HashMap, VecDeque};
use std::fmt;

/*
 * ERRORS
 */

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

/*
 * /ERRORS
 */

/// An expression context is effectively a registry for primitive and invented expressions.
///
/// Most expressions don't make sense without a context, hence the requirement of one for any
/// expression.
#[derive(Debug, Clone, PartialEq)]
pub struct Context {
    primitives: Vec<(String, Type)>,
    invented: Vec<(Variant, Type)>,
}
impl Context {
    pub fn new(primitives: Vec<(String, Type)>, invented: Vec<(Variant, Type)>) -> Self {
        Context {
            primitives,
            invented,
        }
    }
    pub fn invent(&mut self, expr: Variant) -> usize {
        let mut tctx = polytype::Context::default();
        let env = VecDeque::new();
        let mut indices = HashMap::new();
        let tp = expr.infer_internal(&self, &mut tctx, &env, &mut indices)
            .expect("invalid invention");
        self.invented.push((expr, tp));
        self.invented.len() - 1
    }
    pub fn parse(&self, inp: &str) -> Result<Expression, ParseError> {
        let s = inp.trim_left();
        let offset = inp.len() - s.len();
        Variant::parse(self, s, offset).and_then(move |(di, variant)| {
            if s[di..].chars().all(char::is_whitespace) {
                Ok(Expression {
                    ctx: &self,
                    variant,
                })
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
pub enum Variant {
    Primitive(usize),
    Application(Box<Variant>, Box<Variant>),
    Abstraction(Box<Variant>),
    Index(u64),

    Invented(usize),
}
impl Variant {
    fn infer_internal(
        &self,
        ctx: &Context,
        mut tctx: &mut polytype::Context,
        env: &VecDeque<Type>,
        indices: &mut HashMap<u64, Type>,
    ) -> Result<Type, InferenceError> {
        match self {
            &Variant::Primitive(num) => if let Some(prim) = ctx.primitives.get(num as usize) {
                Ok(prim.1.instantiate_indep(tctx))
            } else {
                Err(InferenceError::BadPrimitive(num))
            },
            &Variant::Application(ref f, ref x) => {
                let f_tp = f.infer_internal(ctx, &mut tctx, env, indices)?;
                let x_tp = x.infer_internal(ctx, &mut tctx, env, indices)?;
                let ret_tp = tctx.new_variable();
                tctx.unify(&f_tp, &arrow![x_tp, ret_tp.clone()])?;
                Ok(ret_tp.apply(tctx))
            }
            &Variant::Abstraction(ref body) => {
                let arg_tp = tctx.new_variable();
                let mut env = env.clone();
                env.push_front(arg_tp.clone());
                let ret_tp = body.infer_internal(ctx, &mut tctx, &env, indices)?;
                Ok(arrow![arg_tp, ret_tp].apply(tctx))
            }
            &Variant::Index(i) => {
                if (i as usize) < env.len() {
                    Ok(env[i as usize].apply(tctx))
                } else {
                    Ok(indices
                        .entry(i - (env.len() as u64))
                        .or_insert_with(|| tctx.new_variable())
                        .apply(tctx))
                }
            }
            &Variant::Invented(num) => if let Some(inv) = ctx.invented.get(num as usize) {
                Ok(inv.1.instantiate_indep(tctx))
            } else {
                Err(InferenceError::BadInvented(num))
            },
        }
    }
    fn show(&self, ctx: &Context, is_function: bool) -> String {
        match self {
            &Variant::Primitive(num) => ctx.primitives[num as usize].0.clone(),
            &Variant::Application(ref f, ref x) => if is_function {
                format!("{} {}", f.show(ctx, true), x.show(ctx, false))
            } else {
                format!("({} {})", f.show(ctx, true), x.show(ctx, false))
            },
            &Variant::Abstraction(ref body) => format!("(λ {})", body.show(ctx, false)),
            &Variant::Index(i) => format!("${}", i),
            &Variant::Invented(num) => {
                format!("#{}", ctx.invented[num as usize].0.show(ctx, false))
            }
        }
    }
    /// inp must not have leading whitespace. Does not invent.
    fn parse(
        ctx: &Context,
        inp: &str,
        offset: usize, // for good error messages
    ) -> Result<(usize, Variant), ParseError> {
        let init: Option<Result<(usize, Variant), ParseError>> = None;

        let primitive = || {
            match inp.find(|c: char| c.is_whitespace() || c == ')') {
                None if inp.len() > 0 => Some(inp.len()),
                Some(next) if next > 0 => Some(next),
                _ => None,
            }.map(|di| {
                if let Some(num) = ctx.primitives
                    .iter()
                    .position(|&(ref name, _)| name == &inp[..di])
                {
                    Ok((di, Variant::Primitive(num)))
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
                        let (ndi, variant) = Variant::parse(ctx, &inp[di..], offset + di)?;
                        items.push_back(variant);
                        di += ndi;
                        // skip spaces
                        di += inp[di..].chars().take_while(|c| c.is_whitespace()).count();
                        // check if complete
                        match inp.chars().nth(di) {
                            None => break Err(ParseError(offset + di, "incomplete application")),
                            Some(')') => {
                                di += 1;
                                break if let Some(init) = items.pop_front() {
                                    let app = items.into_iter().fold(init, |a, v| {
                                        Variant::Application(Box::new(a), Box::new(v))
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
                    let (ndi, body) = Variant::parse(ctx, &inp[di..], offset + di)?;
                    di += ndi;
                    // check if complete
                    inp.chars()
                        .nth(di)
                        .and_then(|c| if c == ')' { Some(di + 1) } else { None })
                        .ok_or(ParseError(offset + di, "incomplete application"))
                        .map(|di| (di, Variant::Abstraction(Box::new(body))))
                })
        };
        let index = || {
            if inp.chars().nth(0) == Some('$') && inp.len() > 1 {
                inp[1..]
                    .find(|c: char| c.is_whitespace() || c == ')')
                    .and_then(|i| inp[1..1 + i].parse::<u64>().ok().map(|num| (1 + i, num)))
                    .map(|(di, num)| Ok((di, Variant::Index(num))))
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
                let (ndi, variant) = Variant::parse(ctx, &inp[di..], offset + di)?;
                di += ndi;
                if let Some(num) = ctx.invented
                    .iter()
                    .position(|&(ref var, _)| var == &variant)
                {
                    Ok((di, Variant::Invented(num)))
                } else {
                    Err(ParseError(
                        offset + di,
                        "invented expr is unfamiliar to context",
                    ))
                }
            })
        };
        // These parsers return None if the variant isn't applicable
        // or Some(Err(..)) if the variant applied but was invalid.
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

#[derive(Debug, Clone, PartialEq)]
pub struct Expression<'a> {
    ctx: &'a Context,
    variant: Variant,
}
impl<'a> Expression<'a> {
    pub fn infer(&self) -> Result<Type, InferenceError> {
        let mut tctx = polytype::Context::default();
        let env = VecDeque::new();
        let mut indices = HashMap::new();
        self.variant
            .infer_internal(self.ctx, &mut tctx, &env, &mut indices)
    }
    fn show(&self, is_function: bool) -> String {
        self.variant.show(self.ctx, is_function)
    }
}
impl<'a> fmt::Display for Expression<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "{}", self.show(false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_primitive() {
        let ctx = Context::new(
            vec![
                (String::from("singleton"), arrow![tp!(0), tp!(list(tp!(0)))]),
            ],
            vec![],
        );
        let expr = ctx.parse("singleton").unwrap();
        assert_eq!(expr.variant, Variant::Primitive(0));

        assert!(ctx.parse("something_else").is_err());
        assert!(ctx.parse("singleton singleton").is_err());
    }

    #[test]
    fn test_parse_application() {
        let ctx = Context::new(
            vec![
                (String::from("singleton"), arrow![tp!(0), tp!(list(tp!(0)))]),
                (String::from("thing"), arrow![tp!(int), tp!(int)]),
            ],
            vec![],
        );
        let expr = ctx.parse("(singleton singleton)").unwrap();
        assert_eq!(
            expr.variant,
            Variant::Application(
                Box::new(Variant::Primitive(0)),
                Box::new(Variant::Primitive(0)),
            )
        );

        // not a valid type, but that's not a guarantee the parser makes.
        let expr = ctx.parse("(singleton thing singleton (singleton thing))")
            .unwrap();
        assert_eq!(
            expr.variant,
            Variant::Application(
                Box::new(Variant::Application(
                    Box::new(Variant::Application(
                        Box::new(Variant::Primitive(0)),
                        Box::new(Variant::Primitive(1)),
                    )),
                    Box::new(Variant::Primitive(0)),
                )),
                Box::new(Variant::Application(
                    Box::new(Variant::Primitive(0)),
                    Box::new(Variant::Primitive(1)),
                )),
            )
        );

        assert!(ctx.parse("()").is_err());
    }

    #[test]
    fn test_parse_index() {
        let ctx = Context::new(
            vec![
                (String::from("singleton"), arrow![tp!(0), tp!(list(tp!(0)))]),
            ],
            vec![],
        );
        let expr = ctx.parse("(singleton $0)").unwrap();
        assert_eq!(
            expr.variant,
            Variant::Application(Box::new(Variant::Primitive(0)), Box::new(Variant::Index(0)))
        );

        /// an index never makes sense outside of an application or lambda body.
        assert!(ctx.parse("$0").is_err());
    }

    #[test]
    fn test_parse_invented() {
        let ctx = Context::new(
            vec![
                (String::from("+"), arrow![tp!(int), tp!(int), tp!(int)]),
                (String::from("1"), tp!(int)),
            ],
            vec![
                (
                    Variant::Application(
                        Box::new(Variant::Primitive(0)),
                        Box::new(Variant::Primitive(1)),
                    ),
                    arrow![tp!(int), tp!(int)],
                ),
            ],
        );
        let expr = ctx.parse("(#(+ 1) 1)").unwrap();
        assert_eq!(
            expr.variant,
            Variant::Application(
                Box::new(Variant::Invented(0)),
                Box::new(Variant::Primitive(1)),
            )
        );
        assert!(ctx.parse("(#(+ 1 1) 1)").is_err());
    }

    #[test]
    fn test_parse_abstraction() {
        let ctx = Context::new(
            vec![
                (String::from("+"), arrow![tp!(int), tp!(int), tp!(int)]),
                (String::from("1"), tp!(int)),
            ],
            vec![
                (
                    Variant::Abstraction(Box::new(Variant::Application(
                        Box::new(Variant::Application(
                            Box::new(Variant::Primitive(0)),
                            Box::new(Variant::Application(
                                Box::new(Variant::Application(
                                    Box::new(Variant::Primitive(0)),
                                    Box::new(Variant::Primitive(1)),
                                )),
                                Box::new(Variant::Primitive(1)),
                            )),
                        )),
                        Box::new(Variant::Index(0)),
                    ))),
                    arrow![tp!(int), tp!(int)],
                ),
            ],
        );
        let expr = ctx.parse("(lambda (+ $0))").unwrap();
        assert_eq!(
            expr.variant,
            Variant::Abstraction(Box::new(Variant::Application(
                Box::new(Variant::Primitive(0)),
                Box::new(Variant::Index(0)),
            )))
        );
        let expr = ctx.parse("(#(lambda (+ (+ 1 1) $0)) ((lambda (+ $0 1)) 1))")
            .unwrap();
        assert_eq!(
            expr.variant,
            Variant::Application(
                Box::new(Variant::Invented(0)),
                Box::new(Variant::Application(
                    Box::new(Variant::Abstraction(Box::new(Variant::Application(
                        Box::new(Variant::Application(
                            Box::new(Variant::Primitive(0)),
                            Box::new(Variant::Index(0)),
                        )),
                        Box::new(Variant::Primitive(1)),
                    )))),
                    Box::new(Variant::Primitive(1)),
                )),
            ),
        );
        let expr = ctx.parse("(lambda $0)").unwrap();
        assert_eq!(
            expr.variant,
            Variant::Abstraction(Box::new(Variant::Index(0)))
        );
    }

    #[test]
    fn test_infer() {
        let ctx = Context::new(
            vec![
                (String::from("singleton"), arrow![tp!(0), tp!(list(tp!(0)))]),
                (String::from(">="), arrow![tp!(int), tp!(int), tp!(bool)]),
                (String::from("+"), arrow![tp!(int), tp!(int), tp!(int)]),
                (String::from("0"), tp!(int)),
                (String::from("1"), tp!(int)),
            ],
            vec![
                (
                    Variant::Application(
                        Box::new(Variant::Primitive(2)),
                        Box::new(Variant::Primitive(4)),
                    ),
                    arrow![tp!(int), tp!(int)],
                ),
            ],
        );
        let v = Variant::Application(
            Box::new(Variant::Primitive(0)),
            Box::new(Variant::Application(
                Box::new(Variant::Abstraction(Box::new(Variant::Application(
                    Box::new(Variant::Application(
                        Box::new(Variant::Primitive(1)),
                        Box::new(Variant::Index(0)),
                    )),
                    Box::new(Variant::Primitive(4)),
                )))),
                Box::new(Variant::Application(
                    Box::new(Variant::Invented(0)),
                    Box::new(Variant::Primitive(3)),
                )),
            )),
        );
        let expr = Expression {
            ctx: &ctx,
            variant: v,
        };
        assert_eq!(expr.infer().unwrap(), tp!(list(tp!(bool))));
        assert_eq!(
            format!("{}", expr),
            "(singleton ((λ (>= $0 1)) (#(+ 1) 0)))"
        );
    }
}
