//! A library for program induction and learning grammars.
//!
//! Good places to look are [`DSL`] and [`ec`].
//!
//! [`DSL`]: struct.DSL.html
//! [`ec`]: ec/index.html

#[macro_use]
extern crate polytype;

pub mod ec;

use std::collections::{HashMap, VecDeque};
use std::fmt;
use polytype::{Context, Type};

/// The representation of a task which is solved by an [`Expression`] under some [`DSL`].
///
/// [`DSL`]: struct.DSL.html
/// [`Expression`]: enum.Expression.html
pub struct Task<'a, O> {
    /// evaluate an expression by getting its log-likelihood.
    pub oracle: Box<'a + Fn(&Expression, &DSL) -> f64>,
    pub observation: O,
    pub tp: Type,
}

/// A DSL is a registry for primitive and invented expressions in a polymorphically-typed lambda
/// calculus.
///
/// # Examples
///
/// Stringify and parse expressions in the DSL:
///
/// ```
/// # #[macro_use] extern crate polytype;
/// # extern crate programinduction;
/// # fn main() {
/// # use programinduction::{Expression, DSL};
/// let dsl = DSL{
///     primitives: vec![(String::from("+"), arrow![tp!(int), tp!(int), tp!(int)])],
///     invented: vec![],
/// };
/// let expr = Expression::Abstraction(
///     Box::new(Expression::Application(
///         Box::new(Expression::Primitive(0)),
///         Box::new(Expression::Index(0)),
///     ))
/// );
/// assert_eq!(dsl.stringify(&expr), "(λ (+ $0))");
/// // stringify round-trips with dsl.parse
/// assert_eq!(expr, dsl.parse(&dsl.stringify(&expr)).unwrap());
/// # }
/// ```
///
/// Infer types of expressions in the DSL:
///
/// ```
/// # #[macro_use] extern crate polytype;
/// # extern crate programinduction;
/// # fn main() {
/// # use programinduction::{DSL, Expression};
/// let dsl = DSL{
///     primitives: vec![
///         (String::from("singleton"), arrow![tp!(0), tp!(list(tp!(0)))]),
///         (String::from(">="), arrow![tp!(int), tp!(int), tp!(bool)]),
///         (String::from("+"), arrow![tp!(int), tp!(int), tp!(int)]),
///         (String::from("0"), tp!(int)),
///         (String::from("1"), tp!(int)),
///     ],
///     invented: vec![
///         (
///             Expression::Application(
///                 Box::new(Expression::Primitive(2)),
///                 Box::new(Expression::Primitive(4)),
///             ),
///             arrow![tp!(int), tp!(int)],
///         ),
///     ],
/// };
/// let expr = dsl.parse("(singleton ((λ (>= $0 1)) (#(+ 1) 0)))").unwrap();
/// assert_eq!(dsl.infer(&expr).unwrap(), tp!(list(tp!(bool))));
/// # }
/// ```
///
/// Get expressions which unify with a requested type.
///
/// ```
/// # #[macro_use] extern crate polytype;
/// # extern crate programinduction;
/// # fn main() {
/// # use programinduction::{Expression, DSL};
/// # use std::collections::VecDeque;
/// # use polytype::Context;
/// let dsl = DSL{
///     primitives: vec![
///         (String::from("0"), tp!(int)),
///         (String::from("1"), tp!(int)),
///         (String::from("+"), arrow![tp!(int), tp!(int), tp!(int)]),
///         (String::from(">"), arrow![tp!(int), tp!(int), tp!(bool)]),
///     ],
///     invented: vec![],
/// };
/// let request = tp!(int);
/// let ctx = Context::default();
/// let env = VecDeque::new();
///
/// let candidates = dsl.candidates(&request, &ctx, &env, false);
/// let candidate_exprs: Vec<Expression> = candidates.into_iter().map(|(expr, _)| expr).collect();
/// assert_eq!(candidate_exprs, vec![
///     Expression::Primitive(0),
///     Expression::Primitive(1),
///     Expression::Primitive(2),
/// ]);
/// # }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct DSL {
    pub primitives: Vec<(String, Type)>,
    pub invented: Vec<(Expression, Type)>,
}
impl DSL {
    pub fn primitive(&self, num: usize) -> Option<(&str, &Type)> {
        self.primitives
            .get(num)
            .map(|&(ref name, ref tp)| (name.as_str(), tp))
    }
    pub fn invented(&self, num: usize) -> Option<(&Expression, &Type)> {
        self.invented
            .get(num)
            .map(|&(ref fragment, ref tp)| (fragment, tp))
    }
    /// Register a new invented expression. If it has a valid type, this will be `Ok(num)`.
    pub fn invent(&mut self, expr: Expression) -> Result<usize, InferenceError> {
        let mut ctx = Context::default();
        let env = VecDeque::new();
        let mut indices = HashMap::new();
        let tp = expr.infer_internal(&self, &mut ctx, &env, &mut indices)?;
        self.invented.push((expr, tp));
        Ok(self.invented.len() - 1)
    }
    pub fn infer(&self, expr: &Expression) -> Result<Type, InferenceError> {
        let mut ctx = Context::default();
        let env = VecDeque::new();
        let mut indices = HashMap::new();
        expr.infer_internal(self, &mut ctx, &env, &mut indices)
    }
    /// Get the primitives, invented expressions, and indices which have type that unifies with the
    /// request type. For arrows, this only checks if the return type unifies with the request.
    ///
    /// If you have polymorphic types in the environment, be sure that all such types have been
    /// instantiated in the type context.
    pub fn candidates(
        &self,
        request: &Type,
        ctx: &Context,
        env: &VecDeque<Type>,
        leaf_only: bool,
    ) -> Vec<(Expression, Context)> {
        let mut cands = Vec::new();
        let prims = self.primitives
            .iter()
            .enumerate()
            .map(|(i, &(_, ref tp))| (tp, true, Expression::Primitive(i)));
        let invented = self.invented
            .iter()
            .enumerate()
            .map(|(i, &(_, ref tp))| (tp, true, Expression::Invented(i)));
        let indices = env.iter()
            .enumerate()
            .map(|(i, tp)| (tp, false, Expression::Index(i)));
        for (tp, instantiate, expr) in prims.chain(invented).chain(indices) {
            let mut ctx = ctx.clone();
            let itp;
            let tp = if instantiate {
                itp = tp.instantiate_indep(&mut ctx);
                &itp
            } else {
                tp
            };
            let ret = if let &Type::Arrow(ref arrow) = tp {
                if leaf_only {
                    continue;
                }
                arrow.returns()
            } else {
                &tp
            };
            if let Ok(_) = ctx.unify(ret, request) {
                cands.push((expr, ctx))
            }
        }
        cands
    }
    /// The inverse of [`stringify`].
    ///
    /// Lambda expressions take the form `(lambda BODY)` or `(λ BODY)`, where BODY is an expression
    /// that may use a corresponding De Bruijn [`Index`].
    ///
    /// [`stringify`]: #method.stringify
    /// [`Index`]: enum.Expression.html#variant.Index
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
    /// The inverse of [`parse`].
    ///
    /// [`parse`]: #method.parse
    pub fn stringify(&self, expr: &Expression) -> String {
        expr.show(self, false)
    }
}

/// Expressions of lambda calculus, which only make sense with an accompanying DSL.
#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    /// The number associated with a primitive is used by the DSL to identify the primitive.
    Primitive(usize),
    Application(Box<Expression>, Box<Expression>),
    Abstraction(Box<Expression>),
    /// De Bruijn index referring to the nth-nearest abstraction (0-indexed).
    /// For example, the identify function is `(λ $0)` or `Abstraction(Index(0))`.
    Index(usize),
    /// The number associated with an invented expression is used by the DSL to identify the
    /// invention.
    Invented(usize),
}
impl Expression {
    fn infer_internal(
        &self,
        dsl: &DSL,
        mut ctx: &mut Context,
        env: &VecDeque<Type>,
        indices: &mut HashMap<usize, Type>,
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
                        .entry(i - env.len())
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
        // Ordering is intentional.
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
