//! A polymorphically-typed lambda calculus representation.

mod enumerator;
mod eval;

use std::collections::{HashMap, VecDeque};
use std::f64;
use std::fmt::{self, Debug};
use polytype::{Context, Type};
use super::{InferenceError, Representation, Task};
use super::ec::{Frontier, EC};

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
impl ::std::error::Error for ParseError {
    fn description(&self) -> &str {
        "could not parse expression"
    }
}

/// A Language is a registry for primitive and invented expressions in a polymorphically-typed lambda
/// calculus, as well as corresponding production probabilities.
#[derive(Debug, Clone)]
pub struct Language {
    pub primitives: Vec<(String, Type)>,
    pub invented: Vec<(Expression, Type)>,
    pub variable_logprob: f64,
    pub primitives_logprob: Vec<f64>,
    pub invented_logprob: Vec<f64>,
}
impl Language {
    /// A uniform distribution over primitives and invented expressions, as well as the abstraction
    /// operation.
    pub fn uniform(primitives: Vec<(String, Type)>, invented: Vec<Expression>) -> Self {
        let n_primitives = primitives.len();
        let mut dsl = Self {
            primitives,
            invented: vec![],
            variable_logprob: 0f64,
            primitives_logprob: vec![0f64; n_primitives],
            invented_logprob: vec![],
        };
        if !invented.is_empty() {
            let n_invented = invented.len();
            dsl.invented = invented
                .into_iter()
                .map(|expr| {
                    let tp = dsl.infer(&expr).unwrap();
                    (expr, tp)
                })
                .collect();
            dsl.invented_logprob = vec![0f64; n_invented];
        }
        dsl
    }

    /// As with any [`Representation`], we must be able to infer the type of an [`Expression`]:
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # fn main() {
    /// # use programinduction::lambda::{Language, Expression};
    /// let dsl = Language::uniform(
    ///     vec![
    ///         (String::from("singleton"), arrow![tp!(0), tp!(list(tp!(0)))]),
    ///         (String::from(">="), arrow![tp!(int), tp!(int), tp!(bool)]),
    ///         (String::from("+"), arrow![tp!(int), tp!(int), tp!(int)]),
    ///         (String::from("0"), tp!(int)),
    ///         (String::from("1"), tp!(int)),
    ///     ],
    ///     vec![
    ///         Expression::Application(
    ///             Box::new(Expression::Primitive(2)),
    ///             Box::new(Expression::Primitive(4)),
    ///         ),
    ///     ],
    /// );
    /// let expr = dsl.parse("(singleton ((λ (>= $0 1)) (#(+ 1) 0)))").unwrap();
    /// assert_eq!(dsl.infer(&expr).unwrap(), tp!(list(tp!(bool))));
    /// # }
    /// ```
    ///
    /// [`Representation`]: ../trait.Representation.html
    /// [`Expression`]: ../enum.Expression.html
    pub fn infer(&self, expr: &Expression) -> Result<Type, InferenceError> {
        let mut ctx = Context::default();
        let env = VecDeque::new();
        let mut indices = HashMap::new();
        expr.infer(self, &mut ctx, &env, &mut indices)
    }

    /// Enumerate expressions for a request type (including its probability and appropriately
    /// instantiated `Type`):
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # fn main() {
    /// use programinduction::lambda::{Expression, Language};
    ///
    /// let dsl = Language::uniform(
    ///     vec![
    ///         (String::from("0"), tp!(int)),
    ///         (String::from("1"), tp!(int)),
    ///         (String::from("+"), arrow![tp!(int), tp!(int), tp!(int)]),
    ///         (String::from(">"), arrow![tp!(int), tp!(int), tp!(bool)]),
    ///     ],
    ///     vec![],
    /// );
    /// let exprs: Vec<Expression> = dsl.enumerate(tp!(int))
    ///     .take(8)
    ///     .map(|(expr, _log_prior)| expr)
    ///     .collect();
    ///
    /// assert_eq!(
    ///     exprs,
    ///     vec![
    ///         Expression::Primitive(0),
    ///         Expression::Primitive(1),
    ///         dsl.parse("(+ 0 0)").unwrap(),
    ///         dsl.parse("(+ 0 1)").unwrap(),
    ///         dsl.parse("(+ 1 0)").unwrap(),
    ///         dsl.parse("(+ 1 1)").unwrap(),
    ///         dsl.parse("(+ 0 (+ 0 0))").unwrap(),
    ///         dsl.parse("(+ 0 (+ 0 1))").unwrap(),
    ///     ]
    /// );
    /// # }
    /// ```
    pub fn enumerate<'a>(&'a self, tp: Type) -> Box<Iterator<Item = (Expression, f64)> + 'a> {
        enumerator::new(self, tp)
    }

    /// Evaluate an expressions based on an input/output pair.
    ///
    /// Inputs are given as a sequence representing sequentially applied arguments.
    ///
    /// This is not capable of dealing with first order functions. For now, if you need that kind
    /// of behavior, you must implement your own evaluator (e.g. call scheme and run appropriate
    /// code).
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use]
    /// # extern crate polytype;
    /// # extern crate programinduction;
    /// use programinduction::lambda::Language;
    ///
    /// fn evaluator(name: &str, inps: &[i32]) -> i32 {
    ///     match name {
    ///         "0" => 0,
    ///         "1" => 1,
    ///         "+" => inps[0] + inps[1],
    ///         _ => unreachable!(),
    ///     }
    /// }
    ///
    /// # fn main() {
    /// let dsl = Language::uniform(
    ///     vec![
    ///         (String::from("0"), tp!(int)),
    ///         (String::from("1"), tp!(int)),
    ///         (String::from("+"), arrow![tp!(int), tp!(int), tp!(int)]),
    ///     ],
    ///     vec![],
    /// );
    /// let expr = dsl.parse("(λ (+ (+ 1 $0)))").unwrap();
    /// let inps = vec![2, 5];
    /// let out = 8;
    /// assert!(dsl.check(&expr, &evaluator, &inps, &out));
    /// # }
    /// ```
    pub fn check<V, F>(&self, expr: &Expression, evaluator: &F, inps: &[V], out: &V) -> bool
    where
        F: Fn(&str, &[V]) -> V,
        V: Clone + PartialEq + Debug,
    {
        eval::ReducedExpression::new(self, expr).check(evaluator, inps, out)
    }

    /// Get details (name, type, log-likelihood) about a primitive according to its
    /// identifier (which is used in [`Expression::Primitive`]).
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # fn main() {
    /// # use programinduction::lambda::{Language, Expression};
    /// let dsl = Language::uniform(
    ///     vec![
    ///         (String::from("0"), tp!(int)),
    ///         (String::from("1"), tp!(int)),
    ///         (String::from("+"), arrow![tp!(int), tp!(int), tp!(int)]),
    ///     ],
    ///     vec![],
    /// );
    /// assert_eq!(dsl.primitive(0), Some(("0", &tp!(int), 0.)));
    /// # }
    /// ```
    ///
    /// [`Expression::Primitive`]: enum.Expression.html#variant.Primitive
    pub fn primitive(&self, num: usize) -> Option<(&str, &Type, f64)> {
        self.primitives
            .iter()
            .zip(&self.primitives_logprob)
            .nth(num)
            .map(|(&(ref name, ref tp), &p)| (name.as_str(), tp, p))
    }

    /// Get details (expression, type, log-likelihood) about an invented expression according to
    /// its identifier (which is used in [`Expression::Invented`]).
    ///
    /// [`Expression::Invented`]: enum.Expression.html#variant.Invented
    pub fn invented(&self, num: usize) -> Option<(&Expression, &Type, f64)> {
        self.invented
            .iter()
            .zip(&self.invented_logprob)
            .nth(num)
            .map(|(&(ref fragment, ref tp), &p)| (fragment, tp, p))
    }

    /// Register a new invented expression. If it has a valid type, this will be `Ok(num)`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # fn main() {
    /// # use programinduction::lambda::{Language, Expression};
    /// let mut dsl = Language::uniform(
    ///     vec![
    ///         (String::from("0"), tp!(int)),
    ///         (String::from("1"), tp!(int)),
    ///         (String::from("+"), arrow![tp!(int), tp!(int), tp!(int)]),
    ///     ],
    ///     vec![]
    /// );
    /// let expr = dsl.parse("(+ 1)").unwrap();
    /// dsl.invent(expr.clone(), -0.5).unwrap();
    /// assert_eq!(dsl.invented(0), Some((&expr, &arrow![tp!(int), tp!(int)], -0.5)));
    /// # }
    /// ```
    pub fn invent(
        &mut self,
        expr: Expression,
        log_probability: f64,
    ) -> Result<usize, InferenceError> {
        let mut ctx = Context::default();
        let env = VecDeque::new();
        let mut indices = HashMap::new();
        let tp = expr.infer(self, &mut ctx, &env, &mut indices)?;
        self.invented.push((expr, tp));
        self.invented_logprob.push(log_probability);
        Ok(self.invented.len() - 1)
    }

    /// Remove all invented expressions by pulling out their underlying expressions.
    pub fn strip_invented(&self, expr: &Expression) -> Expression {
        expr.strip_invented(&self.invented)
    }
    /// The inverse of [`stringify`].
    ///
    /// Lambda expressions take the form `(lambda BODY)` or `(λ BODY)`, where BODY is an expression
    /// that may use a corresponding De Bruijn [`Index`].
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # fn main() {
    /// # use programinduction::lambda::{Expression, Language};
    /// let dsl = Language::uniform(
    ///     vec![(String::from("+"), arrow![tp!(int), tp!(int), tp!(int)])],
    ///     vec![],
    /// );
    /// let expr = Expression::Abstraction(
    ///     Box::new(Expression::Application(
    ///         Box::new(Expression::Primitive(0)),
    ///         Box::new(Expression::Index(0)),
    ///     ))
    /// );
    /// assert_eq!(dsl.stringify(&expr), "(λ (+ $0))");
    /// // stringify round-trips with dsl.parse
    /// assert_eq!(expr, dsl.parse("(λ (+ $0))").unwrap());
    /// # }
    /// ```
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
                Err(ParseError::new(
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
impl Representation for Language {
    type Expression = Expression;
    fn infer(&self, expr: &Self::Expression) -> Result<Type, InferenceError> {
        self.infer(expr)
    }
}
impl EC for Language {
    fn enumerate<'a>(&'a self, tp: Type) -> Box<Iterator<Item = (Expression, f64)> + 'a> {
        self.enumerate(tp)
    }
    fn mutate<O: Sync>(&self, tasks: &[Task<Self, O>], frontiers: &[Frontier<Self>]) -> Self {
        let _ = (tasks, frontiers);
        self.clone()
        // TODO
    }
}

/// Expressions of lambda calculus, which only make sense with an accompanying [`Language`].
///
/// [`Language`]: struct.Language.html
#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    /// The number associated with a primitive is used by the Language to identify the primitive.
    Primitive(usize),
    Application(Box<Expression>, Box<Expression>),
    Abstraction(Box<Expression>),
    /// De Bruijn index referring to the nth-nearest abstraction (0-indexed).
    /// For example, the identify function is `(λ $0)` or `Abstraction(Index(0))`.
    Index(usize),
    /// The number associated with an invented expression is used by the Language to identify the
    /// invention.
    Invented(usize),
}
impl Expression {
    fn infer(
        &self,
        dsl: &Language,
        mut ctx: &mut Context,
        env: &VecDeque<Type>,
        indices: &mut HashMap<usize, Type>,
    ) -> Result<Type, InferenceError> {
        match *self {
            Expression::Primitive(num) => if let Some(prim) = dsl.primitives.get(num as usize) {
                Ok(prim.1.instantiate_indep(ctx))
            } else {
                Err(InferenceError::BadExpression(format!(
                    "primitive does not exist: {}",
                    num
                )))
            },
            Expression::Application(ref f, ref x) => {
                let f_tp = f.infer(dsl, &mut ctx, env, indices)?;
                let x_tp = x.infer(dsl, &mut ctx, env, indices)?;
                let ret_tp = ctx.new_variable();
                ctx.unify(&f_tp, &arrow![x_tp, ret_tp.clone()])?;
                Ok(ret_tp.apply(ctx))
            }
            Expression::Abstraction(ref body) => {
                let arg_tp = ctx.new_variable();
                let mut env = env.clone();
                env.push_front(arg_tp.clone());
                let ret_tp = body.infer(dsl, &mut ctx, &env, indices)?;
                Ok(arrow![arg_tp, ret_tp].apply(ctx))
            }
            Expression::Index(i) => {
                if (i as usize) < env.len() {
                    Ok(env[i as usize].apply(ctx))
                } else {
                    Ok(indices
                        .entry(i - env.len())
                        .or_insert_with(|| ctx.new_variable())
                        .apply(ctx))
                }
            }
            Expression::Invented(num) => if let Some(inv) = dsl.invented.get(num as usize) {
                Ok(inv.1.instantiate_indep(ctx))
            } else {
                Err(InferenceError::BadExpression(format!(
                    "invention does not exist: {}",
                    num
                )))
            },
        }
    }
    fn strip_invented(&self, invented: &[(Expression, Type)]) -> Expression {
        match *self {
            Expression::Application(ref f, ref x) => Expression::Application(
                Box::new(f.strip_invented(invented)),
                Box::new(x.strip_invented(invented)),
            ),
            Expression::Abstraction(ref body) => {
                Expression::Abstraction(Box::new(body.strip_invented(invented)))
            }
            Expression::Invented(num) => invented[num].0.strip_invented(invented),
            _ => self.clone(),
        }
    }
    fn show(&self, dsl: &Language, is_function: bool) -> String {
        match *self {
            Expression::Primitive(num) => dsl.primitives[num as usize].0.clone(),
            Expression::Application(ref f, ref x) => if is_function {
                format!("{} {}", f.show(dsl, true), x.show(dsl, false))
            } else {
                format!("({} {})", f.show(dsl, true), x.show(dsl, false))
            },
            Expression::Abstraction(ref body) => format!("(λ {})", body.show(dsl, false)),
            Expression::Index(i) => format!("${}", i),
            Expression::Invented(num) => {
                format!("#{}", dsl.invented[num as usize].0.show(dsl, false))
            }
        }
    }
    /// inp must not have leading whitespace. Does not invent.
    fn parse(
        dsl: &Language,
        inp: &str,
        offset: usize, // for good error messages
    ) -> Result<(usize, Expression), ParseError> {
        let init: Option<Result<(usize, Expression), ParseError>> = None;

        let primitive = || {
            match inp.find(|c: char| c.is_whitespace() || c == ')') {
                None if !inp.is_empty() => Some(inp.len()),
                Some(next) if next > 0 => Some(next),
                _ => None,
            }.map(|di| {
                if let Some(num) = dsl.primitives
                    .iter()
                    .position(|&(ref name, _)| name == &inp[..di])
                {
                    Ok((di, Expression::Primitive(num)))
                } else {
                    Err(ParseError::new(offset + di, "unexpected end of expression"))
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
                            None => {
                                break Err(ParseError::new(offset + di, "incomplete application"))
                            }
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
                        .ok_or_else(|| ParseError::new(offset + di, "incomplete application"))
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
                    Err(ParseError::new(
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
            .unwrap_or_else(|| {
                Err(ParseError::new(
                    offset,
                    "could not parse any expression variant",
                ))
            })
    }
}

/// Create a task based on evaluating lambda calculus expressions on test input/output pairs.
///
/// Here we let all tasks be represented by input/output pairs that are values in the space of
/// type `V`. For example, circuits may have `V` be just `bool`, whereas string editing may
/// have `V` be an enum featuring strings, chars, and natural numbers. All inputs, outputs, and
/// evaluated expressions must be representable by `V`.
///
/// An `evaluator` takes the name of a primitive and a vector of sequential inputs to the
/// expression (so an expression with unary type will have one input in a vec of size 1).
///
/// The resulting task is "all-or-nothing": the oracle returns either `0` if all examples are
/// correctly hit or `f64::NEG_INFINITY` otherwise.
///
/// # Examples
///
/// ```
/// # #[macro_use]
/// # extern crate polytype;
/// # extern crate programinduction;
/// use programinduction::lambda::{Language, task_by_example};
///
/// fn evaluator(name: &str, inps: &[i32]) -> i32 {
///     match name {
///         "0" => 0,
///         "1" => 1,
///         "+" => inps[0] + inps[1],
///         _ => unreachable!(),
///     }
/// }
///
/// # fn main() {
/// let examples = vec![(vec![2, 5], 8), (vec![1, 2], 4)];
/// let tp = arrow![tp!(int), tp!(int), tp!(int)];
/// let task = task_by_example(&evaluator, &examples, tp);
///
/// let dsl = Language::uniform(
///     vec![
///         (String::from("0"), tp!(int)),
///         (String::from("1"), tp!(int)),
///         (String::from("+"), arrow![tp!(int), tp!(int), tp!(int)]),
///     ],
///     vec![],
/// );
/// let expr = dsl.parse("(λ (+ (+ 1 $0)))").unwrap();
/// assert!((task.oracle)(&dsl, &expr).is_finite())
/// # }
/// ```
pub fn task_by_example<'a, V, F>(
    evaluator: &'a F,
    examples: &'a [(Vec<V>, V)],
    tp: Type,
) -> Task<'a, Language, &'a [(Vec<V>, V)]>
where
    V: PartialEq + Clone + Sync + Debug + 'a,
    F: Fn(&str, &[V]) -> V + Sync + 'a,
{
    let oracle = Box::new(move |dsl: &Language, expr: &Expression| {
        let expr = &dsl.strip_invented(expr);
        if examples
            .iter()
            .all(|&(ref inps, ref out)| dsl.check(expr, evaluator, inps, out))
        {
            0f64
        } else {
            f64::NEG_INFINITY
        }
    });
    Task {
        oracle,
        observation: examples,
        tp,
    }
}
