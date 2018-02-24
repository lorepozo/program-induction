//! A polymorphically-typed lambda calculus representation.

use std::collections::{HashMap, VecDeque};
use std::f64;
use std::fmt::{self, Debug};
use std::rc::Rc;
use polytype::{Context, Type};
use super::{InferenceError, Representation, Task};
use super::ec::{Frontier, EC};

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

mod enumerator {
    use std::collections::VecDeque;
    use std::iter;
    use std::f64;
    use std::rc::Rc;
    use polytype::{Context, Type};
    use super::{Expression, Language, LinkedList};

    const BUDGET_INCREMENT: f64 = 1.0;
    const MAX_DEPTH: u32 = 256;

    pub fn new<'a>(
        dsl: &'a Language,
        request: Type,
    ) -> Box<Iterator<Item = (Expression, f64)> + 'a> {
        let budget = |offset: f64| (offset, offset + BUDGET_INCREMENT);
        let ctx = Context::default();
        let env = Rc::new(LinkedList::default());
        let depth = 0;
        Box::new(
            (0..)
                .map(|n| BUDGET_INCREMENT * f64::from(n))
                .flat_map(move |offset| {
                    enumerate(
                        dsl,
                        request.clone(),
                        &ctx,
                        env.clone(),
                        budget(offset),
                        depth,
                    ).map(move |(log_prior, _, expr)| (expr, log_prior))
                }),
        )
    }
    fn enumerate<'a>(
        dsl: &'a Language,
        request: Type,
        ctx: &Context,
        env: Rc<LinkedList<Type>>,
        budget: (f64, f64),
        depth: u32,
    ) -> Box<Iterator<Item = (f64, Context, Expression)> + 'a> {
        if budget.1 <= 0f64 || depth > MAX_DEPTH {
            Box::new(iter::empty())
        } else if let Type::Arrow(arrow) = request {
            let env = LinkedList::prepend(&env, *arrow.arg);
            let it = enumerate(dsl, *arrow.ret, ctx, env, budget, depth)
                .map(|(ll, ctx, body)| (ll, ctx, Expression::Abstraction(Box::new(body))));
            Box::new(it)
        } else {
            Box::new(
                candidates(dsl, &request, ctx, &env.as_vecdeque())
                    .into_iter()
                    .filter(move |&(ll, _, _, _)| -ll <= budget.1)
                    .flat_map(move |(ll, expr, tp, ctx)| {
                        let arg_tps: VecDeque<Type> = if let Type::Arrow(f_tp) = tp {
                            f_tp.args().into_iter().cloned().collect()
                        } else {
                            VecDeque::new()
                        };
                        let budget = (budget.0 + ll, budget.1 + ll);
                        enumerate_application(
                            dsl,
                            &ctx,
                            env.clone(),
                            expr,
                            arg_tps,
                            budget,
                            depth + 1,
                        ).map(move |(l, c, x)| (l + ll, c, x))
                    }),
            )
        }
    }
    fn enumerate_application<'a>(
        dsl: &'a Language,
        ctx: &Context,
        env: Rc<LinkedList<Type>>,
        f: Expression,
        mut arg_tps: VecDeque<Type>,
        budget: (f64, f64),
        depth: u32,
    ) -> Box<Iterator<Item = (f64, Context, Expression)> + 'a> {
        if let Some(arg_tp) = arg_tps.pop_front() {
            let arg_tp = arg_tp.apply(ctx);
            Box::new(
                enumerate(dsl, arg_tp, ctx, env.clone(), (0f64, budget.1), depth).flat_map(
                    move |(arg_ll, ctx, arg)| {
                        let f_next = Expression::Application(Box::new(f.clone()), Box::new(arg));
                        let budget = (budget.0 + arg_ll, budget.1 + arg_ll);
                        enumerate_application(
                            dsl,
                            &ctx,
                            env.clone(),
                            f_next,
                            arg_tps.clone(),
                            budget,
                            depth,
                        ).map(move |(l, c, x)| (arg_ll + l, c, x))
                    },
                ),
            )
        } else if budget.0 < 0f64 && 0f64 <= budget.1 {
            Box::new(iter::once((0f64, ctx.clone(), f)))
        } else {
            Box::new(iter::empty())
        }
    }

    fn candidates(
        dsl: &Language,
        request: &Type,
        ctx: &Context,
        env: &VecDeque<Type>,
    ) -> Vec<(f64, Expression, Type, Context)> {
        let mut cands = Vec::new();
        let prims = dsl.primitives
            .iter()
            .zip(&dsl.primitives_logprob)
            .enumerate()
            .map(|(i, (&(_, ref tp), &p))| (p, tp, true, Expression::Primitive(i)));
        let invented = dsl.invented
            .iter()
            .zip(&dsl.invented_logprob)
            .enumerate()
            .map(|(i, (&(_, ref tp), &p))| (p, tp, true, Expression::Invented(i)));
        let indices = env.iter()
            .enumerate()
            .map(|(i, tp)| (dsl.variable_logprob, tp, false, Expression::Index(i)));
        for (p, tp, instantiate, expr) in prims.chain(invented).chain(indices) {
            let mut ctx = ctx.clone();
            let itp;
            let tp = if instantiate {
                itp = tp.instantiate_indep(&mut ctx);
                &itp
            } else {
                tp
            };
            let ret = if let Type::Arrow(ref arrow) = *tp {
                arrow.returns()
            } else {
                &tp
            };
            if ctx.unify(ret, request).is_ok() {
                let tp = tp.apply(&ctx);
                cands.push((p, expr, tp, ctx))
            }
        }
        // update probabilities for variables (indices)
        let n_indexed = cands
            .iter()
            .filter(|&&(_, ref expr, _, _)| match *expr {
                Expression::Index(_) => true,
                _ => false,
            })
            .count() as f64;
        for mut c in &mut cands {
            if let Expression::Index(_) = c.1 {
                c.0 -= n_indexed.ln()
            }
        }
        // normalize
        let p_largest = cands
            .iter()
            .map(|&(p, _, _, _)| p)
            .fold(f64::NEG_INFINITY, |acc, p| acc.max(p));
        let z = p_largest
            + cands
                .iter()
                .map(|&(p, _, _, _)| (p - p_largest).exp())
                .sum::<f64>()
                .ln();
        for mut c in &mut cands {
            c.0 -= z;
        }
        cands
    }
}

mod eval {
    //! Only works with systems that don't have first order functions. (i.e. evaluation only
    //! happens by calling primitives.)
    use std::collections::VecDeque;
    use std::fmt::Debug;
    use std::rc::Rc;
    use polytype::Type;
    use super::{Expression, Language};

    #[derive(Clone, Debug, PartialEq)]
    pub enum ReducedExpression<'a, V: Clone + PartialEq + Debug> {
        Value(V),
        Primitive(&'a str, &'a Type),
        Application(Vec<ReducedExpression<'a, V>>),
        Abstraction(Box<ReducedExpression<'a, V>>),
        Index(usize),
    }
    impl<'a, V> ReducedExpression<'a, V>
    where
        V: Clone + PartialEq + Debug,
    {
        pub fn new(dsl: &'a Language, expr: &Expression) -> Self {
            Self::from_expr(dsl, &dsl.strip_invented(expr))
        }
        pub fn check<F>(&self, evaluator: &F, inps: &[V], out: &V) -> bool
        where
            F: Fn(&str, &[V]) -> V,
        {
            let expr = self.clone().with_args(inps);
            let env = &Rc::new(VecDeque::new());
            let mut evaluated = expr.eval(evaluator, env);
            loop {
                let next = evaluated.eval(evaluator, env);
                if next == evaluated {
                    break;
                } else {
                    evaluated = next
                }
            }
            match evaluated {
                ReducedExpression::Value(ref o) => o == out,
                _ => false,
            }
        }
        fn eval<F>(
            &self,
            evaluator: &F,
            env: &Rc<VecDeque<ReducedExpression<'a, V>>>,
        ) -> ReducedExpression<'a, V>
        where
            F: Fn(&str, &[V]) -> V,
        {
            match *self {
                ReducedExpression::Application(ref xs) => {
                    let f = &xs[0];
                    let mut xs: Vec<_> = xs[1..].iter().map(|x| x.eval(evaluator, env)).collect();
                    match *f {
                        ReducedExpression::Primitive(name, tp) => {
                            // when applying a primitive, check if all arity-many args are concrete
                            // values and evaluate if possible.
                            if let Type::Arrow(ref arrow) = *tp {
                                let arity = arrow.args().len();
                                if arity <= xs.len() && xs.iter().take(arity).all(|x| match *x {
                                    ReducedExpression::Value(_) => true,
                                    _ => false,
                                }) {
                                    let mut args = xs;
                                    let mut xs = args.split_off(arity);
                                    let args: Vec<V> = args.into_iter()
                                        .map(|x| match x {
                                            ReducedExpression::Value(v) => v,
                                            _ => unreachable!(),
                                        })
                                        .collect();
                                    let v = ReducedExpression::Value(evaluator(name, &args));
                                    if xs.is_empty() {
                                        v
                                    } else {
                                        xs.insert(0, v);
                                        ReducedExpression::Application(xs)
                                    }
                                } else {
                                    xs.insert(0, f.eval(evaluator, env));
                                    ReducedExpression::Application(xs)
                                }
                            } else {
                                panic!("tried to apply a primitive that wasn't a function")
                            }
                        }
                        ReducedExpression::Abstraction(ref body) => {
                            // when applying an abstraction, try to beta-reduce
                            if xs.is_empty() {
                                ReducedExpression::Abstraction(body.clone())
                            } else {
                                let binding = xs.remove(0);
                                let mut env = (**env).clone();
                                env.push_front(binding);
                                let v = body.eval(evaluator, &Rc::new(env));
                                if xs.is_empty() {
                                    v
                                } else if let ReducedExpression::Application(mut v) = v {
                                    v.extend(xs);
                                    ReducedExpression::Application(v)
                                } else {
                                    xs.insert(0, v);
                                    ReducedExpression::Application(xs)
                                }
                            }
                        }
                        _ => {
                            xs.insert(0, f.eval(evaluator, env));
                            ReducedExpression::Application(xs)
                        }
                    }
                }
                ReducedExpression::Primitive(name, tp) => {
                    if let Type::Arrow(_) = *tp {
                        ReducedExpression::Primitive(name, tp)
                    } else {
                        ReducedExpression::Value(evaluator(name, &[]))
                    }
                }
                ReducedExpression::Index(i) => match env.get(i) {
                    Some(x) => x.clone(),
                    None => ReducedExpression::Index(i),
                },
                _ => self.clone(),
            }
        }
        fn with_args(self, inps: &[V]) -> Self {
            let mut inps: Vec<_> = inps.iter()
                .map(|v| ReducedExpression::Value(v.clone()))
                .collect();
            match self {
                ReducedExpression::Application(mut xs) => {
                    xs.extend(inps);
                    ReducedExpression::Application(xs)
                }
                _ => {
                    inps.insert(0, self);
                    ReducedExpression::Application(inps)
                }
            }
        }
        fn from_expr(dsl: &'a Language, expr: &Expression) -> Self {
            match *expr {
                Expression::Primitive(num) => {
                    ReducedExpression::Primitive(&dsl.primitives[num].0, &dsl.primitives[num].1)
                }
                Expression::Application(ref f, ref x) => {
                    let mut v = vec![Self::from_expr(dsl, x)];
                    let mut f: &Expression = f;
                    loop {
                        if let Expression::Application(ref inner_f, ref x) = *f {
                            v.push(Self::from_expr(dsl, x));
                            f = inner_f;
                        } else {
                            v.push(Self::from_expr(dsl, f));
                            break;
                        }
                    }
                    v.reverse();
                    ReducedExpression::Application(v)
                }
                Expression::Abstraction(ref body) => {
                    ReducedExpression::Abstraction(Box::new(Self::from_expr(dsl, body)))
                }
                Expression::Index(i) => ReducedExpression::Index(i),
                Expression::Invented(_) => unreachable!(/* invented was stripped */),
            }
        }
    }
}

#[derive(Debug, Clone)]
struct LinkedList<T: Clone>(Option<(T, Rc<LinkedList<T>>)>);
impl<T: Clone> LinkedList<T> {
    fn prepend(lst: &Rc<LinkedList<T>>, v: T) -> Rc<LinkedList<T>> {
        Rc::new(LinkedList(Some((v, lst.clone()))))
    }
    fn as_vecdeque(&self) -> VecDeque<T> {
        let mut lst: &Rc<LinkedList<T>>;
        let mut out = VecDeque::new();
        if let Some((ref v, ref nlst)) = self.0 {
            out.push_back(v.clone());
            lst = nlst;
            while let Some((ref v, ref nlst)) = lst.0 {
                out.push_back(v.clone());
                lst = nlst;
            }
        }
        out
    }
}
impl<T: Clone> Default for LinkedList<T> {
    fn default() -> Self {
        LinkedList(None)
    }
}

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
