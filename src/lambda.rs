//! A polymorphically-typed lambda calculus representation.

use std::collections::{HashMap, VecDeque};
use std::f64;
use std::fmt;
use polytype::{Context, Type};
use super::{InferenceError, Representation, Task, EC};

/// A DSL is a registry for primitive and invented expressions in a polymorphically-typed lambda
/// calculus, as well as corresponding production probabilities.
///
/// # Examples
///
/// Stringify and parse expressions in the DSL:
///
/// ```
/// # #[macro_use] extern crate polytype;
/// # extern crate programinduction;
/// # fn main() {
/// # use programinduction::lambda::{Expression, DSL};
/// let dsl = DSL::uniform(
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
/// # use programinduction::lambda::{DSL, Expression};
/// let dsl = DSL::uniform(
///     vec![
///         (String::from("singleton"), arrow![tp!(0), tp!(list(tp!(0)))]),
///         (String::from(">="), arrow![tp!(int), tp!(int), tp!(bool)]),
///         (String::from("+"), arrow![tp!(int), tp!(int), tp!(int)]),
///         (String::from("0"), tp!(int)),
///         (String::from("1"), tp!(int)),
///     ],
///     vec![
///         (
///             Expression::Application(
///                 Box::new(Expression::Primitive(2)),
///                 Box::new(Expression::Primitive(4)),
///             ),
///             arrow![tp!(int), tp!(int)],
///         ),
///     ],
/// );
/// let expr = dsl.parse("(singleton ((λ (>= $0 1)) (#(+ 1) 0)))").unwrap();
/// assert_eq!(dsl.infer(&expr).unwrap(), tp!(list(tp!(bool))));
/// # }
/// ```
///
/// Get candidates primitives or invented expressions for a request type (including its probability
/// and appropriately instantiated `Type`):
///
/// ```
/// # #[macro_use] extern crate polytype;
/// # extern crate programinduction;
/// # fn main() {
/// # use std::collections::VecDeque;
/// # use polytype::Context;
/// use programinduction::lambda::{Expression, DSL};
///
/// let dsl = DSL::uniform(
///     vec![
///         (String::from("0"), tp!(int)),
///         (String::from("1"), tp!(int)),
///         (String::from("+"), arrow![tp!(int), tp!(int), tp!(int)]),
///         (String::from(">"), arrow![tp!(int), tp!(int), tp!(bool)]),
///     ],
///     vec![],
/// );
/// let request = tp!(int);
/// let ctx = Context::default();
/// let env = VecDeque::new();
///
/// let candidates = dsl.candidates(&request, &ctx, &env, false);
/// let candidate_exprs: Vec<Expression> = candidates
///     .into_iter()
///     .map(|(p, expr, tp, ctx)| expr)
///     .collect();
/// assert_eq!(candidate_exprs, vec![
///     Expression::Primitive(0),
///     Expression::Primitive(1),
///     Expression::Primitive(2),
/// ]);
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct DSL {
    pub primitives: Vec<(String, Type)>,
    pub invented: Vec<(Expression, Type)>,
    pub variable_logprob: f64,
    pub primitives_logprob: Vec<f64>,
    pub invented_logprob: Vec<f64>,
}
impl DSL {
    pub fn uniform(primitives: Vec<(String, Type)>, invented: Vec<(Expression, Type)>) -> Self {
        let n_primitives = primitives.len();
        let n_invented = invented.len();
        Self {
            primitives,
            invented,
            variable_logprob: 0f64,
            primitives_logprob: vec![0f64; n_primitives],
            invented_logprob: vec![0f64; n_invented],
        }
    }
    pub fn primitive(&self, num: usize) -> Option<(&str, &Type, f64)> {
        self.primitives
            .iter()
            .zip(&self.primitives_logprob)
            .nth(num)
            .map(|(&(ref name, ref tp), &p)| (name.as_str(), tp, p))
    }
    pub fn invented(&self, num: usize) -> Option<(&Expression, &Type, f64)> {
        self.invented
            .iter()
            .zip(&self.invented_logprob)
            .nth(num)
            .map(|(&(ref fragment, ref tp), &p)| (fragment, tp, p))
    }
    /// Register a new invented expression. If it has a valid type, this will be `Ok(num)`.
    pub fn invent(&mut self, expr: Expression) -> Result<usize, InferenceError> {
        let mut ctx = Context::default();
        let env = VecDeque::new();
        let mut indices = HashMap::new();
        let tp = expr.infer(&self, &mut ctx, &env, &mut indices)?;
        self.invented.push((expr, tp));
        Ok(self.invented.len() - 1)
    }
    pub fn infer(&self, expr: &Expression) -> Result<Type, InferenceError> {
        let mut ctx = Context::default();
        let env = VecDeque::new();
        let mut indices = HashMap::new();
        expr.infer(self, &mut ctx, &env, &mut indices)
    }
    pub fn candidates(
        &self,
        request: &Type,
        ctx: &Context,
        env: &VecDeque<Type>,
        leaf_only: bool,
    ) -> Vec<(f64, Expression, Type, Context)> {
        let mut cands = Vec::new();
        let prims = self.primitives
            .iter()
            .zip(&self.primitives_logprob)
            .enumerate()
            .map(|(i, (&(_, ref tp), &p))| (p, tp, true, Expression::Primitive(i)));
        let invented = self.invented
            .iter()
            .zip(&self.invented_logprob)
            .enumerate()
            .map(|(i, (&(_, ref tp), &p))| (p, tp, true, Expression::Invented(i)));
        let indices = env.iter()
            .enumerate()
            .map(|(i, tp)| (self.variable_logprob, tp, false, Expression::Index(i)));
        for (p, tp, instantiate, expr) in prims.chain(invented).chain(indices) {
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
                let tp = tp.apply(&ctx);
                cands.push((p, expr, tp, ctx))
            }
        }
        // update probabilities for variables (indices)
        let n_indexed = cands
            .iter()
            .filter(|&&(_, ref expr, _, _)| match expr {
                &Expression::Index(_) => true,
                _ => false,
            })
            .count() as f64;
        for mut c in &mut cands {
            match c.1 {
                Expression::Index(_) => c.0 -= n_indexed.ln(),
                _ => (),
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
    pub fn check<V, F>(&self, expr: &Expression, evaluator: &F, inps: &Vec<V>, out: &V) -> bool
    where
        F: Fn(&str, &Vec<V>) -> V,
    {
        // TODO: call lisp or something
        false
    }
    pub fn strip_invented(&self, expr: &Expression) -> Expression {
        expr.strip_invented(&self.invented)
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
    fn infer(
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
                let f_tp = f.infer(dsl, &mut ctx, env, indices)?;
                let x_tp = x.infer(dsl, &mut ctx, env, indices)?;
                let ret_tp = ctx.new_variable();
                ctx.unify(&f_tp, &arrow![x_tp, ret_tp.clone()])?;
                Ok(ret_tp.apply(ctx))
            }
            &Expression::Abstraction(ref body) => {
                let arg_tp = ctx.new_variable();
                let mut env = env.clone();
                env.push_front(arg_tp.clone());
                let ret_tp = body.infer(dsl, &mut ctx, &env, indices)?;
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
    fn strip_invented(&self, invented: &Vec<(Expression, Type)>) -> Expression {
        match self {
            &Expression::Application(ref f, ref x) => Expression::Application(
                Box::new(f.strip_invented(invented)),
                Box::new(x.strip_invented(invented)),
            ),
            &Expression::Abstraction(ref body) => {
                Expression::Abstraction(Box::new(body.strip_invented(invented)))
            }
            &Expression::Invented(num) => invented[num].0.strip_invented(invented),
            _ => self.clone(),
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
impl Representation for DSL {
    type Expression = Expression;
    fn infer(&self, expr: &Self::Expression) -> Result<Type, InferenceError> {
        self.infer(expr)
    }
}
impl EC for DSL {
    fn enumerate<'a>(&'a self, tp: Type) -> Box<Iterator<Item = Expression> + 'a> {
        enumerator::new(self, tp)
    }
    fn mutate<O>(
        &self,
        tasks: &Vec<Task<Self, O>>,
        frontiers: &Vec<Vec<Self::Expression>>,
    ) -> Self {
        self.clone()
    }
}

impl<'a, V: 'a> Task<'a, DSL, &'a Vec<(Vec<V>, V)>>
where
    V: PartialEq,
{
    /// Here we let all tasks be represented by input/output pairs that are values in the space of
    /// type `V`. For example, circuits may have `V` be just `bool`, whereas string editing may
    /// have `V` be an enum featuring strings, chars, and natural numbers. All inputs or evaluated
    /// expressions must be representable by `V`.
    ///
    /// An `evaluator` takes the name of a primitive and a vector of sequential inputs to the
    /// expression (so an expression with unary type will have one input in a vec of size 1).
    ///
    /// The resulting task is "all-or-nothing": the oracle returns either `0` if all examples are
    /// correctly hit or `f64::NEG_INFINITY` otherwise.
    pub fn from_evaluated_examples<F>(
        evaluator: &'a F,
        examples: &'a Vec<(Vec<V>, V)>,
        tp: Type,
    ) -> Self
    where
        F: Fn(&str, &Vec<V>) -> V + 'a,
    {
        let oracle = Box::new(move |dsl: &DSL, expr: &Expression| {
            let ref expr = dsl.strip_invented(expr);
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
}

mod enumerator {
    use std::collections::VecDeque;
    use std::iter;
    use std::rc::Rc;
    use polytype::{Context, Type};
    use super::{Expression, DSL};

    const BUDGET_INCREMENT: f64 = 1.0;
    const MAX_DEPTH: u32 = 256;

    pub fn new<'a>(dsl: &'a DSL, request: Type) -> Box<Iterator<Item = Expression> + 'a> {
        let budget = |offset: f64| (offset, offset + BUDGET_INCREMENT);
        let ctx = Context::default();
        let env = Rc::new(LinkedList::default());
        let depth = 0;
        Box::new(
            (0..)
                .map(|n| BUDGET_INCREMENT * (n as f64))
                .flat_map(move |offset| {
                    enumerate(
                        dsl,
                        request.clone(),
                        &ctx,
                        env.clone(),
                        budget(offset),
                        depth,
                    ).map(|(_, _, expr)| expr)
                }),
        )
    }
    fn enumerate<'a>(
        dsl: &'a DSL,
        request: Type,
        ctx: &Context,
        env: Rc<LinkedList<Type>>,
        budget: (f64, f64),
        depth: u32,
    ) -> Box<Iterator<Item = (f64, Context, Expression)> + 'a> {
        if let Type::Arrow(arrow) = request {
            let env = LinkedList::prepend(env, *arrow.arg);
            let it = enumerate(dsl, *arrow.ret, ctx, env, budget, depth)
                .map(|(ll, ctx, body)| (ll, ctx, Expression::Abstraction(Box::new(body))));
            Box::new(it)
        } else {
            let it = dsl
                .candidates(&request, ctx, &LinkedList::as_vecdeque(&env), false)
                .into_iter()
                .filter(move |&(ll, _, _, _)| -ll > budget.1)
                .flat_map(
                    move |(ll, expr, tp, ctx)| -> Box<Iterator<Item = (f64, Context, Expression)>+'a> {
                        let budget = (budget.0 + ll, budget.1 + ll);
                        if budget.1 <= 0f64 || depth > MAX_DEPTH {
                            Box::new(iter::empty())
                        } else if let Type::Arrow(f_tp) = tp {
                            let arg_tps: VecDeque<Type> =
                                f_tp.args().into_iter().cloned().collect();
                            enumerate_application(
                                dsl,
                                &ctx,
                                env.clone(),
                                expr,
                                ll,
                                arg_tps,
                                budget,
                                depth + 1,
                            )
                        } else if budget.0 < 0f64 && 0f64 <= budget.1 {
                            Box::new(iter::once((ll, ctx, expr)))
                        } else {
                            Box::new(iter::empty())
                        }
                    },
                );
            Box::new(it)
        }
    }
    fn enumerate_application<'a>(
        dsl: &'a DSL,
        ctx: &Context,
        env: Rc<LinkedList<Type>>,
        f: Expression,
        f_ll: f64,
        mut arg_tps: VecDeque<Type>,
        budget: (f64, f64),
        depth: u32,
    ) -> Box<Iterator<Item = (f64, Context, Expression)> + 'a> {
        if arg_tps.is_empty() {
            if budget.0 < 0f64 && 0f64 <= budget.1 {
                Box::new(iter::once((f_ll, ctx.clone(), f)))
            } else {
                Box::new(iter::empty())
            }
        } else {
            let arg_tp = arg_tps.pop_front().unwrap();
            let budget = (0f64, budget.1);
            let it = enumerate(dsl, arg_tp, ctx, env.clone(), budget, depth).flat_map(
                move |(arg_ll, ctx, arg)| {
                    let f_next = Expression::Application(Box::new(f.clone()), Box::new(arg));
                    let budget = (budget.0 + arg_ll, budget.1 + arg_ll);
                    enumerate_application(
                        dsl,
                        &ctx,
                        env.clone(),
                        f_next,
                        f_ll + arg_ll,
                        arg_tps.clone(),
                        budget,
                        depth,
                    )
                },
            );
            Box::new(it)
        }
    }

    #[derive(Debug, Clone)]
    struct LinkedList<T: Clone>(Option<(T, Rc<LinkedList<T>>)>);
    impl<T: Clone> LinkedList<T> {
        fn prepend(lst: Rc<LinkedList<T>>, v: T) -> Rc<LinkedList<T>> {
            Rc::new(LinkedList(Some((v, lst.clone()))))
        }
        fn as_vecdeque(mut lst: &Rc<LinkedList<T>>) -> VecDeque<T> {
            let mut out = VecDeque::new();
            loop {
                if let Some((ref v, ref nlst)) = lst.0 {
                    out.push_back(v.clone());
                    lst = nlst;
                } else {
                    break;
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
