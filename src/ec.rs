use std::collections::{HashMap, VecDeque};
use std::f64;
use std::iter;
use std::rc::Rc;
use polytype::{Context, Type};

use super::{Expression, Task, DSL};

const BUDGET_INCREMENT: f64 = 1.0;
const MAX_DEPTH: u32 = 256;

/// The entry point for one iteration of the EC algorithm.
///
/// Lots of type variables here.
///
/// - `S` is some "state" initially passed in as a prior, probably taking the form of production
///   probabilities. But it's ultimately up to you what it is and how it gets used. The important
///   thing is that it's something you can change during ec at any point in the pipeline (which
///   will probably be just compression updating production probabilities and adding to the DSL).
/// - `O` is the observation type, something which the recognizer/explorer can take advantage of
///   instead of just basing search off of the type and oracle response.
/// - `RS` is something the recognizer returns that the explorer can use. This could be, for
///   example, a map from task number to a set of production probabilities.
/// - `R`, `E`, and `C` are the types, described in the `where` clause of the function signature
///   here, for a recognizer, an explorer, and a compressor respectively. Note that the compressor
///   returns a DSL as well as the best solution for each task.
pub fn ec<S, O, RS, R, E, C>(
    prior: &DSL,
    mut state: &mut S,
    tasks: &Vec<Task<O>>,
    recognizer: Option<R>,
    explore: E,
    compress: C,
) -> (DSL, Vec<Option<Expression>>)
where
    R: FnOnce(&DSL, &mut S, &Vec<Task<O>>) -> RS,
    E: FnOnce(&DSL, &mut S, &Vec<Task<O>>, Option<RS>) -> Vec<Vec<Expression>>,
    C: FnOnce(&DSL, &mut S, &Vec<Task<O>>, Vec<Vec<Expression>>) -> (DSL, Vec<Option<Expression>>),
{
    let recognized = recognizer.map(|r| r(prior, &mut state, tasks));
    let frontiers = explore(prior, &mut state, tasks, recognized);
    compress(prior, &mut state, tasks, frontiers)
}

/// Production log-probabilities for each primitive and invented expression.
pub struct Productions {
    pub variable: f64,
    pub primitives: Vec<f64>,
    pub invented: Vec<f64>,
}
impl Productions {
    pub fn uniform(n_primitives: usize, n_invented: usize) -> Self {
        Self {
            variable: 0f64,
            primitives: vec![0f64; n_primitives],
            invented: vec![0f64; n_invented],
        }
    }
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # fn main() {
    /// # use std::collections::VecDeque;
    /// # use polytype::Context;
    /// use programinduction::{Expression, DSL};
    /// use programinduction::ec::Productions;
    ///
    /// let dsl = DSL{
    ///     primitives: vec![
    ///         (String::from("0"), tp!(int)),
    ///         (String::from("1"), tp!(int)),
    ///         (String::from("+"), arrow![tp!(int), tp!(int), tp!(int)]),
    ///         (String::from(">"), arrow![tp!(int), tp!(int), tp!(bool)]),
    ///     ],
    ///     invented: vec![],
    /// };
    /// let productions = Productions::uniform(4, 0);
    /// let request = tp!(int);
    /// let ctx = Context::default();
    /// let env = VecDeque::new();
    ///
    /// let candidates = productions.candidates(&dsl, &request, &ctx, &env, false);
    /// let candidate_exprs: Vec<Expression> = candidates
    ///     .into_iter()
    ///     .map(|(p, expr, _, _)| expr)
    ///     .collect();
    /// assert_eq!(candidate_exprs, vec![
    ///     Expression::Primitive(0),
    ///     Expression::Primitive(1),
    ///     Expression::Primitive(2),
    /// ]);
    /// # }
    /// ```
    pub fn candidates(
        &self,
        dsl: &DSL,
        request: &Type,
        ctx: &Context,
        env: &VecDeque<Type>,
        leaf_only: bool,
    ) -> Vec<(f64, Expression, Type, Context)> {
        let mut cands = Vec::new();
        let prims = self.primitives
            .iter()
            .zip(&dsl.primitives)
            .enumerate()
            .map(|(i, (&p, &(_, ref tp)))| (p, tp, true, Expression::Primitive(i)));
        let invented = self.invented
            .iter()
            .zip(&dsl.invented)
            .enumerate()
            .map(|(i, (&p, &(_, ref tp)))| (p, tp, true, Expression::Invented(i)));
        let indices = env.iter()
            .enumerate()
            .map(|(i, tp)| (self.variable, tp, false, Expression::Index(i)));
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
}

pub struct State {
    pub productions: Productions,
    /// frontier size is the number of task solutions to be hit before enumeration is stopped.
    pub frontier_size: usize,
    /// search limit is a hard limit of the number of expressions that are enumerated for a task.
    /// If this is reached, there may be fewer than `frontier_size` many solutions.
    pub search_limit: usize,
}

/// Enumerate solutions for the given tasks. Considers a "solution" to be any expression with
/// finite log-probability according to a task's oracle. Each task will be associated with at most
/// `frontier_size` many such expressions.
pub fn explore<O>(
    dsl: &DSL,
    state: &mut State,
    tasks: &Vec<Task<O>>,
    recognized: Option<Vec<Productions>>,
) -> Vec<Vec<Expression>> {
    if let Some(productions) = recognized {
        tasks
            .iter()
            .zip(productions)
            .enumerate()
            .map(|(i, (ref t, ref p))| {
                enumerate_solutions(
                    dsl,
                    p,
                    t.tp.clone(),
                    vec![(i, t)],
                    state.frontier_size,
                    state.search_limit,
                ).remove(&i)
                    .unwrap_or(vec![])
            })
            .collect()
    } else {
        let mut tps = HashMap::new();
        for (i, task) in tasks.into_iter().enumerate() {
            tps.entry(&task.tp).or_insert(Vec::new()).push((i, task))
        }
        let mut results = vec![vec![]; tasks.len()];
        for (i, exprs) in tps.into_iter()
            .map(|(tp, tasks)| {
                enumerate_solutions(
                    dsl,
                    &state.productions,
                    tp.clone(),
                    tasks,
                    state.frontier_size,
                    state.search_limit,
                )
            })
            .flat_map(|iter| iter)
        {
            results[i] = exprs
        }
        results
    }
}

fn enumerate_solutions<O>(
    dsl: &DSL,
    productions: &Productions,
    tp: Type,
    mut tasks: Vec<(usize, &Task<O>)>,
    frontier_size: usize,
    search_limit: usize,
) -> HashMap<usize, Vec<Expression>> {
    let mut frontier = HashMap::new();
    let mut searched = 0;
    let mut update = |frontier: &mut HashMap<_, _>, expr: Expression| {
        tasks.retain(|&(i, t)| {
            if (t.oracle)(&expr, dsl).is_finite() {
                let v = frontier.entry(i).or_insert(Vec::new());
                v.push(expr.clone());
                v.len() < frontier_size
            } else {
                true
            }
        });
        if tasks.is_empty() {
            false
        } else {
            searched += 1;
            searched < search_limit
        }
    };
    let ref enumerator = Enumerator::new(tp, dsl, productions);
    for expr in enumerator {
        if !update(&mut frontier, expr) {
            break;
        }
    }
    frontier
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

struct Enumerator<'a> {
    tp: Type,
    dsl: &'a DSL,
    productions: &'a Productions,
    /// budget (lower, upper) in nats, i.e. negative log-likelihood.
    budget: (f64, f64),
}
impl<'a> Enumerator<'a> {
    fn new(tp: Type, dsl: &'a DSL, productions: &'a Productions) -> Self {
        Self {
            tp,
            dsl,
            productions,
            budget: (0f64, 0f64 + BUDGET_INCREMENT),
        }
    }
    fn at_depth(&'a self, depth: f64) -> Box<Iterator<Item = Expression> + 'a> {
        let (l, h) = self.budget;
        let budget = (l + depth, h + depth);
        let ctx = Context::default();
        let env = Rc::new(LinkedList::default());
        Box::new(
            self.enumerator(self.tp.clone(), &ctx, env, budget, 0)
                .map(|(_, _, expr)| expr),
        )
    }
    fn enumerator(
        &'a self,
        request: Type,
        ctx: &Context,
        env: Rc<LinkedList<Type>>,
        budget: (f64, f64),
        depth: u32,
    ) -> Box<Iterator<Item = (f64, Context, Expression)> + 'a> {
        if let Type::Arrow(arrow) = request {
            let env = LinkedList::prepend(env, *arrow.arg);
            let it = self.enumerator(*arrow.ret, ctx, env, budget, depth)
                .map(|(ll, ctx, body)| (ll, ctx, Expression::Abstraction(Box::new(body))));
            Box::new(it)
        } else {
            let it = self.productions
                .candidates(
                    self.dsl,
                    &request,
                    ctx,
                    &LinkedList::as_vecdeque(&env),
                    false,
                )
                .into_iter()
                .filter(move |&(ll, _, _, _)| -ll > budget.1)
                .flat_map(
                    move |(ll, expr, tp, ctx)| -> Box<Iterator<Item = (f64, Context, Expression)> + 'a> {
                        let budget = (budget.0 + ll, budget.1 + ll);
                        if budget.1 <= 0f64 || depth > MAX_DEPTH {
                            Box::new(iter::empty())
                        } else if let Type::Arrow(f_tp) = tp {
                            let arg_tps: VecDeque<Type> = f_tp.args().into_iter().cloned().collect();
                            self.enumerate_application(
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
    fn enumerate_application(
        &'a self,
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
            let it = self.enumerator(arg_tp, ctx, env.clone(), budget, depth)
                .flat_map(move |(arg_ll, ctx, arg)| {
                    let f_next = Expression::Application(Box::new(f.clone()), Box::new(arg));
                    let budget = (budget.0 + arg_ll, budget.1 + arg_ll);
                    self.enumerate_application(
                        &ctx,
                        env.clone(),
                        f_next,
                        f_ll + arg_ll,
                        arg_tps.clone(),
                        budget,
                        depth,
                    )
                });
            Box::new(it)
        }
    }
}
impl<'a> IntoIterator for &'a Enumerator<'a> {
    type Item = Expression;
    type IntoIter = Box<Iterator<Item = Expression> + 'a>;

    fn into_iter(self) -> Self::IntoIter {
        Box::new(
            (0..)
                .map(|n| BUDGET_INCREMENT * (n as f64))
                .flat_map(move |depth| self.at_depth(depth)),
        )
    }
}
