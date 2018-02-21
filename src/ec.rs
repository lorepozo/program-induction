use std::collections::{HashMap, VecDeque};
use std::f64;
use std::iter;
use std::rc::Rc;
use polytype::{Context, Type};

use super::{Expression, ProbabilisticDSL, Task};

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
pub fn ec<'a, S, O, RS, R, E, C>(
    prior: &ProbabilisticDSL<'a>,
    mut state: &mut S,
    tasks: &Vec<Task<O>>,
    recognizer: Option<R>,
    explore: E,
    compress: C,
) -> (ProbabilisticDSL<'a>, Vec<Option<Expression>>)
where
    R: FnOnce(&ProbabilisticDSL, &mut S, &Vec<Task<O>>) -> RS,
    E: FnOnce(&ProbabilisticDSL, &mut S, &Vec<Task<O>>, Option<RS>) -> Vec<Vec<Expression>>,
    C: FnOnce(&ProbabilisticDSL, &mut S, &Vec<Task<O>>, Vec<Vec<Expression>>)
        -> (ProbabilisticDSL<'a>, Vec<Option<Expression>>),
{
    let recognized = recognizer.map(|r| r(prior, &mut state, tasks));
    let frontiers = explore(prior, &mut state, tasks, recognized);
    compress(prior, &mut state, tasks, frontiers)
}

pub struct State {
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
    pdsl: &ProbabilisticDSL,
    state: &mut State,
    tasks: &Vec<Task<O>>,
    recognized: Option<Vec<ProbabilisticDSL>>,
) -> Vec<Vec<Expression>> {
    if let Some(productions) = recognized {
        tasks
            .iter()
            .zip(productions)
            .enumerate()
            .map(|(i, (ref t, ref pdsl))| {
                enumerate_solutions(
                    pdsl,
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
                    pdsl,
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
    pdsl: &ProbabilisticDSL,
    tp: Type,
    mut tasks: Vec<(usize, &Task<O>)>,
    frontier_size: usize,
    search_limit: usize,
) -> HashMap<usize, Vec<Expression>> {
    let mut frontier = HashMap::new();
    let mut searched = 0;
    let mut update = |frontier: &mut HashMap<_, _>, expr: Expression| {
        tasks.retain(|&(i, t)| {
            if (t.oracle)(&expr, pdsl.dsl).is_finite() {
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
    let ref enumerator = Enumerator::new(tp, pdsl);
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
    pdsl: &'a ProbabilisticDSL<'a>,
    /// budget (lower, upper) in nats, i.e. negative log-likelihood.
    budget: (f64, f64),
}
impl<'a> Enumerator<'a> {
    fn new(tp: Type, pdsl: &'a ProbabilisticDSL) -> Self {
        Self {
            tp,
            pdsl,
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
            let it = self.pdsl
                .candidates(
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
