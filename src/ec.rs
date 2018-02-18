use std::collections::{HashMap, VecDeque};
use std::f64;
use polytype::{Context, Type};

use super::{Expression, Task, DSL};

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
                enumerate_tp(
                    dsl,
                    p,
                    &t.tp,
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
                enumerate_tp(
                    dsl,
                    &state.productions,
                    &tp,
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

fn enumerate_tp<O>(
    _dsl: &DSL,
    _productions: &Productions,
    _tp: &Type,
    _tasks: Vec<(usize, &Task<O>)>,
    _frontier_size: usize,
    _search_limit: usize,
) -> HashMap<usize, Vec<Expression>> {
    HashMap::new()
}
