use std::collections::HashMap;
use polytype::Type;

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
