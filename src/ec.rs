//! The Exploration-Compression algorithm.

use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};
use polytype::Type;
use rayon::prelude::*;

use super::{Representation, Task};

/// A set of expressions which solve a task.
///
/// Stores tuples of [`Expression`], log-prior, and log-likelihood.
///
/// [`Expression`]: trait.Representation.html#associatetype.Expression
#[derive(Clone, Debug)]
pub struct Frontier<R: Representation>(pub Vec<(R::Expression, f64, f64)>);
impl<R: Representation> Frontier<R> {
    pub fn push(&mut self, expr: R::Expression, log_prior: f64, log_likelihood: f64) {
        self.0.push((expr, log_prior, log_likelihood))
    }
    pub fn best_solution(&self) -> Option<&(R::Expression, f64, f64)> {
        self.0
            .iter()
            .max_by(|&&(_, _, ref x), &&(_, _, ref y)| x.partial_cmp(y).unwrap())
    }
}
impl<R: Representation> Default for Frontier<R> {
    fn default() -> Self {
        Frontier(vec![])
    }
}
impl<R: Representation> Deref for Frontier<R> {
    type Target = Vec<(R::Expression, f64, f64)>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<R: Representation> DerefMut for Frontier<R> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Parameters for the EC algorithm.
pub struct ECParams {
    /// The maximum frontier size; the number of task solutions to be hit before enumeration is
    /// stopped for a particular task.
    pub frontier_limit: usize,
    /// search limit is a hard limit of the number of expressions that are enumerated for a task.
    /// If this is reached, there may be fewer than `frontier_limit` many solutions.
    pub search_limit: usize,
}

/// A kind of [`Representation`] suitable for EC, and default methods for exploration-compression.
///
/// Implementors of `EC` need only provide an [`enumerate`] and [`mutate`] methods.
///
/// [`Representation`]: trait.Representation.html
/// [`enumerate`]: #method.enumerator
/// [`mutate`]: #method.mutate
pub trait EC: Representation {
    type Params;

    /// Get an iterator over [`Expression`]s for a given type and corresponding log-priors.
    /// This enumeration should be best-first: the log-prior of enumerated expressions should not
    /// significantly decrease.
    ///
    /// This will in most cases iterate infinitely, giving increasingly complex expressions.
    ///
    /// [`Expression`]: ../trait.Representation.html#associatedtype.Expression
    fn enumerate<'a>(&'a self, tp: Type) -> Box<Iterator<Item = (Self::Expression, f64)> + 'a>;
    /// Update the [`Representation`] based on findings of expressions that solve [`Task`]s.
    ///
    /// The `frontiers` argument must always be of the same size as `tasks`. Each frontier is a
    /// possibly-empty list of expressions that solve the corresponding task, and the log-prior and
    /// log-likelihood for that expression.
    ///
    /// [`Representation`]: ../trait.Representation.html
    /// [`Task`]: ../struct.Task.html
    fn mutate<O: Sync>(
        &self,
        params: &Self::Params,
        tasks: &[Task<Self, O>],
        frontiers: &[Frontier<Self>],
    ) -> Self;

    // provided methods:

    /// The entry point for one iteration of the EC algorithm.
    ///
    /// Returned solutions include the log-prior and log-likelihood of successful expressions.
    fn ec<O: Sync>(
        &self,
        ecparams: &ECParams,
        params: &Self::Params,
        tasks: &[Task<Self, O>],
    ) -> (Self, Vec<Frontier<Self>>) {
        let frontiers = self.explore(ecparams, tasks, None);
        let updated = self.mutate(params, tasks, &frontiers);
        (updated, frontiers)
    }

    /// The entry point for one iteration of the EC algorithm with a recognizer.
    ///
    /// The recognizer supplies a representation for every task which is then used for
    /// exploration-compression.
    ///
    /// Returned solutions include the log-prior and log-likelihood of successful expressions.
    fn ec_with_recognition<O: Sync, R>(
        &self,
        ecparams: &ECParams,
        params: &Self::Params,
        tasks: &[Task<Self, O>],
        recognizer: R,
    ) -> (Self, Vec<Frontier<Self>>)
    where
        R: FnOnce(&Self, &[Task<Self, O>]) -> Vec<Self>,
    {
        let recognized = recognizer(self, tasks);
        let frontiers = self.explore(ecparams, tasks, Some(recognized));
        let updated = self.mutate(params, tasks, &frontiers);
        (updated, frontiers)
    }

    /// Enumerate solutions for the given tasks.
    ///
    /// Considers a "solution" to be any expression with finite log-probability according to a
    /// task's oracle.
    ///
    /// Each task will be associated with at most `params.frontier_limit` many such expressions,
    /// and enumeration is stopped when `params.search_limit` valid expressions have been checked.
    fn explore<O: Sync>(
        &self,
        params: &ECParams,
        tasks: &[Task<Self, O>],
        recognized: Option<Vec<Self>>,
    ) -> Vec<Frontier<Self>> {
        if let Some(representations) = recognized {
            tasks
                .par_iter()
                .zip(representations)
                .enumerate()
                .map(|(i, (t, ref repr))| {
                    repr.enumerate_solutions(params, t.tp.clone(), vec![(i, t)])
                        .remove(&i)
                        .unwrap_or_default()
                })
                .collect()
        } else {
            let mut tps = HashMap::new();
            for (i, task) in tasks.into_iter().enumerate() {
                tps.entry(&task.tp).or_insert_with(Vec::new).push((i, task))
            }
            let mut results: Vec<Frontier<Self>> =
                (0..tasks.len()).map(|_| Frontier::default()).collect();
            {
                let mutex = Arc::new(Mutex::new(&mut results));
                tps.into_par_iter()
                    .map(|(tp, tasks)| self.enumerate_solutions(params, tp.clone(), tasks))
                    .flat_map(|iter| iter)
                    .for_each(move |(i, frontier)| {
                        let mut results = mutex.lock().unwrap();
                        results[i] = frontier
                    });
            }
            results
        }
    }

    /// Enumerate solutions for the given tasks which all accord to the given type.
    ///
    /// Considers a "solution" to be any expression with finite log-probability according to a
    /// task's oracle.
    ///
    /// Each task will be associated with at most `params.frontier_limit` many such expressions,
    /// and enumeration is stopped when `params.search_limit` valid expressions have been checked.
    fn enumerate_solutions<O: Sync>(
        &self,
        params: &ECParams,
        tp: Type,
        mut tasks: Vec<(usize, &Task<Self, O>)>,
    ) -> HashMap<usize, Frontier<Self>> {
        let mut frontiers = HashMap::new();
        let mut searched = 0;
        let mut update = |frontiers: &mut HashMap<_, _>,
                          expr: <Self as Representation>::Expression,
                          log_prior: f64| {
            tasks.retain(|&(i, t)| {
                let log_likelihood = (t.oracle)(self, &expr);
                if log_likelihood.is_finite() {
                    let f = frontiers.entry(i).or_insert_with(Frontier::default);
                    f.push(expr.clone(), log_prior, log_likelihood);
                    f.len() < params.frontier_limit
                } else {
                    true
                }
            });
            if tasks.is_empty() {
                false
            } else {
                searched += 1;
                searched < params.search_limit
            }
        };
        for (expr, log_prior) in self.enumerate(tp) {
            if !update(&mut frontiers, expr, log_prior) {
                break;
            }
        }
        frontiers
    }
}
