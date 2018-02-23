use std::collections::HashMap;
use polytype::Type;

use super::{Representation, Task};

pub struct Params {
    /// frontier size is the number of task solutions to be hit before enumeration is stopped.
    pub frontier_size: usize,
    /// search limit is a hard limit of the number of expressions that are enumerated for a task.
    /// If this is reached, there may be fewer than `frontier_size` many solutions.
    pub search_limit: usize,
}

/// A kind of representation suitable for EC, and default methods for exploration-compression.
///
/// Implementors of `EC` need only provide an [`enumerate`] and [`mutate`] methods.
///
/// [`enumerate`]: #method.enumerator
/// [`mutate`]: #method.mutate
pub trait EC: Representation {
    /// Get an iterator over [`Expression`]s for a given type.
    ///
    /// This will in most cases iterate infinitely, giving increasingly complex expressions.
    ///
    /// [`Expression`]: trait.Representation.html#associatedtype.Expression
    fn enumerate<'a>(&'a self, tp: Type) -> Box<Iterator<Item = Self::Expression> + 'a>;
    /// Update the [`Representation`] based on findings of expressions that solve [`Task`]s.
    ///
    /// The `frontiers` argument must always be of the same size as `tasks`. Each frontier is a
    /// possibly-empty list of expressions that solve the corresponding task.
    ///
    /// [`Representation`]: trait.Representation.html
    /// [`Task`]: struct.Task.html
    fn mutate<O>(&self, tasks: &[Task<Self, O>], frontiers: &[Vec<Self::Expression>]) -> Self;

    // provided methods:

    /// The entry point for one iteration of the EC algorithm.
    fn ec<O, R>(
        &self,
        params: &Params,
        tasks: &[Task<Self, O>],
        recognizer: Option<R>,
    ) -> (Self, Vec<Option<<Self as Representation>::Expression>>)
    where
        R: FnOnce(&Self, &[Task<Self, O>]) -> Vec<Self>,
    {
        let recognized = recognizer.map(|r| r(self, tasks));
        let frontiers = self.explore(params, tasks, recognized);
        let updated = self.mutate(tasks, &frontiers);
        let solutions = tasks
            .iter()
            .zip(frontiers)
            .map(|(t, frontier)| {
                frontier
                    .into_iter()
                    .map(|expr| ((t.oracle)(self, &expr), expr))
                    .max_by(|&(ref x, _), &(ref y, _)| x.partial_cmp(y).unwrap())
                    .map(|(_ll, expr)| expr)
            })
            .collect();
        (updated, solutions)
    }

    /// Enumerate solutions for the given tasks.
    ///
    /// Considers a "solution" to be any expression with finite log-probability according to a
    /// task's oracle.
    ///
    /// Each task will be associated with at most `params.frontier_size` many such expressions, and
    /// enumeration is stopped when `params.search_limit` valid expressions have been checked.
    fn explore<O>(
        &self,
        params: &Params,
        tasks: &[Task<Self, O>],
        recognized: Option<Vec<Self>>,
    ) -> Vec<Vec<<Self as Representation>::Expression>> {
        if let Some(representations) = recognized {
            tasks
                .iter()
                .zip(representations)
                .enumerate()
                .map(|(i, (t, ref repr))| {
                    repr.enumerate_solutions(params, t.tp.clone(), vec![(i, t)])
                        .remove(&i)
                        .unwrap_or_else(Vec::new)
                })
                .collect()
        } else {
            let mut tps = HashMap::new();
            for (i, task) in tasks.into_iter().enumerate() {
                tps.entry(&task.tp).or_insert_with(Vec::new).push((i, task))
            }
            let mut results = vec![vec![]; tasks.len()];
            for (i, exprs) in tps.into_iter()
                .map(|(tp, tasks)| self.enumerate_solutions(params, tp.clone(), tasks))
                .flat_map(|iter| iter)
            {
                results[i] = exprs
            }
            results
        }
    }

    /// Enumerate solutions for the given tasks which all accord to the given type.
    ///
    /// Considers a "solution" to be any expression with finite log-probability according to a
    /// task's oracle.
    ///
    /// Each task will be associated with at most `params.frontier_size` many such expressions, and
    /// enumeration is stopped when `params.search_limit` valid expressions have been checked.
    fn enumerate_solutions<O>(
        &self,
        params: &Params,
        tp: Type,
        mut tasks: Vec<(usize, &Task<Self, O>)>,
    ) -> HashMap<usize, Vec<<Self as Representation>::Expression>> {
        let mut frontier = HashMap::new();
        let mut searched = 0;
        let mut update = |frontier: &mut HashMap<_, _>,
                          expr: <Self as Representation>::Expression| {
            tasks.retain(|&(i, t)| {
                if (t.oracle)(self, &expr).is_finite() {
                    let v = frontier.entry(i).or_insert_with(Vec::new);
                    v.push(expr.clone());
                    v.len() < params.frontier_size
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
        for expr in self.enumerate(tp) {
            if !update(&mut frontier, expr) {
                break;
            }
        }
        frontier
    }
}
