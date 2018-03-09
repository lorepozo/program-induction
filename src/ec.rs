//! Representations capable of Exploration-Compression.

use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};
use polytype::Type;
use rayon::prelude::*;

use Task;

/// Parameters for the EC algorithm.
pub struct ECParams {
    /// The maximum frontier size; the number of task solutions to be hit before enumeration is
    /// stopped for a particular task.
    pub frontier_limit: usize,
    /// search limit is a hard limit of the number of expressions that are enumerated for a task.
    /// If this is reached, there may be fewer than `frontier_limit` many solutions.
    pub search_limit: usize,
}

/// A kind of representation suitable for **exploration-compression**.
///
/// Implementors of `EC` need only provide an [`enumerate`] and [`compress`] methods. By doing so,
/// we provide the [`ec`], [`ec_with_recognition`], and [`explore`] methods.
///
/// Typically, you will interact with this trait via existing implementations, such as with
/// [`lambda::Language`] or [`pcfg::Grammar`].
///
/// [`enumerate`]: #method.enumerator
/// [`compress`]: #method.compress
/// [`ec`]: #method.ec
/// [`ec_with_recognition`]: #method.ec_with_recognition
/// [`explore`]: #method.explore
/// [`lambda::Language`]: lambda/struct.Language.html
/// [`pcfg::Grammar`]: pcfg/struct.Grammar.html
pub trait EC: Send + Sync + Sized {
    /// An Expression is a sentence in the representation. Tasks are solved by Expressions.
    type Expression: Clone + Send;
    /// Many representations have some parameters for compression. They belong here.
    type Params;

    /// Get an iterator over [`Expression`]s for a given type and corresponding log-priors.
    /// This enumeration should be best-first: the log-prior of enumerated expressions should
    /// generally increase, so simple expressions are enumerated first.
    ///
    /// This will in most cases iterate infinitely, giving increasingly complex expressions.
    ///
    /// [`Expression`]: ../trait.EC.html#associatedtype.Expression
    fn enumerate<'a>(&'a self, tp: Type) -> Box<Iterator<Item = (Self::Expression, f64)> + 'a>;
    /// Update the representation based on findings of expressions that solve [`Task`]s.
    ///
    /// The `frontiers` argument, and similar return value, must always be of the same size as
    /// `tasks`. Each frontier is a possibly-empty list of expressions that solve the corresponding
    /// task, and the log-prior and log-likelihood for that expression.
    ///
    /// [`Task`]: ../struct.Task.html
    fn compress<O: Sync>(
        &self,
        params: &Self::Params,
        tasks: &[Task<Self, Self::Expression, O>],
        frontiers: Vec<ECFrontier<Self>>,
    ) -> (Self, Vec<ECFrontier<Self>>);

    // provided methods:

    /// The entry point for one iteration of the EC algorithm.
    ///
    /// Returned solutions include the log-prior and log-likelihood of successful expressions.
    ///
    /// # Examples
    ///
    /// ```
    /// # extern crate programinduction;
    /// use programinduction::{lambda, ECParams, EC};
    /// use programinduction::domains::circuits;
    ///
    /// # fn main() {
    /// let mut dsl = circuits::dsl();
    /// let tasks = circuits::make_tasks(250);
    /// let ec_params = ECParams {
    ///     frontier_limit: 10,
    ///     search_limit: 1000,
    /// };
    /// let params = lambda::Params::default();
    ///
    /// let mut frontiers = Vec::new();
    /// for i in 1..6 {
    ///     println!("running EC iteration {}", i);
    ///
    ///     let (new_dsl, new_frontiers) = dsl.ec(&ec_params, &params, &tasks);
    ///     dsl = new_dsl;
    ///     frontiers = new_frontiers;
    ///
    ///     let n_hit = frontiers.iter().filter(|f| !f.is_empty()).count();
    ///     println!("hit {} of {}", n_hit, tasks.len());
    /// }
    /// assert!(!dsl.invented.is_empty());
    /// for &(ref expr, _, _) in &dsl.invented {
    ///     println!("invented {}", dsl.display(expr))
    /// }
    /// # }
    /// ```
    ///
    fn ec<O: Sync>(
        &self,
        ecparams: &ECParams,
        params: &Self::Params,
        tasks: &[Task<Self, Self::Expression, O>],
    ) -> (Self, Vec<ECFrontier<Self>>) {
        let frontiers = self.explore(ecparams, tasks);
        self.compress(params, tasks, frontiers)
    }

    /// The entry point for one iteration of the EC algorithm with a recognizer, very similar to
    /// [`ec`].
    ///
    /// The recognizer supplies a representation for every task which is then used for
    /// exploration-compression.
    ///
    /// Returned solutions include the log-prior and log-likelihood of successful expressions.
    ///
    /// [`ec`]: #method.ec
    fn ec_with_recognition<O: Sync, R>(
        &self,
        ecparams: &ECParams,
        params: &Self::Params,
        tasks: &[Task<Self, Self::Expression, O>],
        recognizer: R,
    ) -> (Self, Vec<ECFrontier<Self>>)
    where
        R: FnOnce(&Self, &[Task<Self, Self::Expression, O>]) -> Vec<Self>,
    {
        let recognized = recognizer(self, tasks);
        let frontiers = self.explore_with_recognition(ecparams, tasks, &recognized);
        self.compress(params, tasks, frontiers)
    }

    /// Enumerate solutions for the given tasks.
    ///
    /// Considers a "solution" to be any expression with finite log-probability according to a
    /// task's oracle.
    ///
    /// Each task will be associated with at most `params.frontier_limit` many such expressions,
    /// and enumeration is stopped when `params.search_limit` valid expressions have been checked.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use]
    /// # extern crate polytype;
    /// # extern crate programinduction;
    /// use programinduction::{EC, ECParams};
    /// use programinduction::pcfg::{Grammar, Rule, task_by_simple_evaluation};
    ///
    /// fn simple_evaluator(name: &str, inps: &[i32]) -> i32 {
    ///     match name {
    ///         "0" => 0,
    ///         "1" => 1,
    ///         "plus" => inps[0] + inps[1],
    ///         _ => unreachable!(),
    ///     }
    /// }
    ///
    /// # fn main() {
    /// let g = Grammar::new(
    ///     tp!(EXPR),
    ///     vec![
    ///         Rule::new("0", tp!(EXPR), 1.0),
    ///         Rule::new("1", tp!(EXPR), 1.0),
    ///         Rule::new("plus", arrow![tp!(EXPR), tp!(EXPR), tp!(EXPR)], 1.0),
    ///     ],
    /// );
    /// let ec_params = ECParams {
    ///     frontier_limit: 1,
    ///     search_limit: 50,
    /// };
    /// // task: the number 4
    /// let task = task_by_simple_evaluation(&simple_evaluator, &4, tp!(EXPR));
    ///
    /// let frontiers = g.explore(&ec_params, &[task]);
    /// assert!(frontiers[0].best_solution().is_some());
    /// # }
    /// ```
    fn explore<O: Sync>(
        &self,
        ec_params: &ECParams,
        tasks: &[Task<Self, Self::Expression, O>],
    ) -> Vec<ECFrontier<Self>> {
        let mut tps = HashMap::new();
        for (i, task) in tasks.into_iter().enumerate() {
            tps.entry(&task.tp).or_insert_with(Vec::new).push((i, task))
        }
        let mut results: Vec<ECFrontier<Self>> =
            (0..tasks.len()).map(|_| ECFrontier::default()).collect();
        {
            let mutex = Arc::new(Mutex::new(&mut results));
            tps.into_par_iter()
                .map(|(tp, tasks)| enumerate_solutions(self, ec_params, tp.clone(), tasks))
                .flat_map(|iter| iter)
                .for_each(move |(i, frontier)| {
                    let mut results = mutex.lock().unwrap();
                    results[i] = frontier
                });
        }
        results
    }

    /// Like [`explore`], but with specific "recognized" representations for each task.
    ///
    /// [`explore`]: #method.explore
    fn explore_with_recognition<O: Sync>(
        &self,
        ec_params: &ECParams,
        tasks: &[Task<Self, Self::Expression, O>],
        representations: &[Self],
    ) -> Vec<ECFrontier<Self>> {
        tasks
            .par_iter()
            .zip(representations)
            .enumerate()
            .map(|(i, (t, repr))| {
                enumerate_solutions(repr, ec_params, t.tp.clone(), vec![(i, t)])
                    .remove(&i)
                    .unwrap_or_default()
            })
            .collect()
    }
}

/// Enumerate solutions for the given tasks which all accord to the given type.
///
/// Considers a "solution" to be any expression with finite log-probability according to a
/// task's oracle.
///
/// Each task will be associated with at most `params.frontier_limit` many such expressions,
/// and enumeration is stopped when `params.search_limit` valid expressions have been checked.
fn enumerate_solutions<L, X, O: Sync>(
    repr: &L,
    params: &ECParams,
    tp: Type,
    mut tasks: Vec<(usize, &Task<L, X, O>)>,
) -> HashMap<usize, ECFrontier<L>>
where
    X: Send + Clone,
    L: EC<Expression = X>,
{
    let mut frontiers = HashMap::new();
    let mut searched = 0;
    let mut update = |frontiers: &mut HashMap<_, _>, expr: L::Expression, log_prior: f64| {
        tasks.retain(|&(i, t)| {
            let log_likelihood = (t.oracle)(repr, &expr);
            if log_likelihood.is_finite() {
                let f = frontiers.entry(i).or_insert_with(ECFrontier::default);
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
    for (expr, log_prior) in repr.enumerate(tp) {
        if !update(&mut frontiers, expr, log_prior) {
            break;
        }
    }
    frontiers
}

/// A set of expressions which solve a task.
///
/// Stores tuples of [`Expression`], log-prior, and log-likelihood.
///
/// [`Expression`]: trait.EC.html#associatetype.Expression
#[derive(Clone, Debug)]
pub struct ECFrontier<L: EC>(pub Vec<(L::Expression, f64, f64)>);
impl<L: EC> ECFrontier<L> {
    pub fn push(&mut self, expr: L::Expression, log_prior: f64, log_likelihood: f64) {
        self.0.push((expr, log_prior, log_likelihood))
    }
    pub fn best_solution(&self) -> Option<&(L::Expression, f64, f64)> {
        self.0
            .iter()
            .max_by(|&&(_, xp, xl), &&(_, yp, yl)| (xp + xl).partial_cmp(&(yp + yl)).unwrap())
    }
}
impl<L: EC> Default for ECFrontier<L> {
    fn default() -> Self {
        ECFrontier(vec![])
    }
}
impl<L: EC> Deref for ECFrontier<L> {
    type Target = Vec<(L::Expression, f64, f64)>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<L: EC> DerefMut for ECFrontier<L> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
