//! Representations capable of Exploration-Compression.

use crossbeam_channel::bounded;
use polytype::TypeSchema;
use rayon::prelude::*;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::Duration;

use crate::Task;

/// Parameters for the EC algorithm.
///
/// The first of these limits/timeouts to be hit determines termination of enumeration. It is
/// dangerous to have both search limits set to `None`!
pub struct ECParams {
    /// The maximum frontier size; the number of task solutions to be hit before enumeration is
    /// stopped for a particular task.
    pub frontier_limit: usize,
    /// A timeout before enumeration is stopped, run independently per distinct `TypeSchema` being
    /// enumerated. If this is reached, there may be fewer than `frontier_limit` many solutions.
    pub search_limit_timeout: Option<Duration>,
    /// An approximate limit on enumerated description length. If this is reached, there may be
    /// fewer than `frontier_limit` many solutions.
    pub search_limit_description_length: Option<f64>,
}

/// A kind of representation suitable for **exploration-compression**.
///
/// For details on the EC algorithm, see the module-level documentation [here].
///
/// Implementors of `EC` need only provide an [`enumerate`] and [`compress`] methods. By doing so,
/// we provide the [`ec`], [`ec_with_recognition`], and [`explore`] methods.
///
/// Typically, you will interact with this trait via existing implementations, such as with
/// [`lambda::Language`] or [`pcfg::Grammar`].
///
/// # Examples
///
/// Using an existing domain in the lambda calculus representation [`lambda::Language`]:
///
/// ```ignore
/// use programinduction::domains::circuits;
/// use programinduction::{lambda, ECParams, EC};
///
/// fn main() {
///     let mut dsl = circuits::dsl();
///     let tasks = circuits::make_tasks(250);
///     let ec_params = ECParams {
///         frontier_limit: 10,
///         search_limit_timeout: None,
///         search_limit_description_length: Some(9.0),
///     };
///     let params = lambda::CompressionParams::default();
///
///     let mut frontiers = Vec::new();
///     for _ in 0..5 {
///         let (new_dsl, new_frontiers) = dsl.ec(&ec_params, &params, &tasks);
///         dsl = new_dsl;
///         frontiers = new_frontiers;
///     }
///     let n_invented = dsl.invented.len();
///     let n_hit = frontiers.iter().filter(|f| !f.is_empty()).count();
///     println!(
///         "hit {} of {} using {} invented primitives",
///         n_hit,
///         tasks.len(),
///         n_invented,
///     );
/// }
/// ```
///
/// [here]: index.html#bayesian-program-learning-with-the-ec-algorithm
/// [`enumerate`]: #tymethod.enumerate
/// [`compress`]: #tymethod.compress
/// [`ec`]: #method.ec
/// [`ec_with_recognition`]: #method.ec_with_recognition
/// [`explore`]: #method.explore
/// [`lambda::Language`]: lambda/struct.Language.html
/// [`pcfg::Grammar`]: pcfg/struct.Grammar.html
pub trait EC: Send + Sync + Sized {
    /// An Expression is a sentence in the representation. Tasks are solved by Expressions.
    type Expression: Clone + Send + Sync;
    /// Many representations have some parameters for compression. They belong here.
    type Params;

    /// Iterate over [`Expression`]s for a given type, with their corresponding log-priors, until a
    /// condition is met. This enumeration should be best-first: the log-prior of enumerated
    /// expressions should generally increase, so simple expressions are enumerated first.
    ///
    /// The `termination_condition` acts as a callback for each enumerated `(Expression, f64)`.
    /// If it responds with true, enumeration must stop (i.e. this method call should terminate).
    ///
    /// [`Expression`]: #associatedtype.Expression
    fn enumerate<F>(&self, tp: TypeSchema, termination_condition: F)
    where
        F: Fn(Self::Expression, f64) -> bool + Send + Sync;
    /// Update the representation based on findings of expressions that solve [`Task`]s.
    ///
    /// The `frontiers` argument, and similar return value, must always be of the same size as
    /// `tasks`. Each frontier is a possibly-empty list of expressions that solve the corresponding
    /// task, and the log-prior and log-likelihood for that expression.
    ///
    /// [`Task`]: struct.Task.html
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
    /// ```ignore
    /// use programinduction::domains::circuits;
    /// use programinduction::{lambda, ECParams, EC};
    ///
    /// # fn main() {
    /// let mut dsl = circuits::dsl();
    /// let tasks = circuits::make_tasks(250);
    /// let ec_params = ECParams {
    ///     frontier_limit: 10,
    ///     search_limit_timeout: None,
    ///     search_limit_description_length: Some(8.0),
    /// };
    /// let params = lambda::CompressionParams::default();
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
        if cfg!(feature = "verbose") {
            eprintln!(
                "EXPLORE-COMPRESS: explored {} frontiers with {} hits",
                frontiers.len(),
                frontiers.iter().filter(|f| !f.is_empty()).count()
            )
        }
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
    /// use polytype::tp;
    /// use programinduction::pcfg::{Grammar, Rule, task_by_evaluation};
    /// use programinduction::{EC, ECParams};
    ///
    /// fn evaluator(name: &str, inps: &[i32]) -> Result<i32, ()> {
    ///     match name {
    ///         "0" => Ok(0),
    ///         "1" => Ok(1),
    ///         "plus" => Ok(inps[0] + inps[1]),
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
    ///         Rule::new("plus", tp!(@arrow[tp!(EXPR), tp!(EXPR), tp!(EXPR)]), 1.0),
    ///     ],
    /// );
    /// let ec_params = ECParams {
    ///     frontier_limit: 1,
    ///     search_limit_timeout: Some(std::time::Duration::new(1, 0)),
    ///     search_limit_description_length: None,
    /// };
    /// // task: the number 4
    /// let task = task_by_evaluation(&evaluator, &4, tp!(EXPR));
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
        for (i, task) in tasks.iter().enumerate() {
            tps.entry(&task.tp).or_insert_with(Vec::new).push((i, task))
        }
        let mut results: Vec<ECFrontier<Self>> =
            (0..tasks.len()).map(|_| ECFrontier::default()).collect();
        {
            let mutex = Arc::new(Mutex::new(&mut results));
            tps.into_par_iter()
                .flat_map(|(tp, tasks)| enumerate_solutions(self, ec_params, tp.clone(), tasks))
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
                    .pop()
                    .unwrap()
                    .1
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
    tp: TypeSchema,
    tasks: Vec<(usize, &Task<L, X, O>)>,
) -> Vec<(usize, ECFrontier<L>)>
where
    X: Send + Sync + Clone,
    L: EC<Expression = X>,
{
    // initialization
    let frontiers: Vec<_> = tasks // associate task id with task and frontier
        .into_iter()
        .map(|(j, t)| (j, Some(t), ECFrontier::default()))
        .collect();
    let frontiers = Arc::new(RwLock::new(frontiers));

    // termination conditions
    let mut timeout_complete: Box<dyn Fn() -> bool + Send + Sync> = Box::new(|| false);
    let (tx, rx) = bounded(1);
    if let Some(duration) = params.search_limit_timeout {
        thread::spawn(move || {
            thread::sleep(duration);
            tx.send(()).unwrap_or(())
        });
        timeout_complete = Box::new(move || rx.try_recv().is_ok());
    }
    let mut dl_complete: Box<dyn Fn(f64) -> bool + Send + Sync> = Box::new(|_| false);
    if let Some(dl) = params.search_limit_description_length {
        dl_complete = Box::new(move |logprior| -logprior > dl);
    }
    let is_terminated = Arc::new(RwLock::new(false));

    // update frontiers and check for termination
    let termination_condition = {
        let frontiers = Arc::clone(&frontiers);
        move |expr: X, logprior: f64| {
            {
                if *is_terminated.read().unwrap() {
                    return true;
                }
            }
            let hits: Vec<_> = frontiers
                .read()
                .expect("enumeration frontiers poisoned")
                .iter()
                .enumerate()
                .filter_map(|(i, &(_, ref ot, _))| ot.as_ref().map(|t| (i, t))) // only check incomplete tasks
                .filter_map(|(i, t)| {
                    let l = (t.oracle)(repr, &expr);
                    if l.is_finite() {
                        Some((i, expr.clone(), logprior, l))
                    } else {
                        None
                    }
                })
                .collect();
            if !hits.is_empty() {
                let mut frontiers = frontiers.write().expect("enumeration frontiers poisoned");
                for (i, expr, logprior, l) in hits {
                    frontiers[i].2.push(expr, logprior, l);
                    if frontiers[i].2.len() >= params.frontier_limit {
                        frontiers[i].1 = None
                    }
                }
            }
            let mut is_terminated = is_terminated.write().unwrap();
            if *is_terminated
                | frontiers
                    .read()
                    .expect("enumeration frontiers poisoned")
                    .is_empty()
                | timeout_complete()
                | dl_complete(logprior)
            {
                *is_terminated = true;
                true
            } else {
                false
            }
        }
    };

    repr.enumerate(tp, termination_condition);
    if let Ok(l) = Arc::try_unwrap(frontiers) {
        let frontiers = l.into_inner().expect("enumeration frontiers poisoned");
        frontiers.into_iter().map(|(j, _, f)| (j, f)).collect()
    } else {
        panic!("enumeration lifetime exceeded its scope")
    }
}

/// A set of expressions which solve a task.
///
/// Stores tuples of [`Expression`], log-prior, and log-likelihood.
///
/// [`Expression`]: trait.EC.html#associatedtype.Expression
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
