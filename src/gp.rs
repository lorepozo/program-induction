//! Representations capable of Genetic Programming.

use itertools::Itertools;
use polytype::TypeSchema;
use rand::{seq, Rng};
use std::cmp::Ordering;
use utils::{logsumexp, weighted_sample};

use Task;

/// The mechanism by which individuals are selected for inclusion in the
/// population.
#[derive(Deserialize, Serialize)]
pub enum GPSelection {
    /// `Deterministic` implies a strict survival-of-the-fittest selection
    /// mechanism, in which the best individuals are always retained. An
    /// individual can only be removed from a population if a better-scoring
    /// individual arises to take its place.
    #[serde(alias = "deterministic")]
    Deterministic,
    /// `Probabilistic` implies a noisy survival-of-the-fittest selection
    /// mechanism, in which a population is selected probabilistically from a
    /// set of possible populations in proportion to its overall fitness. An
    /// individual can be removed from a population even by lower-scoring
    /// individuals, though this is relatively unlikely.
    #[serde(alias = "probabilistic")]
    Probabilistic,
}

/// Parameters for genetic programming.
#[derive(Deserialize, Serialize)]
pub struct GPParams {
    /// The mechanism by which individuals are selected for inclusion in the
    /// population.
    pub selection: GPSelection,
    pub population_size: usize,
    pub tournament_size: usize,
    /// Probability for a mutation. If mutation doesn't happen, the crossover will happen.
    pub mutation_prob: f64,
    /// The number of new children added to the population with each step of evolution.
    /// Traditionally, this would be set to 1. If it is larger than 1, mutations and crossover will
    /// be repeated until the threshold of `n_delta` is met.
    pub n_delta: usize,
}

/// A kind of representation suitable for **genetic programming**.
///
/// Implementors of `GP` must provide methods for [`genesis`], [`mutate`], [`crossover`]. A
/// [`Task`] provides a fitness function via its [`oracle`]: we adopt the convention that smaller
/// values are better (so one can think of the [`oracle`] as providing a measure of error). To opt
/// out of the default tournament-based selection, implementors may override the [`tournament`]
/// method.
///
/// Typically, you will interact with this trait using the [`init`] and [`evolve`] methods on
/// existing implementations, such as [`pcfg::Grammar`].
///
/// The provided methods (namely [`init`] and [`evolve`]) manipulate a _population_, which is a
/// sorted list of programs according to fitness. If you otherwise modify the population such that
/// it is no longer sorted appropriately, these methods will behave incorrectly.
///
/// # Examples
///
/// This example has a lot of code, but it is quite straightforward. Given a PCFG and an evaluator
/// for sentences, a task is created by measuring proximity of an evaluted sentence to some target.
/// We finally pick some parameters and evolve a population for a few hundred generations.
///
/// ```
/// #[macro_use]
/// extern crate polytype;
/// extern crate programinduction;
/// extern crate rand;
/// use programinduction::pcfg::{self, Grammar, Rule};
/// use programinduction::{GPParams, Task, GP, GPSelection};
/// use rand::{rngs::SmallRng, SeedableRng};
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
/// fn main() {
///     let g = Grammar::new(
///         tp!(EXPR),
///         vec![
///             Rule::new("0", tp!(EXPR), 1.0),
///             Rule::new("1", tp!(EXPR), 1.0),
///             Rule::new("plus", tp!(@arrow[tp!(EXPR), tp!(EXPR), tp!(EXPR)]), 1.0),
///         ],
///     );
///     let target = 6;
///     let task = Task {
///         oracle: Box::new(|g: &Grammar, expr| {
///             if let Ok(n) = g.eval(expr, &evaluator) {
///                 (n - target).abs() as f64 // numbers close to target
///             } else {
///                 std::f64::INFINITY
///             }
///         }),
///         tp: ptp!(EXPR),
///         observation: (),
///     };
///
///     let gpparams = GPParams {
///         selection: GPSelection::Deterministic,
///         population_size: 10,
///         tournament_size: 5,
///         mutation_prob: 0.6,
///         n_delta: 1,
///     };
///     let params = pcfg::GeneticParams::default();
///     let generations = 1000;
///     let rng = &mut SmallRng::from_seed([1u8; 16]);
///
///     let mut pop = g.init(&params, rng, &gpparams, &task);
///     for _ in 0..generations {
///         g.evolve(&params, rng, &gpparams, &task, &mut pop)
///     }
///
///     // perfect winner is found!
///     let &(ref winner, score) = &pop[0];
///     assert_eq!(6, g.eval(winner, &evaluator).unwrap());
///     assert_eq!(0.0, score);
/// }
/// ```
///
/// [`genesis`]: #tymethod.genesis
/// [`mutate`]: #tymethod.mutate
/// [`crossover`]: #tymethod.crossover
/// [`tournament`]: #method.tournament
/// [`init`]: #method.mutate
/// [`evolve`]: #method.crossover
/// [`Task`]: struct.Task.html
/// [`oracle`]: struct.Task.html#structfield.oracle
/// [`pcfg::Grammar`]: pcfg/struct.Grammar.html
pub trait GP: Send + Sync + Sized {
    /// An Expression is a sentence in the representation. **Tasks are solved by Expressions**.
    type Expression: Clone + Send + Sync;
    /// Extra parameters for a representation go here.
    type Params;

    /// Create an initial population for a particular requesting type.
    fn genesis<R: Rng>(
        &self,
        params: &Self::Params,
        rng: &mut R,
        pop_size: usize,
        tp: &TypeSchema,
    ) -> Vec<Self::Expression>;

    /// Mutate a single program.
    fn mutate<R: Rng>(
        &self,
        params: &Self::Params,
        rng: &mut R,
        prog: &Self::Expression,
    ) -> Self::Expression;

    /// Perform crossover between two programs. There must be at least one child.
    fn crossover<R: Rng>(
        &self,
        params: &Self::Params,
        rng: &mut R,
        parent1: &Self::Expression,
        parent2: &Self::Expression,
    ) -> Vec<Self::Expression>;

    /// A tournament selects an individual from a population.
    fn tournament<'a, R: Rng>(
        &self,
        rng: &mut R,
        tournament_size: usize,
        population: &'a [(Self::Expression, f64)],
    ) -> &'a Self::Expression {
        seq::sample_iter(rng, 0..population.len(), tournament_size)
            .expect("tournament size was bigger than population")
            .into_iter()
            .map(|i| &population[i])
            .max_by(|&&(_, ref x), &&(_, ref y)| x.partial_cmp(y).expect("found NaN"))
            .map(|&(ref expr, _)| expr)
            .expect("tournament cannot select winner from no contestants")
    }

    /// Initializes a population, which is a list of programs and their scores sorted by score.
    /// The most-fit individual is the first element in the population.
    fn init<R: Rng, O: Sync>(
        &self,
        params: &Self::Params,
        rng: &mut R,
        gpparams: &GPParams,
        task: &Task<Self, Self::Expression, O>,
    ) -> Vec<(Self::Expression, f64)> {
        let exprs = self.genesis(params, rng, gpparams.population_size, &task.tp);
        exprs
            .into_iter()
            .map(|expr| {
                let l = (task.oracle)(self, &expr);
                (expr, l)
            })
            .sorted_by(|&(_, ref x), &(_, ref y)| x.partial_cmp(y).expect("found NaN"))
    }

    /// Determines whether some newly created offspring is even viable for
    /// consideration as part of the population. This allows you to do things
    /// like ensure a population of unique individuals.
    fn valid_individuals(
        &self,
        params: &Self::Params,
        population: &[(Self::Expression, f64)],
        individuals: Vec<Self::Expression>,
    ) -> Vec<Self::Expression> {
        individuals
    }

    /// Evolves a population. This will repeatedly run a Bernoulli trial with parameter
    /// [`mutation_prob`] and perform mutation or crossover depending on the outcome until
    /// [`n_delta`] expressions are determined.
    ///
    /// [`mutation_prob`]: struct.GPParams.html#mutation_prob
    /// [`n_delta`]: struct.GPParams.html#n_delta
    fn evolve<R: Rng, O: Sync>(
        &self,
        params: &Self::Params,
        rng: &mut R,
        gpparams: &GPParams,
        task: &Task<Self, Self::Expression, O>,
        population: &mut Vec<(Self::Expression, f64)>,
    ) {
        let mut individuals = Vec::with_capacity(gpparams.n_delta);
        while individuals.len() < gpparams.n_delta {
            if rng.gen_bool(gpparams.mutation_prob) {
                let parent = self.tournament(rng, gpparams.tournament_size, population);
                let child = self.mutate(params, rng, parent);
                individuals.push(child);
            } else {
                let parent1 = self.tournament(rng, gpparams.tournament_size, population);
                let parent2 = self.tournament(rng, gpparams.tournament_size, population);
                let mut children = self.crossover(params, rng, parent1, parent2);
                individuals.append(&mut children);
            }
            individuals = self.valid_individuals(params, population, individuals);
        }
        individuals.truncate(gpparams.n_delta);
        let scored_children = individuals
            .into_iter()
            .map(|child| {
                let fitness = (task.oracle)(self, &child);
                (child, fitness)
            })
            .collect();
        match gpparams.selection {
            GPSelection::Probabilistic => sample_pop(scored_children, population),
            GPSelection::Deterministic => {
                for child in scored_children {
                    sorted_place(child, population);
                }
            }
        }
    }
}

/// Given a mutable `Vec`, `pop`, of item-score pairs sorted by score, and a
/// `Vec` of expressions, `new_exprs`, sample a new score-sorted population in
/// inverse proportion to its overall score. The length of `pop` does *not* change.
fn sample_pop<T: Clone>(mut new_exprs: Vec<(T, f64)>, pop: &mut Vec<(T, f64)>) {
    let n = pop.len();
    let mut options = vec![];
    options.append(pop);
    options.append(&mut new_exprs);
    let (idxs, scores): (Vec<usize>, Vec<f64>) = options
        .iter()
        .map(|&(_, score)| score)
        .combinations(n)
        .map(|combo| (-combo.iter().sum::<f64>()))
        .enumerate()
        .unzip();
    let sum_scores = logsumexp(&scores);
    let scores = scores
        .iter()
        .map(|x| (x - sum_scores).exp())
        .collect::<Vec<_>>();
    let idx = weighted_sample(&idxs, &scores);
    *pop = options.into_iter().combinations(n).nth(*idx).unwrap();
}

/// Given a mutable vector, `pop`, of item-score pairs sorted by score, insert
/// an item, `child`, into `pop` while maintaining score-sorted order.
///
/// The length of the list does *not* change, so if the item would be inserted
/// at the end of the list, no insertion occurs. Also, if existing items have
/// the same score as `child`, `child` is inserted *after* these items.
///
/// This function calls `unsafe` methods but in ways that should not fail.
fn sorted_place<T>(child: (T, f64), pop: &mut Vec<(T, f64)>) {
    let orig_size = pop.len();
    let mut size = orig_size;
    if size == 0 {
        return;
    }
    let idx = {
        let mut base = 0usize;
        while size > 1 {
            let half = size / 2;
            let mid = base + half;
            // mid is always in [0, size), that means mid is >= 0 and < size.
            // mid >= 0: by definition
            // mid < size: mid = size / 2 + size / 4 + size / 8 ...
            let other = unsafe { pop.get_unchecked(mid) };
            let cmp = other.1.partial_cmp(&child.1).expect("found NaN");
            base = if cmp == Ordering::Greater { base } else { mid };
            size -= half;
        }
        // base is always in [0, size) because base <= mid.
        let other = unsafe { pop.get_unchecked(base) };
        let cmp = other.1.partial_cmp(&child.1).expect("found NaN");
        base + (cmp != Ordering::Greater) as usize
    };
    if idx < orig_size {
        pop.pop();
        pop.insert(idx, child);
    }
}
