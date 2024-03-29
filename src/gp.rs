//! Representations capable of Genetic Programming.

use itertools::Itertools;
use polytype::TypeScheme;
use rand::{distributions::Distribution, distributions::WeightedIndex, seq::SliceRandom, Rng};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

use crate::utils::weighted_permutation;
use crate::Task;

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
    /// `Drift(alpha)` implies a noisy survival-of-the-fittest selection
    /// mechanism, in which individuals are selected probabilistically without
    /// replacement from the combination of the population and offspring to form
    /// a new population. Offspring fitness is simply determined by the task,
    /// while the fitness of members of the pre-existing population is computed
    /// as a linear combination of their prior fitness and fitness on the
    /// current task, given as `fitness_{:t} = alpha * fitness_{:t-1} +
    /// (1-alpha) * fitness_t`. An individual can be removed from a population
    /// by lower-scoring individuals, though this is relatively unlikely.
    #[serde(alias = "drift")]
    Drift(f64),
    /// `Hybrid(deterministic_proportion)` implies a selection mechanism in which
    /// some portion of the population is selected deterministically such that the
    /// best individuals are always retained. The remainder of the population is
    /// sampled without replacement from the remaining individuals. An individual
    /// can be removed from a population by lower-scoring individuals, though this
    /// is relatively unlikely, and impossible if the individual is considered one
    /// of the "best" in the population. The number of "best" individuals is
    /// `floor(population.len() * deterministic_proportion)`.
    /// The `deterministic_proportion` should be between 0 and 1.
    #[serde(alias = "hybrid")]
    Hybrid(f64),
    /// `Probabilistic` implies a noisy survival-of-the-fittest selection
    /// mechanism, in which a population is selected probabilistically from a
    /// set of possible populations in proportion to its overall fitness. An
    /// individual can be removed from a population even by lower-scoring
    /// individuals, though this is relatively unlikely.
    #[serde(alias = "probabilistic")]
    Probabilistic,
    /// `Resample` implies that individuals are selected by sampling from the
    /// offspring *with* replacement, as in a particle filter.
    Resample,
}

impl GPSelection {
    pub(crate) fn update_population<R: Rng, X: Clone>(
        &self,
        population: &mut Vec<(X, f64)>,
        children: Vec<X>,
        oracle: impl Fn(&X) -> f64,
        rng: &mut R,
    ) {
        let mut scored_children = children
            .into_iter()
            .map(|child| {
                let fitness = oracle(&child);
                (child, fitness)
            })
            .collect_vec();
        match self {
            GPSelection::Drift(alpha) => {
                for (p, old_fitness) in population.iter_mut() {
                    let new_fitness = oracle(p);
                    *old_fitness = *alpha * *old_fitness + (1.0 - alpha) * new_fitness;
                }
                let pop_size = population.len();
                population.extend(scored_children);
                *population = sample_without_replacement(population, pop_size, rng);
            }
            GPSelection::Resample => {
                let pop_size = population.len();
                *population = sample_with_replacement(&scored_children, pop_size, rng);
            }
            GPSelection::Deterministic => {
                for child in scored_children {
                    sorted_place(child, population);
                }
            }
            GPSelection::Hybrid(_) | GPSelection::Probabilistic => {
                let pop_size = population.len();
                let mut options = Vec::with_capacity(pop_size + scored_children.len());
                options.append(population);
                options.append(&mut scored_children);
                let mut sample_size = pop_size;
                if let GPSelection::Hybrid(det_proportion) = self {
                    let n_best = (pop_size as f64 * det_proportion).ceil() as usize;
                    sample_size -= n_best;
                    options.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                    let rest = options.split_off(n_best);
                    *population = options;
                    options = rest;
                }
                population.append(&mut sample_pop(rng, options, sample_size));
            }
        }
    }
}

/// Parameters for genetic programming.
#[derive(Deserialize, Serialize)]
pub struct GPParams {
    /// The mechanism by which individuals are selected for inclusion in the
    /// population.
    pub selection: GPSelection,
    pub population_size: usize,
    // The number of individuals selected uniformly at random to participate in
    // a tournament. If 1, a single individual is selected uniformly at random,
    // as if the population were unweighted. This is useful for mimicking
    // uniform weights after resampling, as in a particle filter.
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
/// use polytype::{ptp, tp};
/// use programinduction::{
///     simple_task, GP, GPParams, GPSelection,
///     pcfg::{self, Grammar, Rule}
/// };
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
/// let g = Grammar::new(
///     tp!(EXPR),
///     vec![
///         Rule::new("0", tp!(EXPR), 1.0),
///         Rule::new("1", tp!(EXPR), 1.0),
///         Rule::new("plus", tp!(@arrow[tp!(EXPR), tp!(EXPR), tp!(EXPR)]), 1.0),
///     ],
/// );
/// let target = 6;
/// let task = simple_task(|g: &Grammar, expr| {
///     if let Ok(n) = g.eval(expr, &evaluator) {
///         (n - target).abs() as f64 // numbers close to target
///     } else {
///         f64::INFINITY
///     }
/// }, ptp!(EXPR));
///
/// let gpparams = GPParams {
///     selection: GPSelection::Deterministic,
///     population_size: 10,
///     tournament_size: 5,
///     mutation_prob: 0.6,
///     n_delta: 1,
/// };
/// let params = pcfg::GeneticParams::default();
/// let generations = 1000;
/// let rng = &mut SmallRng::from_seed([1u8; 32]);
///
/// let mut pop = g.init(&params, rng, &gpparams, &task);
/// for _ in 0..generations {
///     g.evolve(&params, rng, &gpparams, &task, &mut pop)
/// }
///
/// // perfect winner is found!
/// let (winner, score) = &pop[0];
/// assert_eq!(6, g.eval(winner, &evaluator).unwrap());
/// assert_eq!(0.0, *score);
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
pub trait GP<Observation: ?Sized> {
    /// An Expression is a sentence in the representation. **Tasks are solved by Expressions**.
    type Expression: Clone;
    /// Extra parameters for a representation go here.
    type Params;

    /// Create an initial population for a particular requesting type.
    fn genesis<R: Rng>(
        &self,
        params: &Self::Params,
        rng: &mut R,
        pop_size: usize,
        tp: &TypeScheme,
    ) -> Vec<Self::Expression>;

    /// Mutate a single program, potentially producing multiple offspring
    fn mutate<R: Rng>(
        &self,
        params: &Self::Params,
        rng: &mut R,
        prog: &Self::Expression,
        obs: &Observation,
    ) -> Vec<Self::Expression>;

    /// Perform crossover between two programs. There must be at least one child.
    fn crossover<R: Rng>(
        &self,
        params: &Self::Params,
        rng: &mut R,
        parent1: &Self::Expression,
        parent2: &Self::Expression,
        obs: &Observation,
    ) -> Vec<Self::Expression>;

    /// A tournament selects an individual from a population.
    fn tournament<'a, R: Rng>(
        &self,
        rng: &mut R,
        tournament_size: usize,
        population: &'a [(Self::Expression, f64)],
    ) -> &'a Self::Expression {
        let tribute = if tournament_size == 1 {
            population.choose(rng)
        } else {
            population
                .choose_multiple(rng, tournament_size)
                .max_by(|&(_, x), &(_, y)| x.partial_cmp(y).expect("found NaN"))
        };
        tribute
            .map(|(expr, _)| expr)
            .expect("tournament cannot select winner from no contestants")
    }

    /// Initializes a population, which is a list of programs and their scores sorted by score.
    /// The most-fit individual is the first element in the population.
    fn init<R: Rng>(
        &self,
        params: &Self::Params,
        rng: &mut R,
        gpparams: &GPParams,
        task: &impl Task<Observation, Representation = Self, Expression = Self::Expression>,
    ) -> Vec<(Self::Expression, f64)> {
        let exprs = self.genesis(params, rng, gpparams.population_size, task.tp());
        exprs
            .into_iter()
            .map(|expr| {
                let l = task.oracle(self, &expr);
                (expr, l)
            })
            .sorted_by(|(_, x), (_, y)| x.partial_cmp(y).expect("found NaN"))
            .collect()
    }

    /// This should be a filter-like operation on `offspring`. The intended
    /// semantics is that `validate_offspring` reduces the set of newly created
    /// individuals in `offspring` to just those viable for consideration as
    /// part of the total `population`, taking into account other `children`
    /// that are part of the current generation. This allows you to do things
    /// like ensure a population of unique individuals.
    fn validate_offspring(
        &self,
        _params: &Self::Params,
        _population: &[(Self::Expression, f64)],
        _children: &[Self::Expression],
        _offspring: &mut Vec<Self::Expression>,
    ) {
    }

    /// Evolves a population. This will repeatedly run a Bernoulli trial with parameter
    /// [`mutation_prob`] and perform mutation or crossover depending on the outcome until
    /// [`n_delta`] expressions are determined.
    ///
    /// [`mutation_prob`]: struct.GPParams.html#mutation_prob
    /// [`n_delta`]: struct.GPParams.html#n_delta
    fn evolve<R: Rng>(
        &self,
        params: &Self::Params,
        rng: &mut R,
        gpparams: &GPParams,
        task: &impl Task<Observation, Representation = Self, Expression = Self::Expression>,
        population: &mut Vec<(Self::Expression, f64)>,
    ) {
        let mut children = Vec::with_capacity(gpparams.n_delta);
        while children.len() < gpparams.n_delta {
            let mut offspring = if rng.gen_bool(gpparams.mutation_prob) {
                let parent = self.tournament(rng, gpparams.tournament_size, population);
                self.mutate(params, rng, parent, task.observation())
            } else {
                let parent1 = self.tournament(rng, gpparams.tournament_size, population);
                let parent2 = self.tournament(rng, gpparams.tournament_size, population);
                self.crossover(params, rng, parent1, parent2, task.observation())
            };
            self.validate_offspring(params, population, &children, &mut offspring);
            children.append(&mut offspring);
        }
        children.truncate(gpparams.n_delta);
        gpparams.selection.update_population(
            population,
            children,
            |child| task.oracle(self, child),
            rng,
        );
    }
}

/// Given a `Vec` of item-score pairs sorted by score, and some `sample_size`,
/// return a score-sorted sample selected in proportion to its overall score.
fn sample_pop<T: Clone, R: Rng>(
    rng: &mut R,
    options: Vec<(T, f64)>,
    sample_size: usize,
) -> Vec<(T, f64)> {
    let mut sample = options
        .choose_multiple_weighted(rng, sample_size, |(_, score)| *score)
        .expect("bad weight")
        .cloned()
        .collect::<Vec<_>>();
    sample.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    sample
}

/// Given a `Vec` of item-score pairs sorted by score, and some `sample_size`,
/// return a score-sorted subset sampled without replacement from the `Vec`
/// according to score.
fn sample_without_replacement<R: Rng, T: Clone>(
    options: &[(T, f64)],
    sample_size: usize,
    rng: &mut R,
) -> Vec<(T, f64)> {
    let weights = options.iter().map(|(_, weight)| *weight).collect_vec();
    let mut sample = weighted_permutation(rng, options, &weights, Some(sample_size));
    sample.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    sample
}

/// Given a `Vec` of item-score pairs sorted by score, and some `sample_size`,
/// return a score-sorted subset sampled with replacement from the `Vec`
/// according to score.
fn sample_with_replacement<R: Rng, T: Clone>(
    options: &[(T, f64)],
    sample_size: usize,
    rng: &mut R,
) -> Vec<(T, f64)> {
    let dist = WeightedIndex::new(options.iter().map(|(_, weight)| *weight)).unwrap();
    let mut sample = Vec::with_capacity(sample_size);
    for _ in 0..sample_size {
        // cloning because we don't know if we'll be using multiple times
        sample.push(options[dist.sample(rng)].clone());
    }
    sample.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    sample
}

/// Given a mutable vector, `pop`, of item-score pairs sorted by score, insert
/// an item, `child`, into `pop` while maintaining score-sorted order.
///
/// The length of the list does *not* change, so if the item would be inserted
/// at the end of the list, no insertion occurs. If existing items have
/// the same score as `child`, `child` is inserted somewhere deterministically
/// among those items.
fn sorted_place<T>(child: (T, f64), pop: &mut Vec<(T, f64)>) {
    let r = pop.binary_search_by(|probe| probe.1.partial_cmp(&child.1).expect("found NaN"));
    let idx = match r {
        Ok(found) => found,
        Err(insertion_point) => insertion_point,
    };
    if idx < pop.len() {
        pop.pop();
        pop.insert(idx, child);
    }
}
