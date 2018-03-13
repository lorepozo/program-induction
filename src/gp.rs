//! Representations capable of Genetic Programming.

use std::cmp::Ordering;
use itertools::Itertools;
use polytype::Type;
use rand::{seq, Rng};

use Task;

/// Parameters for genetic programming.
pub struct GPParams {
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
/// [`Task`] provides a fitness function via its [`oracle`].
///
/// [`genesis`]: #tymethod.genesis
/// [`mutate`]: #tymethod.mutate
/// [`crossover`]: #tymethod.crossover
/// [`Task`]: struct.Task.html
/// [`oracle`]: struct.Task.html#structfield.oracle
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
        tp: &Type,
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

    fn evolve<R: Rng, O: Sync>(
        &self,
        params: &Self::Params,
        rng: &mut R,
        gpparams: &GPParams,
        task: &Task<Self, Self::Expression, O>,
        population: &mut Vec<(Self::Expression, f64)>,
    ) {
        let mut new_exprs = Vec::with_capacity(gpparams.n_delta);
        while new_exprs.len() < gpparams.n_delta {
            if rng.gen_range(0f64, 1f64) < gpparams.mutation_prob {
                let parent = self.tournament(rng, gpparams.tournament_size, population);
                let child = self.mutate(params, rng, parent);
                let fitness = (task.oracle)(self, &child);
                new_exprs.push((child, fitness));
            } else {
                let parent1 = self.tournament(rng, gpparams.tournament_size, population);
                let parent2 = self.tournament(rng, gpparams.tournament_size, population);
                let children = self.crossover(params, rng, parent1, parent2);
                let mut scored_children = children
                    .into_iter()
                    .map(|child| {
                        let fitness = (task.oracle)(self, &child);
                        (child, fitness)
                    })
                    .collect();
                new_exprs.append(&mut scored_children);
            }
        }
        new_exprs.truncate(gpparams.n_delta);
        for child in new_exprs {
            sorted_place(child, population)
        }
    }

    fn init_and_evolve<R: Rng, O: Sync>(
        &self,
        params: &Self::Params,
        rng: &mut R,
        gpparams: &GPParams,
        task: &Task<Self, Self::Expression, O>,
        generations: u32,
    ) -> Vec<(Self::Expression, f64)> {
        let mut pop = self.init(params, rng, gpparams, task);
        for _ in 0..generations {
            self.evolve(params, rng, gpparams, task, &mut pop)
        }
        pop
    }
}

fn sorted_place<T>(child: (T, f64), pop: &mut Vec<(T, f64)>) {
    let mut size = pop.len();
    let idx = {
        if size == 0 {
            0
        } else {
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
            if cmp == Ordering::Equal {
                base
            } else {
                base + (cmp == Ordering::Less) as usize
            }
        }
    };
    pop[idx] = child;
}
