//! (representation) Probabilistic context-free grammar without bound variables or polymorphism.
//!
//! # Examples
//!
//! ```
//! # #[macro_use]
//! # extern crate polytype;
//! # extern crate programinduction;
//! use programinduction::pcfg::{task_by_evaluation, Grammar, Rule};
//!
//! fn evaluator(name: &str, inps: &[i32]) -> Result<i32, ()> {
//!     match name {
//!         "0" => Ok(0),
//!         "1" => Ok(1),
//!         "plus" => Ok(inps[0] + inps[1]),
//!         _ => unreachable!(),
//!     }
//! }
//!
//! # fn main() {
//! let g = Grammar::new(
//!     tp!(EXPR),
//!     vec![
//!         Rule::new("0", tp!(EXPR), 1.0),
//!         Rule::new("1", tp!(EXPR), 1.0),
//!         Rule::new("plus", tp!(@arrow[tp!(EXPR), tp!(EXPR), tp!(EXPR)]), 1.0),
//!     ],
//! );
//!
//! // task: the number 4
//! let task = task_by_evaluation(&evaluator, &4, tp!(EXPR));
//!
//! // solution:
//! let expr = g.parse("plus(1, plus(1, plus(1,1)))").unwrap();
//! assert!((task.oracle)(&g, &expr).is_finite())
//! # }
//! ```

mod enumerator;
mod parser;
pub use self::parser::ParseError;

use crossbeam_channel::bounded;
use itertools::Itertools;
use polytype::{Type, TypeSchema};
use rand::distributions::Range;
use rand::Rng;
use rayon::prelude::*;
use rayon::spawn;
use std::cmp;
use std::collections::HashMap;
use std::f64;
use std::fmt::Debug;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use {ECFrontier, Task, EC, GP};

/// (representation) Probabilistic context-free grammar. Currently cannot handle bound variables or
/// polymorphism.
///
/// Each nonterminal corresponds to a non-polymorphic `Type`.
#[derive(Debug, Clone)]
pub struct Grammar {
    pub start: Type,
    pub rules: HashMap<Type, Vec<Rule>>,
}
impl Grammar {
    /// Rules are normalized according to their associated nonterminal proportional to the supplied
    /// probabilities.
    ///
    /// So each rules' `logprob` is _not_ treated as log-probability in this constructor, they are
    /// treated like un-normalized probabilities.
    pub fn new(start: Type, all_rules: Vec<Rule>) -> Self {
        let mut rules = HashMap::new();
        for mut rule in all_rules {
            let nt = if let Some(ret) = rule.production.returns() {
                ret.clone()
            } else {
                rule.production.clone()
            };
            rule.logprob = rule.logprob.ln();
            rules.entry(nt).or_insert_with(Vec::new).push(rule)
        }
        let mut g = Grammar { start, rules };
        g.normalize();
        g
    }
    /// Enumerate statements in the PCFG, including log-probabilities.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use]
    /// # extern crate polytype;
    /// # extern crate programinduction;
    /// use programinduction::pcfg::{Grammar, Rule, AppliedRule};
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
    /// let exprs: Vec<AppliedRule> = g.enumerate()
    ///     .take(8)
    ///     .map(|(ar, _logprior)| ar)
    ///     .collect();
    ///
    /// assert_eq!(
    ///     exprs,
    ///     vec![
    ///         g.parse("0").unwrap(),
    ///         g.parse("1").unwrap(),
    ///         g.parse("plus(0,0)").unwrap(),
    ///         g.parse("plus(0,1)").unwrap(),
    ///         g.parse("plus(1,0)").unwrap(),
    ///         g.parse("plus(1,1)").unwrap(),
    ///         g.parse("plus(0,plus(0,0))").unwrap(),
    ///         g.parse("plus(0,plus(0,1))").unwrap(),
    ///     ]
    /// );
    /// # }
    /// ```
    pub fn enumerate(&self) -> Box<Iterator<Item = (AppliedRule, f64)>> {
        self.enumerate_nonterminal(self.start.clone())
    }
    /// Enumerate subsentences in the Grammar for the given nonterminal.
    pub fn enumerate_nonterminal(
        &self,
        nonterminal: Type,
    ) -> Box<Iterator<Item = (AppliedRule, f64)>> {
        let (tx, rx) = bounded(1);
        let g = self.clone();
        spawn(move || {
            let tx = tx.clone();
            let termination_condition = &mut |expr, logprior| tx.send((expr, logprior)).is_err();
            enumerator::new(&g, nonterminal, termination_condition)
        });
        Box::new(rx.into_iter())
    }
    /// Set parameters based on supplied sentences. This is performed by [`Grammar::compress`].
    ///
    /// [`Grammar::compress`]: ../trait.EC.html#method.compress
    pub fn update_parameters(&mut self, params: &EstimationParams, sentences: &[AppliedRule]) {
        let mut counts: HashMap<Type, Vec<AtomicUsize>> = HashMap::new();
        // initialize counts to pseudocounts
        for (nt, rs) in &self.rules {
            counts.insert(
                nt.clone(),
                (0..rs.len())
                    .map(|_| AtomicUsize::new(params.pseudocounts as usize))
                    .collect(),
            );
        }
        // update counts based on occurrence
        let counts = Arc::new(counts);
        sentences
            .par_iter()
            .for_each(|ar| update_counts(ar, &counts));
        // assign raw logprobabilities from counts
        for (nt, cs) in Arc::try_unwrap(counts).unwrap() {
            for (i, c) in cs.into_iter().enumerate() {
                self.rules.get_mut(&nt).unwrap()[i].logprob = (c.into_inner() as f64).ln();
            }
        }
        self.normalize();
    }
    /// Evaluate a sentence using a evaluator.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use]
    /// # extern crate polytype;
    /// # extern crate programinduction;
    /// use programinduction::pcfg::{Grammar, Rule, task_by_evaluation};
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
    ///
    /// let expr = g.parse("plus(1, plus(1, plus(1,1)))").unwrap();
    /// assert_eq!(Ok(4), g.eval(&expr, &evaluator));
    /// # }
    /// ```
    pub fn eval<V, E, F>(&self, ar: &AppliedRule, evaluator: &F) -> Result<V, E>
    where
        F: Fn(&str, &[V]) -> Result<V, E>,
    {
        let args = ar.2
            .iter()
            .map(|ar| self.eval(ar, evaluator))
            .collect::<Result<Vec<V>, E>>()?;
        evaluator(self.rules[&ar.0][ar.1].name, &args)
    }
    /// Sample a statement of the PCFG.
    ///
    /// ```
    /// #[macro_use] extern crate polytype;
    /// extern crate programinduction;
    /// extern crate rand;
    /// # fn main() {
    ///
    /// use programinduction::pcfg::{Grammar, Rule};
    ///
    /// let g = Grammar::new(
    ///     tp!(EXPR),
    ///     vec![
    ///         Rule::new("0", tp!(EXPR), 1.0),
    ///         Rule::new("1", tp!(EXPR), 1.0),
    ///         Rule::new("plus", tp!(@arrow[tp!(EXPR), tp!(EXPR), tp!(EXPR)]), 1.0),
    ///     ],
    /// );
    /// let ar = g.sample(&tp!(EXPR), &mut rand::thread_rng());
    /// assert_eq!(&ar.0, &tp!(EXPR));
    /// println!("{}", g.display(&ar));
    /// # }
    /// ```
    pub fn sample<R: Rng>(&self, tp: &Type, rng: &mut R) -> AppliedRule {
        enumerator::sample(self, tp, rng)
    }
    /// Get the log-likelihood of an expansion for the given nonterminal.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # fn main() {
    /// use programinduction::pcfg::{Grammar, Rule};
    ///
    /// let g = Grammar::new(
    ///     tp!(EXPR),
    ///     vec![
    ///         Rule::new("0", tp!(EXPR), 1.0),
    ///         Rule::new("1", tp!(EXPR), 1.0),
    ///         Rule::new("plus", tp!(@arrow[tp!(EXPR), tp!(EXPR), tp!(EXPR)]), 1.0),
    ///         Rule::new("zero?", tp!(@arrow[tp!(EXPR), tp!(BOOL)]), 1.0),
    ///         Rule::new("if", tp!(@arrow[tp!(BOOL), tp!(EXPR), tp!(EXPR)]), 1.0),
    ///         Rule::new("nand", tp!(@arrow[tp!(BOOL), tp!(BOOL), tp!(BOOL)]), 1.0),
    ///     ]
    /// );
    ///
    /// let expr = g.parse("plus(0,0)").unwrap();
    /// assert_eq!(g.likelihood(&expr), -4.1588830833596715);
    ///
    /// let expr = g.parse("if( zero?(plus(0 , 0)), 1, 0)").unwrap();
    /// assert_eq!(g.likelihood(&expr), -7.6246189861593985);
    /// # }
    /// ```
    pub fn likelihood(&self, ar: &AppliedRule) -> f64 {
        self.rules[&ar.0][ar.1].logprob + ar.2.iter().map(|ar| self.likelihood(ar)).sum::<f64>()
    }
    /// Parse a valid sentence in the Grammar. The inverse of [`display`].
    ///
    /// Non-terminating production rules are followed by parentheses containing comma-separated
    /// productions `plus(0, 1)`. Extraneous white space is ignored.
    ///
    /// [`display`]: #method.display
    pub fn parse(&self, inp: &str) -> Result<AppliedRule, ParseError> {
        self.parse_nonterminal(inp, self.start.clone())
    }
    /// Parse a valid subsentence in the Grammar which is producible from the given nonterminal.
    pub fn parse_nonterminal(
        &self,
        inp: &str,
        nonterminal: Type,
    ) -> Result<AppliedRule, ParseError> {
        parser::parse(self, inp, nonterminal)
    }
    /// The inverse of [`parse`].
    ///
    /// [`parse`]: #method.parse
    pub fn display(&self, ar: &AppliedRule) -> String {
        let r = &self.rules[&ar.0][ar.1];
        if r.production.as_arrow().is_some() {
            let args = ar.2.iter().map(|ar| self.display(ar)).join(",");
            format!("{}({})", r.name, args)
        } else {
            format!("{}", r.name)
        }
    }

    fn normalize(&mut self) {
        for rs in self.rules.values_mut() {
            let lp_largest = rs.iter()
                .fold(f64::NEG_INFINITY, |acc, r| acc.max(r.logprob));
            let z = lp_largest
                + rs.iter()
                    .map(|r| (r.logprob - lp_largest).exp())
                    .sum::<f64>()
                    .ln();
            for r in rs {
                r.logprob -= z;
            }
        }
    }
}

/// Parameters for PCFG parameter estimation.
pub struct EstimationParams {
    pub pseudocounts: u64,
}
impl Default for EstimationParams {
    /// The default for PCFG `EstimationParams` prevents completely discarding rules by having
    /// non-zero pseudocounts:
    ///
    /// ```
    /// # use programinduction::pcfg::EstimationParams;
    /// EstimationParams { pseudocounts: 1 }
    /// # ;
    /// ```
    fn default() -> Self {
        EstimationParams { pseudocounts: 1 }
    }
}

impl EC for Grammar {
    type Expression = AppliedRule;
    type Params = EstimationParams;

    fn enumerate<F>(&self, tp: TypeSchema, termination_condition: F)
    where
        F: FnMut(Self::Expression, f64) -> bool,
    {
        match tp {
            TypeSchema::Monotype(tp) => enumerator::new(self, tp, termination_condition),
            _ => panic!("PCFGs can't handle polytypes"),
        }
    }
    /// This is exactly the same as [`Grammar::update_parameters`], but optimized to deal with
    /// frontiers.
    ///
    /// [`Grammar::update_parameters`]: #method.update_parameters
    fn compress<O: Sync>(
        &self,
        params: &Self::Params,
        _tasks: &[Task<Self, Self::Expression, O>],
        frontiers: Vec<ECFrontier<Self>>,
    ) -> (Self, Vec<ECFrontier<Self>>) {
        let mut counts: HashMap<Type, Vec<AtomicUsize>> = HashMap::new();
        // initialize counts to pseudocounts
        for (nt, rs) in &self.rules {
            counts.insert(
                nt.clone(),
                (0..rs.len())
                    .map(|_| AtomicUsize::new(params.pseudocounts as usize))
                    .collect(),
            );
        }
        // update counts based on occurrence
        // NOTE: these lines are the only difference with Grammar::update_parameters
        let counts = Arc::new(counts);
        frontiers
            .par_iter()
            .flat_map(|f| &f.0)
            .for_each(|&(ref ar, _, _)| update_counts(ar, &counts));
        let mut g = self.clone();
        // assign raw logprobabilities from counts
        for (nt, cs) in Arc::try_unwrap(counts).unwrap() {
            for (i, c) in cs.into_iter().enumerate() {
                g.rules.get_mut(&nt).unwrap()[i].logprob = (c.into_inner() as f64).ln();
            }
        }
        g.normalize();
        (g, frontiers)
    }
}

/// Parameters for PCFG genetic programming ([`GP`]).
///
/// Values for each `mutation_` field should be probabilities that sum to 1. Every mutation will
/// randomly select one of these variants.
///
/// [`GP`]: ../trait.GP.html
pub struct GeneticParams {
    pub max_crossover_depth: u32,
    pub mutation_point: f64,
    pub mutation_subtree: f64,
    pub mutation_reproduction: f64,
}

impl GP for Grammar {
    type Expression = AppliedRule;
    type Params = GeneticParams;

    fn genesis<R: Rng>(
        &self,
        _params: &Self::Params,
        rng: &mut R,
        pop_size: usize,
        tp: &TypeSchema,
    ) -> Vec<Self::Expression> {
        let tp = match *tp {
            TypeSchema::Monotype(ref tp) => tp,
            _ => panic!("PCFGs can't handle polytypes"),
        };
        (0..pop_size).map(|_| self.sample(tp, rng)).collect()
    }
    fn mutate<R: Rng>(
        &self,
        params: &Self::Params,
        rng: &mut R,
        prog: &Self::Expression,
    ) -> Self::Expression {
        let tot = params.mutation_point + params.mutation_subtree + params.mutation_reproduction;
        match Range::sample_single(0f64, tot, rng) {
            x if x < params.mutation_point => mutate_random_node(prog.clone(), rng, |ar, rng| {
                let rule = &self.rules[&ar.0][ar.1];
                let mut candidates: Vec<_> = self.rules[&ar.0]
                    .iter()
                    .enumerate()
                    .filter(|&(i, r)| r.production == rule.production && i != ar.1)
                    .map(|(i, _)| i)
                    .collect();
                if candidates.is_empty() {
                    ar
                } else {
                    rng.shuffle(&mut candidates);
                    AppliedRule(ar.0, candidates[0], ar.2)
                }
            }),
            x if x < params.mutation_point + params.mutation_subtree => {
                mutate_random_node(prog.clone(), rng, |ar, rng| self.sample(&ar.0, rng))
            }
            _ => prog.clone(), // reproduction
        }
    }
    fn crossover<R: Rng>(
        &self,
        params: &Self::Params,
        rng: &mut R,
        parent1: &Self::Expression,
        parent2: &Self::Expression,
    ) -> Vec<Self::Expression> {
        // TODO
        let _ = (rng, params);
        vec![parent1.clone(), parent2.clone()]
    }
}

/// Identifies a rule by its location in [`grammar.rules`].
///
/// [`grammar.rules`]: struct.Grammar.html#structfield.rules
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppliedRule(pub Type, pub usize, pub Vec<AppliedRule>);

/// A PCFG rule specifies a production that can happen for a particular nonterminal.
///
/// A rule associates a production name with a production type. Rules that are not arrows are
/// terminals for the supplied nonterminal type. Rules that _are_ arrows expand nonterminals that
/// correspond to the arrow's return type.
///
/// Log-probabilities are normalized when initializing a [`Grammar`].
///
/// [`Grammar`]: struct.Grammar.html
#[derive(Debug, Clone)]
pub struct Rule {
    pub name: &'static str,
    pub production: Type,
    pub logprob: f64,
}
impl Rule {
    pub fn new(name: &'static str, production: Type, logprob: f64) -> Self {
        Rule {
            name,
            production,
            logprob,
        }
    }
}
impl Ord for Rule {
    fn cmp(&self, other: &Rule) -> cmp::Ordering {
        self.partial_cmp(other)
            .expect("logprob for rule is not finite")
    }
}
impl PartialOrd for Rule {
    fn partial_cmp(&self, other: &Rule) -> Option<cmp::Ordering> {
        self.logprob.partial_cmp(&other.logprob)
    }
}
impl PartialEq for Rule {
    fn eq(&self, other: &Rule) -> bool {
        self.name == other.name && self.production == other.production
    }
}
impl Eq for Rule {}

fn update_counts<'a>(ar: &'a AppliedRule, counts: &Arc<HashMap<Type, Vec<AtomicUsize>>>) {
    counts[&ar.0][ar.1].fetch_add(1, Ordering::Relaxed);
    ar.2.iter().for_each(move |ar| update_counts(ar, counts));
}

/// Create a task based on evaluating a PCFG sentence and comparing its output against data.
///
/// Here we let all tasks be represented by an output valued in the space of type `V`. In practice,
/// `V` will often be an enum corresponding to each nonterminal in the PCFG. All outputs and
/// evaluated sentences must be representable by `V`.
///
/// An `evaluator` takes the name of a production and a vector corresponding to evaluated results
/// of each child node of the production in a particular derivation.
///
/// The resulting task is "all-or-nothing": the oracle returns either `0` if all examples are
/// correctly hit or `f64::NEG_INFINITY` otherwise.
///
/// # Examples
///
/// ```
/// # #[macro_use]
/// # extern crate polytype;
/// # extern crate programinduction;
/// use programinduction::pcfg::{task_by_evaluation, Grammar, Rule};
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
///
/// let output = 4;
/// let tp = tp!(EXPR);
/// let task = task_by_evaluation(&evaluator, &output, tp);
///
/// let expr = g.parse("plus(1, plus(1, plus(1,1)))").unwrap();
/// assert!((task.oracle)(&g, &expr).is_finite())
/// # }
/// ```
pub fn task_by_evaluation<'a, V, E, F>(
    evaluator: &'a F,
    output: &'a V,
    tp: Type,
) -> Task<'a, Grammar, AppliedRule, &'a V>
where
    V: PartialEq + Clone + Sync + Debug + 'a,
    F: Fn(&str, &[V]) -> Result<V, E> + Sync + 'a,
{
    let oracle = Box::new(move |g: &Grammar, ar: &AppliedRule| {
        if let Ok(o) = g.eval(ar, evaluator) {
            if o == *output {
                0f64
            } else {
                f64::NEG_INFINITY
            }
        } else {
            f64::NEG_INFINITY
        }
    });
    Task {
        oracle,
        observation: output,
        tp: TypeSchema::Monotype(tp),
    }
}

fn mutate_random_node<R, F>(ar: AppliedRule, rng: &mut R, mutation: F) -> AppliedRule
where
    R: Rng,
    F: Fn(AppliedRule, &mut R) -> AppliedRule,
{
    // TODO: set ar to a random node within the tree (see commented code below)
    mutation(ar, rng)
}
