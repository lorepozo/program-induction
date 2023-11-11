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
use rand::distributions::{Distribution, Uniform};
use rand::seq::SliceRandom;
use rand::Rng;
use rayon::prelude::*;
use rayon::spawn;
use std::cmp;
use std::collections::HashMap;
use std::f64;
use std::fmt::Debug;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::{ECFrontier, Task, EC, GP};

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
    /// use programinduction::pcfg::{AppliedRule, Grammar, Rule};
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
    /// let exprs: Vec<AppliedRule> = g.enumerate().take(8).map(|(ar, _logprior)| ar).collect();
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
    pub fn enumerate(&self) -> Box<dyn Iterator<Item = (AppliedRule, f64)>> {
        self.enumerate_nonterminal(self.start.clone())
    }
    /// Enumerate subsentences in the Grammar for the given nonterminal.
    pub fn enumerate_nonterminal(
        &self,
        nonterminal: Type,
    ) -> Box<dyn Iterator<Item = (AppliedRule, f64)>> {
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
    /// let expr = g.parse("plus(1, plus(1, plus(1,1)))").unwrap();
    /// assert_eq!(Ok(4), g.eval(&expr, &evaluator));
    /// # }
    /// ```
    pub fn eval<V, E, F>(&self, ar: &AppliedRule, evaluator: &F) -> Result<V, E>
    where
        F: Fn(&str, &[V]) -> Result<V, E>,
    {
        let args =
            ar.2.iter()
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
    ///     ],
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
            r.name.to_string()
        }
    }

    fn normalize(&mut self) {
        for rs in self.rules.values_mut() {
            let lp_largest = rs
                .iter()
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
    /// The progeny factor determines the distribution over nodes in a statement when a
    /// node/subtree is randomly selected. If set to `1`, each node has uniform probability of
    /// being chosen for mutation. If set to `2`, then every parent is half as likely to be chosen
    /// than any one of its children.
    pub progeny_factor: f64,
    /// A point mutation replaces a single node with a valid rule that may take its place, without
    /// changing any children.
    pub mutation_point: f64,
    /// A subtree mutation replaces a subtree with a valid sentence that may take its place.
    pub mutation_subtree: f64,
    /// A reproduction mutation is a no-op.
    pub mutation_reproduction: f64,
}
impl Default for GeneticParams {
    fn default() -> GeneticParams {
        GeneticParams {
            progeny_factor: 2f64,
            mutation_point: 0.45,
            mutation_subtree: 0.45,
            mutation_reproduction: 0.1,
        }
    }
}

impl GP for Grammar {
    type Expression = AppliedRule;
    type Params = GeneticParams;
    type Observation = ();

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
        _obs: &Self::Observation,
    ) -> Vec<Self::Expression> {
        let tot = params.mutation_point + params.mutation_subtree + params.mutation_reproduction;
        match Uniform::from(0f64..tot).sample(rng) {
            // point mutation
            x if x < params.mutation_point => {
                vec![mutate_random_node(params, prog.clone(), rng, |ar, rng| {
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
                        candidates.shuffle(rng);
                        AppliedRule(ar.0, candidates[0], ar.2)
                    }
                })]
            }
            // subtree mutation
            x if x < params.mutation_point + params.mutation_subtree => {
                vec![mutate_random_node(params, prog.clone(), rng, |ar, rng| {
                    self.sample(&ar.0, rng)
                })]
            }
            // reproduction
            _ => vec![prog.clone()], // reproduction
        }
    }
    fn crossover<R: Rng>(
        &self,
        params: &Self::Params,
        rng: &mut R,
        parent1: &Self::Expression,
        parent2: &Self::Expression,
        _obs: &Self::Observation,
    ) -> Vec<Self::Expression> {
        vec![
            crossover_random_node(params, parent1, parent2, rng),
            crossover_random_node(params, parent2, parent1, rng),
        ]
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

use self::gp::{crossover_random_node, mutate_random_node};
mod gp {
    use super::{AppliedRule, GeneticParams};
    use polytype::Type;
    use rand::distributions::{Distribution, Uniform};
    use rand::Rng;

    pub fn mutate_random_node<R, F>(
        params: &GeneticParams,
        ar: AppliedRule,
        rng: &mut R,
        mutation: F,
    ) -> AppliedRule
    where
        R: Rng,
        F: Fn(AppliedRule, &mut R) -> AppliedRule,
    {
        let mut arc = WeightedAppliedRule::new(params, ar);
        let mut selection = Uniform::from(0.0..arc.2).sample(rng);
        // selection is like an index in a flattened weighted tree
        {
            let mut cur = &mut arc;
            while selection > 1.0 {
                // subtree
                selection -= 1.0;
                selection /= params.progeny_factor;
                let prev = cur;
                cur = prev
                    .3
                    .iter_mut()
                    .find(|arc| {
                        if selection > arc.2 {
                            selection -= arc.2;
                            false
                        } else {
                            true
                        }
                    })
                    .unwrap();
            }
            // we've selected a node to mutate
            let inp = AppliedRule::from(cur.clone());
            let mutated = mutation(inp, rng);
            *cur = WeightedAppliedRule::new(params, mutated);
        }
        AppliedRule::from(arc)
    }

    pub fn crossover_random_node<R: Rng>(
        params: &GeneticParams,
        parent1: &AppliedRule,
        parent2: &AppliedRule,
        rng: &mut R,
    ) -> AppliedRule {
        mutate_random_node(params, parent1.clone(), rng, |ar, rng| {
            // find node in parent2 with type ar.0
            let mut viables = Vec::new();
            fetch_subtrees_with_type(params, parent2, &ar.0, 0, &mut viables);
            if viables.is_empty() {
                ar
            } else {
                let total = viables.iter().map(|&(weight, _)| weight).sum();
                let mut idx = Uniform::from(0f64..total).sample(rng);
                viables
                    .into_iter()
                    .find(|&(weight, _)| {
                        if idx > weight {
                            idx -= weight;
                            false
                        } else {
                            true
                        }
                    })
                    .unwrap()
                    .1
                    .clone()
            }
        })
    }

    fn fetch_subtrees_with_type<'a>(
        params: &GeneticParams,
        ar: &'a AppliedRule,
        tp: &Type,
        depth: usize,
        viables: &mut Vec<(f64, &'a AppliedRule)>,
    ) {
        if &ar.0 == tp {
            viables.push((params.progeny_factor.powf(depth as f64), ar))
        }
        for ar in &ar.2 {
            fetch_subtrees_with_type(params, ar, tp, depth + 1, viables)
        }
    }

    #[derive(Debug, Clone)]
    struct WeightedAppliedRule(Type, usize, f64, Vec<WeightedAppliedRule>);
    impl WeightedAppliedRule {
        fn new(params: &GeneticParams, ar: AppliedRule) -> Self {
            if ar.2.is_empty() {
                WeightedAppliedRule(ar.0, ar.1, 1.0, vec![])
            } else {
                let children: Vec<_> =
                    ar.2.into_iter()
                        .map(|ar| WeightedAppliedRule::new(params, ar))
                        .collect();
                let children_weight: f64 = children.iter().map(|arc| arc.2).sum();
                let weight = 1.0 + params.progeny_factor * children_weight;
                WeightedAppliedRule(ar.0, ar.1, weight, children)
            }
        }
    }
    impl From<WeightedAppliedRule> for AppliedRule {
        fn from(arc: WeightedAppliedRule) -> Self {
            if arc.3.is_empty() {
                AppliedRule(arc.0, arc.1, vec![])
            } else {
                let children = arc.3.into_iter().map(Self::from).collect();
                AppliedRule(arc.0, arc.1, children)
            }
        }
    }
}
