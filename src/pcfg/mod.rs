//! (representation) Probabilistic context-free grammar without bound variables or polymorphism.
//!
//! # Examples
//!
//! ```
//! # #[macro_use]
//! # extern crate polytype;
//! # extern crate programinduction;
//! use programinduction::pcfg::{Grammar, Rule, task_by_simple_evaluation};
//!
//! fn simple_evaluator(name: &str, inps: &[i32]) -> i32 {
//!     match name {
//!         "0" => 0,
//!         "1" => 1,
//!         "plus" => inps[0] + inps[1],
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
//!         Rule::new("plus", arrow![tp!(EXPR), tp!(EXPR), tp!(EXPR)], 1.0),
//!     ],
//! );
//!
//! // task: the number 4
//! let task = task_by_simple_evaluation(&simple_evaluator, &4, tp!(EXPR));
//!
//! // solution:
//! let expr = g.parse("plus(1, plus(1, plus(1,1)))").unwrap();
//! assert!((task.oracle)(&g, &expr).is_finite())
//! # }
//! ```

mod enumerator;
mod parser;
pub use self::parser::ParseError;

use std::cmp;
use std::collections::HashMap;
use std::f64;
use std::fmt::Debug;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use itertools::Itertools;
use polytype::Type;
use rayon::prelude::*;
use super::{Frontier, InferenceError, Representation, Task, EC};

/// Probabilistic context-free grammar. Currently cannot handle bound variables or polymorphism.
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
            let nt = if let Type::Arrow(ref arrow) = rule.production {
                arrow.returns().clone()
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
    ///         Rule::new("plus", arrow![tp!(EXPR), tp!(EXPR), tp!(EXPR)], 1.0),
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
    pub fn enumerate<'a>(&'a self) -> Box<Iterator<Item = (AppliedRule, f64)> + 'a> {
        self.enumerate_nonterminal(self.start.clone())
    }
    /// Enumerate subsentences in the Grammar for the given nonterminal.
    pub fn enumerate_nonterminal<'a>(
        &'a self,
        tp: Type,
    ) -> Box<Iterator<Item = (AppliedRule, f64)> + 'a> {
        enumerator::new(self, tp)
    }
    /// Set parameters based on supplied sentences. This is performed by [`Grammar::compress`].
    ///
    /// [`Grammar::compress`]: ../trait.EC.html#method.compress
    pub fn update_parameters(&mut self, params: &Params, sentences: &[AppliedRule]) {
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
    /// Evaluate a sentence using a simple evaluator.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use]
    /// # extern crate polytype;
    /// # extern crate programinduction;
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
    ///
    /// let expr = g.parse("plus(1, plus(1, plus(1,1)))").unwrap();
    /// assert_eq!(4, g.eval(&expr, &simple_evaluator));
    /// # }
    /// ```
    pub fn eval<V, F>(&self, ar: &AppliedRule, simple_evaluator: &F) -> V
    where
        F: Fn(&str, &[V]) -> V,
    {
        let args: Vec<V> = ar.2
            .iter()
            .map(|ar| self.eval(ar, simple_evaluator))
            .collect();
        simple_evaluator(self.rules[&ar.0][ar.1].name, &args)
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
    ///         Rule::new("plus", arrow![tp!(EXPR), tp!(EXPR), tp!(EXPR)], 1.0),
    ///         Rule::new("zero?", arrow![tp!(EXPR), tp!(BOOL)], 1.0),
    ///         Rule::new("if", arrow![tp!(BOOL), tp!(EXPR), tp!(EXPR)], 1.0),
    ///         Rule::new("nand", arrow![tp!(BOOL), tp!(BOOL), tp!(BOOL)], 1.0),
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
        if let Type::Arrow(_) = r.production {
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
impl Representation for Grammar {
    type Expression = AppliedRule;

    fn infer(&self, expr: &Self::Expression) -> Result<Type, InferenceError> {
        Ok(expr.0.clone())
    }
    fn display(&self, expr: &Self::Expression) -> String {
        self.display(expr)
    }
}
impl EC for Grammar {
    type Params = Params;

    fn enumerate<'a>(&'a self, tp: Type) -> Box<Iterator<Item = (Self::Expression, f64)> + 'a> {
        self.enumerate_nonterminal(tp)
    }
    /// This is exactly the same as [`Grammar::update_parameters`], but optimized to deal with
    /// frontiers.
    ///
    /// [`Grammar::update_parameters`]: #method.update_parameters
    fn compress<O: Sync>(
        &self,
        params: &Self::Params,
        _tasks: &[Task<Self, O>],
        frontiers: &[Frontier<Self>],
    ) -> Self {
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
        g
    }
}

/// Parameters for PCFG parameter estimation.
pub struct Params {
    pub pseudocounts: u64,
}
impl Default for Params {
    /// The default for PCFG `Params` prevents completely discarding rules by having non-zero
    /// pseudocounts:
    ///
    /// ```
    /// # use programinduction::pcfg::Params;
    /// Params { pseudocounts: 1 }
    /// # ;
    /// ```
    fn default() -> Self {
        Params { pseudocounts: 1 }
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
/// A `simple_evaluator` takes the name of a production and a vector corresponding to evaluated results
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
///
/// let output = 4;
/// let tp = tp!(EXPR);
/// let task = task_by_simple_evaluation(&simple_evaluator, &output, tp);
///
/// let expr = g.parse("plus(1, plus(1, plus(1,1)))").unwrap();
/// assert!((task.oracle)(&g, &expr).is_finite())
/// # }
/// ```
pub fn task_by_simple_evaluation<'a, V, F>(
    simple_evaluator: &'a F,
    output: &'a V,
    tp: Type,
) -> Task<'a, Grammar, &'a V>
where
    V: PartialEq + Clone + Sync + Debug + 'a,
    F: Fn(&str, &[V]) -> V + Sync + 'a,
{
    let oracle = Box::new(move |g: &Grammar, ar: &AppliedRule| {
        if output == &g.eval(ar, simple_evaluator) {
            0f64
        } else {
            f64::NEG_INFINITY
        }
    });
    Task {
        oracle,
        observation: output,
        tp,
    }
}
