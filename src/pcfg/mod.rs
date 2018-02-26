//! (representation) Probabilistic context-free grammar without bound variables or polymorphism.

mod enumerator;
mod parser;
pub use self::parser::ParseError;

use std::cmp::Ordering;
use std::collections::HashMap;
use std::f64;
use std::fmt::Debug;
use itertools::Itertools;
use polytype::Type;
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
            let nt = if let &Type::Arrow(ref arrow) = &rule.production {
                arrow.returns().clone()
            } else {
                rule.production.clone()
            };
            rule.logprob = rule.logprob.ln();
            rules.entry(nt).or_insert_with(Vec::new).push(rule)
        }
        for rs in rules.values_mut() {
            let p_largest = rs.iter()
                .fold(f64::NEG_INFINITY, |acc, r| acc.max(r.logprob));
            let z = p_largest
                + rs.iter()
                    .map(|r| (r.logprob - p_largest).exp())
                    .sum::<f64>()
                    .ln();
            for r in rs {
                r.logprob = r.logprob - z;
            }
        }
        Grammar { start, rules }
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
    /// Evaluate a sentence using an evaluator.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use]
    /// # extern crate polytype;
    /// # extern crate programinduction;
    /// use programinduction::pcfg::{Grammar, Rule, task_by_output};
    ///
    /// fn evaluator(name: &str, inps: &[i32]) -> i32 {
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
    /// assert_eq!(4, g.eval(&expr, &evaluator));
    /// # }
    /// ```
    pub fn eval<V, F>(&self, ar: &AppliedRule, evaluator: &F) -> V
    where
        F: Fn(&str, &[V]) -> V,
    {
        let args: Vec<V> = ar.2.iter().map(|ar| self.eval(ar, evaluator)).collect();
        evaluator(self.rules[&ar.0][ar.1].name, &args)
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
    /// Parse a valid sentence in the Grammar. The inverse of [`stringify`].
    ///
    /// Non-terminating production rules are followed by parentheses containing comma-separated
    /// productions `plus(0, 1)`. Extraneous white space is ignored.
    ///
    /// [`stringify`]: #method.stringify
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
    pub fn stringify(&self, ar: &AppliedRule) -> String {
        let r = &self.rules[&ar.0][ar.1];
        if let Type::Arrow(_) = r.production {
            let args = ar.2.iter().map(|ar| self.stringify(ar)).join(",");
            format!("{}({})", r.name, args)
        } else {
            format!("{}", r.name)
        }
    }
}
impl Representation for Grammar {
    type Expression = AppliedRule;

    fn infer(&self, expr: &Self::Expression) -> Result<Type, InferenceError> {
        Ok(expr.0.clone())
    }
}
impl EC for Grammar {
    type Params = ();

    fn enumerate<'a>(&'a self, tp: Type) -> Box<Iterator<Item = (Self::Expression, f64)> + 'a> {
        self.enumerate_nonterminal(tp)
    }
    fn mutate<O: Sync>(
        &self,
        params: &Self::Params,
        tasks: &[Task<Self, O>],
        frontiers: &[Frontier<Self>],
    ) -> Self {
        let _ = (params, tasks, frontiers);
        self.clone()
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
    fn cmp(&self, other: &Rule) -> Ordering {
        self.partial_cmp(&other)
            .expect("logprob for rule is not finite")
    }
}
impl PartialOrd for Rule {
    fn partial_cmp(&self, other: &Rule) -> Option<Ordering> {
        self.logprob.partial_cmp(&other.logprob)
    }
}
impl PartialEq for Rule {
    fn eq(&self, other: &Rule) -> bool {
        self.name == other.name && self.production == other.production
    }
}
impl Eq for Rule {}

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
/// use programinduction::pcfg::{Grammar, Rule, task_by_output};
///
/// fn evaluator(name: &str, inps: &[i32]) -> i32 {
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
/// let task = task_by_output(&evaluator, &output, tp);
///
/// let expr = g.parse("plus(1, plus(1, plus(1,1)))").unwrap();
/// assert!((task.oracle)(&g, &expr).is_finite())
/// # }
/// ```
pub fn task_by_output<'a, V, F>(
    evaluator: &'a F,
    output: &'a V,
    tp: Type,
) -> Task<'a, Grammar, &'a V>
where
    V: PartialEq + Clone + Sync + Debug + 'a,
    F: Fn(&str, &[V]) -> V + Sync + 'a,
{
    let oracle = Box::new(move |g: &Grammar, ar: &AppliedRule| {
        if output == &g.eval(ar, evaluator) {
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
