//! (representation) Probabilistic context-free grammar without bound variables or polymorphism.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::f64;
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
    /// Rules are normalized according to their associated nonterminal.
    pub fn new(start: Type, all_rules: Vec<Rule>) -> Self {
        let mut rules = HashMap::new();
        for rule in all_rules {
            let nt = if let &Type::Arrow(ref arrow) = &rule.production {
                arrow.returns().clone()
            } else {
                rule.production.clone()
            };
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
    /// use programinduction::pcfg::{Grammar, Rule};
    ///
    /// # fn main() {
    /// let g = Grammar::new(
    ///     tp!(EXPR),
    ///     vec![
    ///         Rule::new("0", tp!(EXPR), 0.0),
    ///         Rule::new("1", tp!(EXPR), 0.0),
    ///         Rule::new("plus", arrow![tp!(EXPR), tp!(EXPR), tp!(EXPR)], 0.0),
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
    ///         Rule::new("0", tp!(EXPR), 0.0),
    ///         Rule::new("1", tp!(EXPR), 0.0),
    ///         Rule::new("plus", arrow![tp!(EXPR), tp!(EXPR), tp!(EXPR)], 0.0),
    ///         Rule::new("zero?", arrow![tp!(EXPR), tp!(BOOL)], 0.0),
    ///         Rule::new("if", arrow![tp!(BOOL), tp!(EXPR), tp!(EXPR)], 0.0),
    ///         Rule::new("nand", arrow![tp!(BOOL), tp!(BOOL), tp!(BOOL)], 0.0),
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
    fn enumerate<'a>(&'a self, tp: Type) -> Box<Iterator<Item = (Self::Expression, f64)> + 'a> {
        self.enumerate_nonterminal(tp)
    }
    fn mutate<O: Sync>(&self, tasks: &[Task<Self, O>], frontiers: &[Frontier<Self>]) -> Self {
        let _ = (tasks, frontiers);
        self.clone()
    }
}

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

/// Identifies a rule by its location in [`grammar.rules`].
///
/// [`grammar.rules`]: struct.Grammar.html#structfield.rules
#[derive(Debug, Clone, PartialEq)]
pub struct AppliedRule(pub Type, pub usize, pub Vec<AppliedRule>);

mod enumerator {
    const BUDGET_INCREMENT: f64 = 2.0;
    const MAX_DEPTH: u32 = 512;

    use std::collections::VecDeque;
    use std::iter;
    use itertools::Itertools;
    use polytype::Type;
    use super::{AppliedRule, Grammar};

    pub fn new<'a>(
        g: &'a Grammar,
        nonterminal: Type,
    ) -> Box<Iterator<Item = (AppliedRule, f64)> + 'a> {
        let budget = |offset: f64| (offset, offset + BUDGET_INCREMENT);
        let depth = 0;
        Box::new(
            (0..5)
                .map(|n| BUDGET_INCREMENT * f64::from(n))
                .flat_map(move |offset| enumerate(g, nonterminal.clone(), budget(offset), depth)),
        )
    }

    fn enumerate<'a>(
        g: &'a Grammar,
        tp: Type,
        budget: (f64, f64),
        depth: u32,
    ) -> Box<Iterator<Item = (AppliedRule, f64)> + 'a> {
        if budget.1 <= 0f64 || depth > MAX_DEPTH {
            Box::new(iter::empty())
        } else {
            Box::new(
                g.rules[&tp]
                    .iter()
                    .enumerate()
                    .filter(move |&(_, r)| -r.logprob <= budget.1)
                    .sorted()
                    .into_iter()
                    .flat_map(move |(i, r)| {
                        let ar = AppliedRule(tp.clone(), i, vec![]);
                        let arg_tps: VecDeque<Type> = if let Type::Arrow(ref arr) = r.production {
                            arr.args().into_iter().cloned().collect()
                        } else {
                            VecDeque::new()
                        };
                        let budget = (budget.0 + r.logprob, budget.1 + r.logprob);
                        enumerate_many(g, ar, arg_tps, budget, depth + 1)
                            .map(move |(x, l)| (x, l + r.logprob))
                    }),
            )
        }
    }

    fn enumerate_many<'a>(
        g: &'a Grammar,
        ar: AppliedRule,
        mut arg_tps: VecDeque<Type>,
        budget: (f64, f64),
        depth: u32,
    ) -> Box<Iterator<Item = (AppliedRule, f64)> + 'a> {
        if let Some(arg_tp) = arg_tps.pop_front() {
            Box::new(enumerate(g, arg_tp, (0f64, budget.1), depth).flat_map(
                move |(arg, arg_ll)| {
                    let mut ar = ar.clone();
                    ar.2.push(arg);
                    let budget = (budget.0 + arg_ll, budget.1 + arg_ll);
                    enumerate_many(g, ar, arg_tps.clone(), budget, depth)
                        .map(move |(x, l)| (x, arg_ll + l))
                },
            ))
        } else if budget.0 < 0f64 && 0f64 <= budget.1 {
            Box::new(iter::once((ar, 0f64)))
        } else {
            Box::new(iter::empty())
        }
    }
}

pub use self::parser::ParseError;
mod parser {
    use std::{error, fmt};
    use polytype::Type;
    use nom::types::CompleteStr;
    use super::{AppliedRule, Grammar, Rule};

    #[derive(Clone, Debug)]
    pub enum ParseError {
        InapplicableRule(Type, String),
        NomError(String),
    }
    impl fmt::Display for ParseError {
        fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
            match *self {
                ParseError::InapplicableRule(ref nt, ref s) => {
                    write!(f, "invalide rule {} for nonterminal {}", s, nt)
                }
                ParseError::NomError(ref err) => write!(f, "could not parse: {}", err),
            }
        }
    }
    impl error::Error for ParseError {
        fn description(&self) -> &str {
            "could not parse expression"
        }
    }

    fn location<'a>(
        grammar: &'a Grammar,
        nt: &Type,
        name: &str,
    ) -> Result<(usize, &'a Rule), ParseError> {
        if let Some(rules) = grammar.rules.get(nt) {
            rules
                .iter()
                .enumerate()
                .find(|&(_, r)| r.name == name)
                .ok_or_else(|| ParseError::InapplicableRule(nt.clone(), String::from(name)))
        } else {
            Err(ParseError::InapplicableRule(nt.clone(), String::from(name)))
        }
    }

    #[derive(Debug)]
    struct Item(String, Vec<Item>);
    impl Item {
        fn into_applied(&self, grammar: &Grammar, nt: Type) -> Result<AppliedRule, ParseError> {
            let (loc, r) = location(grammar, &nt, &self.0)?;
            if let Type::Arrow(ref arr) = r.production {
                let inner: Result<Vec<AppliedRule>, ParseError> = self.1
                    .iter()
                    .zip(arr.args())
                    .map(move |(item, nt)| item.into_applied(grammar, nt.clone()))
                    .collect();
                Ok(AppliedRule(nt, loc, inner?))
            } else {
                Ok(AppliedRule(nt, loc, vec![]))
            }
        }
    }

    /// doesn't match parentheses or comma, but matches most ascii printable characters.
    fn alphanumeric_ext(c: char) -> bool {
        (c >= 0x21 as char && c <= 0x7E as char) && !(c == '(' || c == ')' || c == ',')
    }

    named!(var<CompleteStr, Item>,
        do_parse!(
            name: ws!( take_while!(alphanumeric_ext) ) >>
            (Item(name.0.to_string(), vec![]))
        ));
    named!(func<CompleteStr, Item>,
        do_parse!(
            name: ws!( take_while!(alphanumeric_ext) ) >>
            args: delimited!(tag!("("), separated_list!(tag!(","), expr), tag!(")")) >>
            (Item(name.0.to_string(), args))
        ));
    named!(expr<CompleteStr, Item>, alt!(func | var));

    pub fn parse(grammar: &Grammar, input: &str, nt: Type) -> Result<AppliedRule, ParseError> {
        match expr(CompleteStr(input)) {
            Ok((_, item)) => item.into_applied(grammar, nt),
            Err(err) => Err(ParseError::NomError(format!("{:?}", err))),
        }
    }
}
