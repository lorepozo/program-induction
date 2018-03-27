const BUDGET_INCREMENT: f64 = 2.0;
const MAX_DEPTH: u32 = 512;

use std::collections::VecDeque;
use std::f64;
use std::iter;
use itertools::Itertools;
use polytype::Type;
use rand::distributions::Range;
use rand::Rng;

use super::{AppliedRule, Grammar};

pub fn new<'a>(g: &'a Grammar, nonterminal: Type) -> Box<Iterator<Item = (AppliedRule, f64)> + 'a> {
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
        Box::new(
            enumerate(g, arg_tp, (0f64, budget.1), depth).flat_map(move |(arg, arg_ll)| {
                let mut ar = ar.clone();
                ar.2.push(arg);
                let budget = (budget.0 + arg_ll, budget.1 + arg_ll);
                enumerate_many(g, ar, arg_tps.clone(), budget, depth)
                    .map(move |(x, l)| (x, arg_ll + l))
            }),
        )
    } else if budget.0 < 0f64 && 0f64 <= budget.1 {
        Box::new(iter::once((ar, 0f64)))
    } else {
        Box::new(iter::empty())
    }
}

pub fn sample<R: Rng>(g: &Grammar, tp: &Type, rng: &mut R) -> AppliedRule {
    let mut t = Range::sample_single(0f64, 1.0, rng);
    for (i, r) in g.rules[tp].iter().enumerate() {
        t -= r.logprob.exp();
        if t < 0f64 {
            // selected a rule
            let args = if let Type::Arrow(ref arr) = r.production {
                arr.args()
                    .into_iter()
                    .map(|tp| sample(g, tp, rng))
                    .collect()
            } else {
                vec![]
            };
            return AppliedRule(tp.clone(), i, args);
        }
    }
    panic!("rules were not normalized");
}
