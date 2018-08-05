const BUDGET_INCREMENT: f64 = 2.0;
const MAX_DEPTH: u32 = 512;

use itertools::Itertools;
use polytype::Type;
use rand::Rng;
use std::collections::VecDeque;
use std::f64;

use super::{AppliedRule, Grammar};

pub fn new<F>(g: &Grammar, nonterminal: Type, mut termination_condition: F)
where
    F: FnMut(AppliedRule, f64) -> bool,
{
    let budget = |offset: f64| (offset, offset + BUDGET_INCREMENT);
    let depth = 0;
    let cb = &mut |expr, logprior| !termination_condition(expr, logprior);
    (0..)
        .map(|n| BUDGET_INCREMENT * f64::from(n))
        .all(move |offset| enumerate(g, nonterminal.clone(), budget(offset), depth, cb));
}

/// returns whether the caller should continue enumerating
fn enumerate(
    g: &Grammar,
    tp: Type,
    budget: (f64, f64),
    depth: u32,
    cb: &mut FnMut(AppliedRule, f64) -> bool,
) -> bool {
    if budget.1 <= 0f64 || depth > MAX_DEPTH {
        true
    } else {
        g.rules[&tp]
            .iter()
            .enumerate()
            .filter(move |&(_, r)| -r.logprob <= budget.1)
            .sorted()
            .into_iter()
            .all(move |(i, r)| {
                let ar = AppliedRule(tp.clone(), i, vec![]);
                let arg_tps: VecDeque<Type> = r
                    .production
                    .args()
                    .map(|args| args.into_iter().cloned().collect())
                    .unwrap_or_else(VecDeque::new);
                let budget = (budget.0 + r.logprob, budget.1 + r.logprob);
                enumerate_many(g, ar, arg_tps, budget, r.logprob, depth + 1, cb)
            })
    }
}

fn enumerate_many(
    g: &Grammar,
    ar: AppliedRule,
    mut arg_tps: VecDeque<Type>,
    budget: (f64, f64),
    offset: f64,
    depth: u32,
    cb: &mut FnMut(AppliedRule, f64) -> bool,
) -> bool {
    if let Some(arg_tp) = arg_tps.pop_front() {
        let cb = &mut |arg, ll| {
            let mut ar = ar.clone();
            ar.2.push(arg);
            let arg_tps = arg_tps.clone();
            let budget = (budget.0 + ll, budget.1 + ll);
            let offset = offset + ll;
            enumerate_many(g, ar, arg_tps, budget, offset, depth, cb)
        };
        enumerate(g, arg_tp, (0f64, budget.1), depth, cb)
    } else if budget.0 < 0f64 && 0f64 <= budget.1 {
        cb(ar, offset)
    } else {
        true
    }
}

pub fn sample<R: Rng>(g: &Grammar, tp: &Type, rng: &mut R) -> AppliedRule {
    let mut t: f64 = rng.gen();
    for (i, r) in g.rules[tp].iter().enumerate() {
        t -= r.logprob.exp();
        if t < 0f64 {
            // selected a rule
            let args = if let Some(args) = r.production.args() {
                args.into_iter().map(|tp| sample(g, tp, rng)).collect()
            } else {
                vec![]
            };
            return AppliedRule(tp.clone(), i, args);
        }
    }
    panic!("rules were not normalized");
}
