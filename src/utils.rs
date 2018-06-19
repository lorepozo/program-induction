use rand::{
    distributions::{Distribution, Uniform},
    thread_rng,
};
use std::f64;

#[inline(always)]
pub fn logsumexp(lps: &[f64]) -> f64 {
    let largest = lps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let x = lps.iter().map(|lp| (lp - largest).exp()).sum::<f64>().ln();
    largest + x
}

/// Samples an item from `xs` given the weights `ws`.
pub fn weighted_sample<'a, T>(xs: &'a [T], ws: &[f64]) -> &'a T {
    assert_eq!(xs.len(), ws.len(), "weighted sample given invalid inputs");
    let total = ws.iter().fold(0f64, |acc, x| acc + x);
    let threshold: f64 = Uniform::new(0f64, total).sample(&mut thread_rng());
    let mut cum = 0f64;
    for (wp, x) in ws.into_iter().zip(xs) {
        cum += *wp;
        if threshold <= cum {
            return x;
        }
    }
    unreachable!()
}
