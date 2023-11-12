use rand::{
    distributions::{Distribution, Uniform},
    Rng,
};
use std::cmp;
use std::f64;

#[inline(always)]
pub fn logsumexp(lps: &[f64]) -> f64 {
    let largest = lps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let x = lps.iter().map(|lp| (lp - largest).exp()).sum::<f64>().ln();
    largest + x
}

pub fn weighted_permutation<T: Clone, R: Rng>(
    rng: &mut R,
    xs: &[T],
    ws: &[f64],
    n: Option<usize>,
) -> Vec<T> {
    let mut ws = ws.to_vec();
    let mut idxs: Vec<_> = (0..(ws.len())).collect();
    let mut permutation = vec![];
    let length = cmp::min(n.unwrap_or(xs.len()), xs.len());
    while permutation.len() < length {
        let jidxs: Vec<_> = idxs.iter().cloned().enumerate().collect();
        let &(jdx, idx): &(usize, usize) = weighted_sample(rng, &jidxs, &ws);
        permutation.push(xs[idx].clone());
        idxs.remove(jdx);
        ws.remove(jdx);
    }
    permutation
}

/// Samples an item from `xs` given the weights `ws`.
pub fn weighted_sample<'a, T, R: Rng>(rng: &mut R, xs: &'a [T], ws: &[f64]) -> &'a T {
    assert_eq!(xs.len(), ws.len(), "weighted sample given invalid inputs");
    let total = ws.iter().fold(0f64, |acc, x| acc + x);
    let threshold: f64 = Uniform::new(0f64, total).sample(rng);
    let mut cum = 0f64;
    for (wp, x) in ws.iter().zip(xs) {
        cum += *wp;
        if threshold <= cum {
            return x;
        }
    }
    unreachable!()
}
