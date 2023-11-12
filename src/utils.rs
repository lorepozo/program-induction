use rand::{
    distributions::{Distribution, WeightedIndex},
    Rng,
};

#[inline(always)]
pub fn logsumexp(lps: &[f64]) -> f64 {
    let largest = lps.iter().copied().fold(0f64, f64::max);
    let x = lps.iter().map(|lp| (lp - largest).exp()).sum::<f64>().ln();
    largest + x
}

/// Samples `n` items from `xs` weighted by `ws` without replacement.
pub fn weighted_permutation<T: Clone, R: Rng>(
    rng: &mut R,
    xs: &[T],
    ws: &[f64],
    n: Option<usize>,
) -> Vec<T> {
    let sample_size = match n {
        Some(n) => n.min(xs.len()),
        None => xs.len(),
    };
    let mut ws = ws.to_vec();
    let mut permutation = Vec::with_capacity(sample_size);
    for _ in 0..sample_size {
        let dist = WeightedIndex::new(&ws).unwrap();
        let sampled_idx = dist.sample(rng);
        permutation.push(xs[sampled_idx].clone());
        ws[sampled_idx] = 0.0;
    }
    permutation
}
