use rand::random;
use std::f64::NEG_INFINITY;

/// Returns the log probability of selecting `n` items from `xs` uniformly at random with replacement.
pub fn log_n_of<T>(xs: &[T], n: usize, empty: f64) -> f64 {
    if xs.is_empty() {
        empty.ln()
    } else {
        (n as f64).ln() - (xs.len() as f64).ln()
    }
}

/// Samples an item from `xs` given the probabilities `ps`.
pub fn weighted_sample<T: Clone>(xs: &[T], ps: &[f64]) -> Option<T> {
    if xs.len() == ps.len() {
        let cumsum = ps.iter()
            .scan(0.0, |acc, x| {
                *acc += *x;
                Some(*acc)
            })
            .collect::<Vec<f64>>();
        let threshold: f64 = random();
        for (i, &c) in cumsum.iter().enumerate() {
            if c <= threshold {
                return Some(xs[i].clone());
            }
        }
    }
    None
}

/// Computes logsumexp.
pub fn logsumexp(lps: &[f64]) -> f64 {
    if lps.is_empty() {
        return NEG_INFINITY;
    }
    let mut a = NEG_INFINITY;
    for &lp in lps {
        if lp > a {
            a = lp;
        }
    }
    let x = lps.iter().map(|lp| (lp - a).exp()).sum::<f64>().ln();
    a + x
}
