use crossbeam_channel;
use rand::{
    distributions::{Distribution, Uniform},
    thread_rng,
};
use std::f64;

/// We wrap crossbeam_channel's sender/receiver so that `Sender`s will not
/// block indefinitely when a corresponding `Receiver` is dropped.
/// (In other words, so that `send` detects a closed channel.)
pub fn bounded<T>(cap: usize) -> (Sender<T>, Receiver<T>) {
    let (s, r) = crossbeam_channel::bounded(cap);
    let (ds, dr) = crossbeam_channel::bounded(0);
    let s = Sender { inner: s, drop: dr };
    let r = Receiver { inner: r, drop: ds };
    (s, r)
}
#[derive(Clone, Debug)]
pub struct Sender<T> {
    inner: crossbeam_channel::Sender<T>,
    drop: crossbeam_channel::Receiver<T>,
}
impl<T> Sender<T> {
    pub fn send(&self, msg: T) -> Result<(), T> {
        select! {
            recv(self.drop) => Err(msg),
            send(self.inner, msg) => Ok(()),
        }
    }
}
#[derive(Clone, Debug)]
pub struct Receiver<T> {
    inner: crossbeam_channel::Receiver<T>,
    /// when dropped, this channel gets closed
    #[allow(dead_code)]
    drop: crossbeam_channel::Sender<T>,
}
impl<T> Receiver<T> {
    pub fn try_recv(&self) -> Option<T> {
        self.inner.try_recv()
    }
}
impl<T> Iterator for Receiver<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.recv()
    }
}

#[inline(always)]
pub fn logsumexp(lps: &[f64]) -> f64 {
    let largest = lps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let x = lps.iter().map(|lp| (lp - largest).exp()).sum::<f64>().ln();
    largest + x
}

pub fn weighted_permutation<T: Clone>(xs: &[T], ws: &[f64]) -> Vec<T> {
    let mut ws = ws.to_vec();
    let mut idxs: Vec<_> = (0..(ws.len())).collect();
    let mut permutation = vec![];
    while !ws.is_empty() {
        let jidxs: Vec<_> = idxs.iter().cloned().enumerate().collect();
        let &(jdx, idx): &(usize, usize) = weighted_sample(&jidxs, &ws);
        permutation.push(xs[idx].clone());
        idxs.remove(jdx);
        ws.remove(jdx);
    }
    permutation
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
