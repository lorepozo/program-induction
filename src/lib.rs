//! A library for program induction and learning representations.
//!
//! Implements Bayesian program learning and genetic programming.

extern crate itertools;
#[macro_use]
extern crate nom;
#[macro_use]
extern crate polytype;
extern crate rand;
extern crate rayon;
extern crate workerpool;

pub mod domains;
mod ec;
mod gp;
pub mod lambda;
pub mod pcfg;
pub use ec::*;
pub use gp::*;

use std::f64;
use polytype::Type;

/// A task which is solved by an expression under some representation.
///
/// A task can be made from a simple evaluator and examples with
/// [`lambda::task_by_simple_evaluation`] or [`pcfg::task_by_simple_evaluation`].
/// Tasks which require more complex (and expensive) evaluation can be made with a
/// [`lambda::LispEvaluator`]
///
/// [`lambda::task_by_simple_evaluation`]: lambda/fn.task_by_simple_evaluation.html
/// [`pcfg::task_by_simple_evaluation`]: pcfg/fn.task_by_simple_evaluation.html
/// [`lambda::LispEvaluator`]: lambda/struct.LispEvaluator.html
pub struct Task<'a, R: Send + Sync + Sized, X: Clone + Send, O: Sync> {
    /// Evaluate an expression by getting its log-likelihood.
    pub oracle: Box<Fn(&R, &X) -> f64 + Send + Sync + 'a>,
    /// Some program induction methods can take advantage of observations. This may often
    /// practically be the unit type `()`.
    pub observation: O,
    /// An expression that is considered valid for the `oracle` is one of this Type.
    pub tp: Type,
}
