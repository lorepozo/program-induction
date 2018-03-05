//! A library for program induction and learning representations.

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
pub mod lambda;
pub mod pcfg;
pub use ec::*;

use std::f64;
use std::fmt;
use polytype::Type;

/// The representation of a task which is solved by an [`Expression`] under some
/// [`Representation`].
///
/// A task can be made from an evaluator and examples with [`lambda::task_by_example`].
///
/// [`Representation`]: trait.Representation.html
/// [`Expression`]: trait.Representation.html#associatedtype.Expression
/// [`lambda::task_by_example`]: lambda/fn.task_by_example.html
pub struct Task<'a, R: Representation, O: Sync> {
    /// Evaluate an expression by getting its log-likelihood.
    pub oracle: Box<Fn(&R, &R::Expression) -> f64 + Send + Sync + 'a>,
    /// Some program induction methods can take advantage of observations. This may often
    /// practically be the unit type `()`.
    pub observation: O,
    /// An expression that is considered valid for the `oracle` is one of this Type.
    pub tp: Type,
}

/// A representation gives a space of expressions. It will, in most cases, be a probability
/// distribution over expressions (e.g. PCFG).
pub trait Representation: Send + Sync + Sized {
    /// An Expression is a sentence in the representation. Tasks are solved by Expressions.
    type Expression: Clone + Send;

    fn infer(&self, expr: &Self::Expression) -> Result<Type, InferenceError>;

    fn display(&self, expr: &Self::Expression) -> String {
        let _ = expr;
        "<Expression>".to_owned()
    }
}

#[derive(Debug, Clone)]
pub enum InferenceError {
    BadExpression(String),
    Unify(polytype::UnificationError),
}
impl From<polytype::UnificationError> for InferenceError {
    fn from(err: polytype::UnificationError) -> Self {
        InferenceError::Unify(err)
    }
}
impl fmt::Display for InferenceError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            InferenceError::BadExpression(ref msg) => write!(f, "invalid expression: '{}'", msg),
            InferenceError::Unify(ref err) => write!(f, "could not unify to infer type: {}", err),
        }
    }
}
impl ::std::error::Error for InferenceError {
    fn description(&self) -> &str {
        "could not infer type"
    }
}
