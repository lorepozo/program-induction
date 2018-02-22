//! A library for program induction and learning grammars.
//!
//! Good places to look are [`DSL`] and [`ec`].
//!
//! [`DSL`]: struct.DSL.html
//! [`ec`]: ec/index.html

#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate polytype;

pub mod domains;
mod ec;
pub mod lambda;
pub use ec::EC;

use std::f64;
use std::fmt;
use polytype::Type;

/// The representation of a task which is solved by an [`Expression`] under some [`DSL`].
///
/// A task can be made from an evaluator and examples with [`from_examples`].
///
/// [`DSL`]: struct.DSL.html
/// [`Expression`]: enum.Expression.html
/// [`from_examples`]: #method.from_examples
pub struct Task<'a, R: Representation, O> {
    /// evaluate an expression by getting its log-likelihood.
    pub oracle: Box<Fn(&R, &R::Expression) -> f64 + 'a>,
    pub observation: O,
    pub tp: Type,
}

/// A representation gives a space of expressions. It will, in most cases, be a probability
/// distribution over expressions (e.g. PCFG).
pub trait Representation: Sized {
    /// An Expression is a sentence in the representation. Tasks are solved by Expressions.
    type Expression: Clone;

    fn infer(&self, expr: &Self::Expression) -> Result<Type, InferenceError>;
}

#[derive(Debug, Clone)]
pub enum InferenceError {
    BadPrimitive(usize),
    BadInvented(usize),
    Unify(polytype::UnificationError),
}
impl From<polytype::UnificationError> for InferenceError {
    fn from(err: polytype::UnificationError) -> Self {
        InferenceError::Unify(err)
    }
}
impl fmt::Display for InferenceError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            &InferenceError::BadPrimitive(i) => write!(f, "invalid primitive: '{}'", i),
            &InferenceError::BadInvented(i) => write!(f, "invalid invented: '{}'", i),
            &InferenceError::Unify(ref err) => write!(f, "could not unify to infer type: {}", err),
        }
    }
}
impl ::std::error::Error for InferenceError {
    fn description(&self) -> &str {
        "could not infer type"
    }
}
