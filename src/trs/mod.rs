//! (representation) Polymorphically-typed term rewriting system.
//!
//! An evaluatable first-order [Term Rewriting System][0] (TRS) with a [Hindley-Milner type
//! system][1].
//!
//! [0]: https://wikipedia.org/wiki/Hindley–Milner_type_system
//!      "Wikipedia - Hindley-Milner Type System"
//! [1]: https://en.wikipedia.org/wiki/Rewriting#Term_rewriting_systems
//!      "Wikipedia - Term Rewriting Systems"

mod lexicon;
mod rewrite;
pub use self::lexicon::{GeneticParams, Lexicon};
pub use self::rewrite::TRS;
use Task;

use polytype;
use std::fmt;
use term_rewriting::{Rule, TRSError};

#[derive(Debug, Clone)]
/// The error type for type inference.
pub enum TypeError {
    Unification(polytype::UnificationError),
    OpNotFound,
    VarNotFound,
}
impl From<polytype::UnificationError> for TypeError {
    fn from(e: polytype::UnificationError) -> TypeError {
        TypeError::Unification(e)
    }
}
impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            TypeError::Unification(ref e) => write!(f, "unification error: {}", e),
            TypeError::OpNotFound => write!(f, "operator not found"),
            TypeError::VarNotFound => write!(f, "variable not found"),
        }
    }
}
impl ::std::error::Error for TypeError {
    fn description(&self) -> &'static str {
        "type error"
    }
}

#[derive(Debug, Clone)]
/// The error type for sampling operations.
pub enum SampleError {
    TypeError(TypeError),
    TRSError(TRSError),
    DepthExceeded(usize, usize),
    OptionsExhausted,
    Subterm,
}
impl From<TypeError> for SampleError {
    fn from(e: TypeError) -> SampleError {
        SampleError::TypeError(e)
    }
}
impl From<TRSError> for SampleError {
    fn from(e: TRSError) -> SampleError {
        SampleError::TRSError(e)
    }
}
impl From<polytype::UnificationError> for SampleError {
    fn from(e: polytype::UnificationError) -> SampleError {
        SampleError::TypeError(TypeError::Unification(e))
    }
}
impl<T> From<Vec<T>> for SampleError {
    fn from(_e: Vec<T>) -> SampleError {
        SampleError::OptionsExhausted
    }
}
impl fmt::Display for SampleError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            SampleError::TypeError(ref e) => write!(f, "type error: {}", e),
            SampleError::TRSError(ref e) => write!(f, "TRS error: {}", e),
            SampleError::DepthExceeded(depth, max_depth) => {
                write!(f, "depth {} exceeded maximum of {}", depth, max_depth)
            }
            SampleError::OptionsExhausted => write!(f, "failed to sample (options exhausted)"),
            SampleError::Subterm => write!(f, "cannot sample subterm"),
        }
    }
}
impl ::std::error::Error for SampleError {
    fn description(&self) -> &'static str {
        "sample error"
    }
}

/// Parameters for a TRS-based probabilistic model.
#[derive(Debug, Copy, Clone)]
pub struct ModelParams {
    /// How much partial credit is given for incorrect answers
    pub p_partial: f64,
    /// The (non-log) probability of generating observations at arbitrary evaluation steps (i.e. not just normal forms). Typically 0.0.
    pub p_observe: f64,
    /// The number of evaluation steps you would like to explore in the trace.
    pub max_steps: usize,
    /// The largest term that will be considered for evaluation. `None` will evaluate all terms.
    pub max_size: Option<usize>,
}
impl Default for ModelParams {
    fn default() -> ModelParams {
        ModelParams {
            p_partial: 0.0,
            p_observe: 0.0,
            max_steps: 50,
            max_size: Some(500),
        }
    }
}

/// Construct a [`Task`] suitable for [genetic programming] with a [`Lexicon`].
///
/// [`Task`]: ../struct.Task.html
/// [genetic programming]: struct.Lexicon.html#impl-GP
/// [`Lexicon`]: struct.Lexicon.html
pub fn make_task_from_data(
    data: &[Rule],
    tp: polytype::TypeSchema,
    params: ModelParams,
) -> Task<Lexicon, TRS, ()> {
    Task {
        oracle: Box::new(move |_s: &Lexicon, h: &TRS| -h.posterior(data, params)),
        // TODO: compute type schema from the data
        tp,
        observation: (),
    }
}
