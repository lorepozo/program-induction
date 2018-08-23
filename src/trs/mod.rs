//! (representation) Polymorphically-typed term rewriting system.
//!
//! An evaluatable first-order [Term Rewriting System][0] (TRS) with a [Hindley-Milner type
//! system][1].
//!
//! [0]: https://wikipedia.org/wiki/Hindley–Milner_type_system
//!      "Wikipedia - Hindley-Milner Type System"
//! [1]: https://en.wikipedia.org/wiki/Rewriting#Term_rewriting_systems
//!      "Wikipedia - Term Rewriting Systems"
//!
//! # Example
//!
//! ```
//! # #[macro_use] extern crate polytype;
//! # extern crate programinduction;
//! # extern crate term_rewriting;
//! # use programinduction::trs::{TRS, Lexicon};
//! # use term_rewriting::{Signature, parse_rule};
//! # fn main() {
//! let mut sig = Signature::default();
//!
//! let mut ops = vec![];
//! sig.new_op(2, Some("PLUS".to_string()));
//! ops.push(ptp![@arrow[tp!(int), tp!(int), tp!(int)]]);
//! sig.new_op(1, Some("SUCC".to_string()));
//! ops.push(ptp![@arrow[tp!(int), tp!(int)]]);
//! sig.new_op(0, Some("ZERO".to_string()));
//! ops.push(ptp![int]);
//!
//! let rules = vec![
//!     parse_rule(&mut sig, "PLUS(x_ ZERO) = x_").expect("parsed rule"),
//!     parse_rule(&mut sig, "PLUS(x_ SUCC(y_)) = SUCC(PLUS(x_ y_))").expect("parsed rule"),
//! ];
//!
//! let vars = vec![
//!     ptp![int],
//!     ptp![int],
//!     ptp![int],
//! ];
//!
//! let lexicon = Lexicon::from_signature(sig, ops, vars, vec![], false);
//!
//! let trs = TRS::new(&lexicon, rules);
//! # }
//! ```

mod lexicon;
pub mod parser;
mod rewrite;
pub use self::lexicon::{GeneticParams, Lexicon};
pub use self::parser::{
    parse_context, parse_lexicon, parse_rulecontext, parse_templates, parse_trs,
};
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
    /// How much partial credit is given for incorrect answers; it should be a
    /// probability (i.e. in [0, 1]).
    pub p_partial: f64,
    /// The (non-log) probability of generating observations at arbitrary
    /// evaluation steps (i.e. not just normal forms). Typically 0.0.
    pub p_observe: f64,
    /// The number of evaluation steps you would like to explore in the trace.
    pub max_steps: usize,
    /// The largest term that will be considered for evaluation. `None` will
    /// evaluate all terms.
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

/// Construct a [`Task`] evaluating [`TRS`]s (constructed from a [`Lexicon`])
/// using rewriting of inputs to outputs.
///
/// Each [`term_rewriting::Rule`] in `data` must have a single RHS term. The
/// resulting [`Task`] checks whether each datum's LHS gets rewritten to its RHS
/// under a [`TRS`] within the constraints specified by the [`ModelParams`].
///
/// [`Lexicon`]: struct.Lexicon.html
/// [`ModelParams`]: struct.ModelParams.html
/// [`term_rewriting::Rule`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Rule.html
/// [`Task`]: ../struct.Task.html
/// [`TRS`]: struct.TRS.html
pub fn task_by_rewrite<'a, O: Sync>(
    data: &'a [Rule],
    params: ModelParams,
    lex: &Lexicon,
    ctx: &mut polytype::Context,
    observation: O,
) -> Result<Task<'a, Lexicon, TRS, O>, TypeError> {
    Ok(Task {
        oracle: Box::new(move |_s: &Lexicon, h: &TRS| -h.posterior(data, params)),
        tp: lex.infer_rules(data, ctx)?,
        observation,
    })
}
