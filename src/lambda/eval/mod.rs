mod simple;

// lisp through racket
#[cfg(feature = "racket")]
mod lisp_racket;
#[cfg(feature = "racket")]
use self::lisp_racket as lisp;

// lisp through rust
#[cfg(not(feature = "racket"))]
mod lisp_rust;
#[cfg(not(feature = "racket"))]
mod interp;
#[cfg(not(feature = "racket"))]
mod sexp;
#[cfg(not(feature = "racket"))]
use self::lisp_rust as lisp;

pub use self::lisp::{LispError, LispEvaluator};

use self::simple::ReducedExpression;
use super::{Expression, Language};

pub fn simple_eval<V, F>(dsl: &Language, expr: &Expression, evaluator: &F, inps: &[V]) -> Option<V>
where
    F: Fn(&str, &[V]) -> V,
    V: Clone + PartialEq + ::std::fmt::Debug,
{
    ReducedExpression::new(dsl, expr).eval_inps(evaluator, inps)
}
