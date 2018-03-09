mod interp;
mod lisp;
mod sexp;
mod simple;
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
