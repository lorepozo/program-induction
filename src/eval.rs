use super::{Expression, DSL};

pub fn check<V, F>(dsl: &DSL, expr: &Expression, evaluator: &F, inps: &Vec<V>, out: &V) -> bool
where
    F: Fn(&str, &Vec<V>) -> V,
{
    // TODO: call lisp or something
    false
}
