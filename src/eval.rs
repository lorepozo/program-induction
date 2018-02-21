use super::Expression;

/// an evaluator takes a primitive name and
pub fn eval<F, V>(evaluator: &F, expr: &Expression, inps: &Vec<V>, out: &V) -> bool
where
    F: Fn(&str, &Vec<V>) -> V,
{
    false
}
