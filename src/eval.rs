use super::{Expression, DSL};

pub enum ReducedExpression<'a, V> {
    Value(V),
    Primitive(&'a str),
    Application(Box<ReducedExpression<'a, V>>, Box<ReducedExpression<'a, V>>),
    Abstraction(Box<ReducedExpression<'a, V>>),
    Index(usize),
}
impl<'a, V> ReducedExpression<'a, V> {
    pub fn eval<F>(&self, evaluator: &F, inps: &Vec<V>, out: &V) -> bool
    where
        F: Fn(&str, &Vec<V>) -> V,
    {
        false
    }
    pub fn new(dsl: &'a DSL, expr: &Expression) -> Self {
        Self::from_expr(dsl, &dsl.strip_invented(expr))
    }
    /// expr must be stripped of invented expressions.
    fn from_expr(dsl: &'a DSL, expr: &Expression) -> Self {
        match expr {
            &Expression::Primitive(num) => ReducedExpression::Primitive(&dsl.primitives[num].0),
            &Expression::Application(ref f, ref x) => ReducedExpression::Application(
                Box::new(Self::from_expr(dsl, f)),
                Box::new(Self::from_expr(dsl, x)),
            ),
            &Expression::Abstraction(ref body) => {
                ReducedExpression::Abstraction(Box::new(Self::from_expr(dsl, body)))
            }
            &Expression::Index(i) => ReducedExpression::Index(i),
            &Expression::Invented(num) => unreachable!(/* invented was stripped */),
        }
    }
}
