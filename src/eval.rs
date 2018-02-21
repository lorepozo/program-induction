//! For internal use only.

use polytype::Type;
use super::{Expression, DSL};

/// do not manipulate a ReducedExpression manually.
pub enum ReducedExpression<'a, V> {
    Value(V, Type),
    Primitive(&'a str, &'a Type),
    Application(Box<Vec<ReducedExpression<'a, V>>>, Type),
    Abstraction(Box<ReducedExpression<'a, V>>, Type),
    Index(usize),
}
impl<'a, V> ReducedExpression<'a, V>
where
    V: PartialEq,
{
    pub fn new(dsl: &'a DSL, expr: &Expression) -> Self {
        Self::from_expr(dsl, &dsl.strip_invented(expr))
    }
    pub fn check<F>(&self, evaluator: &F, inps: &Vec<V>, out: &V) -> bool
    where
        F: Fn(&str, &Vec<V>) -> V,
    {
        match self.eval(evaluator, inps) {
            ReducedExpression::Value(ref o, _) => o == out,
            _ => false,
        }
    }
    fn eval<F>(&self, evaluator: &F, inps: &Vec<V>) -> ReducedExpression<V>
    where
        F: Fn(&str, &Vec<V>) -> V,
    {
        ReducedExpression::Index(0) // TODO
    }
    /// expr must be stripped of invented expressions.
    fn from_expr(dsl: &'a DSL, expr: &Expression) -> Self {
        match expr {
            &Expression::Primitive(num) => {
                ReducedExpression::Primitive(&dsl.primitives[num].0, &dsl.primitives[num].1)
            }
            &Expression::Application(ref f, ref x) => {
                let mut v = vec![Self::from_expr(dsl, x)];
                let mut f: &Expression = f;
                loop {
                    if let &Expression::Application(ref inner_f, ref x) = f {
                        v.push(Self::from_expr(dsl, x));
                        f = &inner_f;
                    } else {
                        v.push(Self::from_expr(dsl, f));
                        break;
                    }
                }
                v.reverse();
                ReducedExpression::Application(Box::new(v), dsl.infer(expr).unwrap())
            }
            &Expression::Abstraction(ref body) => ReducedExpression::Abstraction(
                Box::new(Self::from_expr(dsl, body)),
                dsl.infer(expr).unwrap(),
            ),
            &Expression::Index(i) => ReducedExpression::Index(i),
            &Expression::Invented(_) => unreachable!(/* invented was stripped */),
        }
    }
}

// TODO:
pub trait HigherOrderValueSpace {
    /// Reduce is only called if the first arg is an arrow which takes a type that unifies with
    /// type of the second arg.
    fn reduce(Self, Self) -> Self;
}

impl<'a, V> ReducedExpression<'a, V>
where
    V: PartialEq + HigherOrderValueSpace,
{
    // replace from_expr with something more complicated
}
