//! For internal use only.

use polytype::Type;
use super::{Expression, DSL};

/// do not manipulate a ReducedExpression manually.
pub enum ReducedExpression<'a, V> {
    Value(V),
    Primitive { name: &'a str, arity: usize },
    Application(Box<Vec<ReducedExpression<'a, V>>>),
    Abstraction(Box<ReducedExpression<'a, V>>),
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
            ReducedExpression::Value(ref o) => o == out,
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
                let arity = if let Type::Arrow(ref arrow) = dsl.primitives[num].1 {
                    arrow.args().len()
                } else {
                    0
                };
                ReducedExpression::Primitive {
                    name: &dsl.primitives[num].0,
                    arity,
                }
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
                ReducedExpression::Application(Box::new(v))
            }
            &Expression::Abstraction(ref body) => {
                ReducedExpression::Abstraction(Box::new(Self::from_expr(dsl, body)))
            }
            &Expression::Index(i) => ReducedExpression::Index(i),
            &Expression::Invented(_) => unreachable!(/* invented was stripped */),
        }
    }
}
