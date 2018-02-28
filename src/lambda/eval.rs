//! Only works with systems that don't have first order functions. (i.e. evaluation only
//! happens by calling primitives.)

use std::collections::VecDeque;
use std::fmt::Debug;
use std::rc::Rc;
use polytype::Type;
use super::{Expression, Language};

#[derive(Clone, Debug, PartialEq)]
pub enum ReducedExpression<'a, V: Clone + PartialEq + Debug> {
    Value(V),
    Primitive(&'a str, &'a Type),
    Application(Vec<ReducedExpression<'a, V>>),
    /// store depth (never zero) for nested abstractions.
    Abstraction(usize, Box<ReducedExpression<'a, V>>),
    Index(usize),
}
impl<'a, V> ReducedExpression<'a, V>
where
    V: Clone + PartialEq + Debug,
{
    pub fn new(dsl: &'a Language, expr: &Expression) -> Self {
        Self::from_expr(dsl, &dsl.strip_invented(expr))
    }
    pub fn eval_inps<F>(&self, evaluator: &F, inps: &[V]) -> Option<V>
    where
        F: Fn(&str, &[V]) -> V,
    {
        let expr = self.clone().with_args(inps);
        let env = &Rc::new(VecDeque::new());
        let mut evaluated = expr.eval(evaluator, env);
        loop {
            let next = evaluated.eval(evaluator, env);
            if next == evaluated {
                break;
            } else {
                evaluated = next
            }
        }
        match evaluated {
            ReducedExpression::Value(o) => Some(o),
            _ => None,
        }
    }
    fn eval<F>(
        &self,
        evaluator: &F,
        env: &Rc<VecDeque<ReducedExpression<'a, V>>>,
    ) -> ReducedExpression<'a, V>
    where
        F: Fn(&str, &[V]) -> V,
    {
        match *self {
            ReducedExpression::Application(ref xs) => {
                let f = &xs[0];
                let mut xs: Vec<_> = xs[1..].iter().map(|x| x.eval(evaluator, env)).collect();
                match *f {
                    ReducedExpression::Primitive(name, tp) => {
                        // when applying a primitive, check if all arity-many args are concrete
                        // values and evaluate if possible.
                        if let Type::Arrow(ref arrow) = *tp {
                            let arity = arrow.args().len();
                            if arity <= xs.len() && xs.iter().take(arity).all(|x| match *x {
                                ReducedExpression::Value(_) => true,
                                _ => false,
                            }) {
                                let mut args = xs;
                                let mut xs = args.split_off(arity);
                                let args: Vec<V> = args.into_iter()
                                    .map(|x| match x {
                                        ReducedExpression::Value(v) => v,
                                        _ => unreachable!(),
                                    })
                                    .collect();
                                let v = ReducedExpression::Value(evaluator(name, &args));
                                if xs.is_empty() {
                                    v
                                } else {
                                    xs.insert(0, v);
                                    ReducedExpression::Application(xs)
                                }
                            } else {
                                xs.insert(0, f.eval(evaluator, env));
                                ReducedExpression::Application(xs)
                            }
                        } else {
                            panic!("tried to apply a primitive that wasn't a function")
                        }
                    }
                    ReducedExpression::Abstraction(ref depth, ref body) => {
                        // when applying an abstraction, try to beta-reduce
                        if xs.is_empty() {
                            ReducedExpression::Abstraction(*depth, body.clone())
                        } else {
                            let mut env = (**env).clone();
                            let mut depth: usize = *depth;
                            xs.reverse();
                            while !xs.is_empty() && depth > 0 {
                                let binding = xs.pop().unwrap();
                                env.push_front(binding);
                                depth -= 1;
                            }
                            xs.reverse();
                            let v = body.eval(evaluator, &Rc::new(env));
                            if depth > 0 {
                                ReducedExpression::Abstraction(depth, Box::new(v))
                            } else if xs.is_empty() {
                                v
                            } else if let ReducedExpression::Application(mut v) = v {
                                v.extend(xs);
                                ReducedExpression::Application(v)
                            } else {
                                xs.insert(0, v);
                                ReducedExpression::Application(xs)
                            }
                        }
                    }
                    _ => {
                        xs.insert(0, f.eval(evaluator, env));
                        ReducedExpression::Application(xs)
                    }
                }
            }
            ReducedExpression::Primitive(name, tp) => {
                if let Type::Arrow(_) = *tp {
                    ReducedExpression::Primitive(name, tp)
                } else {
                    ReducedExpression::Value(evaluator(name, &[]))
                }
            }
            ReducedExpression::Index(i) => match env.get(i) {
                Some(x) => x.clone(),
                None => ReducedExpression::Index(i),
            },
            _ => self.clone(),
        }
    }
    fn with_args(self, inps: &[V]) -> Self {
        let mut inps: Vec<_> = inps.iter()
            .map(|v| ReducedExpression::Value(v.clone()))
            .collect();
        match self {
            ReducedExpression::Application(mut xs) => {
                xs.extend(inps);
                ReducedExpression::Application(xs)
            }
            _ => {
                inps.insert(0, self);
                ReducedExpression::Application(inps)
            }
        }
    }
    fn from_expr(dsl: &'a Language, expr: &Expression) -> Self {
        match *expr {
            Expression::Primitive(num) => {
                ReducedExpression::Primitive(&dsl.primitives[num].0, &dsl.primitives[num].1)
            }
            Expression::Application(ref f, ref x) => {
                let mut v = vec![Self::from_expr(dsl, x)];
                let mut f: &Expression = f;
                loop {
                    if let Expression::Application(ref inner_f, ref x) = *f {
                        v.push(Self::from_expr(dsl, x));
                        f = inner_f;
                    } else {
                        v.push(Self::from_expr(dsl, f));
                        break;
                    }
                }
                v.reverse();
                ReducedExpression::Application(v)
            }
            Expression::Abstraction(ref body) => {
                let mut body: &Expression = body;
                let mut depth = 1;
                while let Expression::Abstraction(ref inner_body) = *body {
                    depth += 1;
                    body = inner_body;
                }
                ReducedExpression::Abstraction(depth, Box::new(Self::from_expr(dsl, body)))
            }
            Expression::Index(i) => ReducedExpression::Index(i),
            Expression::Invented(_) => unreachable!(/* invented was stripped */),
        }
    }
}
