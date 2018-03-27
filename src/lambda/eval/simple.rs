//! Evaluation happens by calling primitives provided by an evaluator.

use std::collections::VecDeque;
use std::sync::Arc;
use polytype::Type;

use lambda::{Evaluator, Expression, Language};
use super::LiftedFunction;

#[derive(Clone, PartialEq)]
pub enum ReducedExpression<V: Clone + PartialEq + Sync> {
    Value(V),
    Primitive(String, Type),
    Application(Vec<ReducedExpression<V>>),
    /// store depth (never zero) for nested abstractions.
    Abstraction(usize, Box<ReducedExpression<V>>),
    Index(usize),
}
impl<V> ReducedExpression<V>
where
    V: Clone + PartialEq + Send + Sync,
{
    pub fn new(dsl: &Language, expr: &Expression) -> Self {
        Self::from_expr(dsl, &dsl.strip_invented(expr))
    }
    pub fn eval_inps_with_env<E>(
        &self,
        evaluator: Arc<E>,
        env: Arc<VecDeque<ReducedExpression<V>>>,
        inps: &[V],
    ) -> Option<V>
    where
        E: Evaluator<Space = V>,
    {
        let expr = self.clone().with_args(inps);
        let mut evaluated = expr.eval(evaluator.clone(), env.clone());
        loop {
            let next = evaluated.eval(evaluator.clone(), env.clone());
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
    pub fn eval_inps<E>(&self, evaluator: Arc<E>, inps: &[V]) -> Option<V>
    where
        E: Evaluator<Space = V>,
    {
        let env = Arc::new(VecDeque::new());
        self.eval_inps_with_env(evaluator, env, inps)
    }
    fn eval<E>(
        &self,
        evaluator: Arc<E>,
        env: Arc<VecDeque<ReducedExpression<V>>>,
    ) -> ReducedExpression<V>
    where
        E: Evaluator<Space = V>,
    {
        match *self {
            ReducedExpression::Application(ref xs) => {
                let f = &xs[0];
                let mut xs: Vec<_> = xs[1..]
                    .iter()
                    .map(|x| x.eval(evaluator.clone(), env.clone()))
                    .collect();
                match *f {
                    ReducedExpression::Primitive(ref name, ref tp) => {
                        // when applying a primitive, check if all arity-many args are concrete
                        // values, try lifting abstractions, and evaluate if possible.
                        if let Type::Arrow(ref arrow) = *tp {
                            let arity = arrow.args().len();
                            if arity <= xs.len() && xs.iter().take(arity).all(|x| match *x {
                                ReducedExpression::Value(_) => true,
                                ReducedExpression::Abstraction(_, _) => true,
                                _ => false,
                            }) {
                                let mut args = xs;
                                let mut xs = args.split_off(arity);
                                let args: Vec<V> = args.into_iter()
                                    .map(|x| match x {
                                        ReducedExpression::Value(v) => v,
                                        ReducedExpression::Abstraction(_, _) => {
                                            let env = env.clone();
                                            evaluator
                                                .clone()
                                                .lift(LiftedFunction(
                                                    Arc::new(x),
                                                    evaluator.clone(),
                                                    env.clone(),
                                                ))
                                                .expect("evaluator could not lift an abstraction")
                                        }
                                        _ => unreachable!(),
                                    })
                                    .collect();
                                let v = ReducedExpression::Value(evaluator.evaluate(name, &args));
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
                            let mut env = (*env).clone();
                            let mut depth: usize = *depth;
                            xs.reverse();
                            while !xs.is_empty() && depth > 0 {
                                let binding = xs.pop().unwrap();
                                env.push_front(binding);
                                depth -= 1;
                            }
                            xs.reverse();
                            let v = body.eval(evaluator, Arc::new(env));
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
            ReducedExpression::Primitive(ref name, ref tp) => {
                if let Type::Arrow(_) = *tp {
                    ReducedExpression::Primitive(name.clone(), tp.clone())
                } else {
                    ReducedExpression::Value(evaluator.evaluate(&name, &[]))
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
    fn from_expr(dsl: &Language, expr: &Expression) -> Self {
        match *expr {
            Expression::Primitive(num) => ReducedExpression::Primitive(
                dsl.primitives[num].0.clone(),
                dsl.primitives[num].1.clone(),
            ),
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
