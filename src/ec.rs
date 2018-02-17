use polytype::Type;

use std::collections::HashMap;

use expression::{Expression, DSL};

pub struct Task<'a, T> {
    oracle: Box<'a + Fn(Expression) -> f64>,
    observation: T,
    tp: Type,
}
impl<'a, T> Task<'a, T> {
    pub fn new<F: 'a + Fn(Expression) -> f64>(oracle: F, observation: T, tp: Type) -> Self {
        Self {
            oracle: Box::new(oracle),
            observation,
            tp,
        }
    }
}

pub fn ec<S, O, RS, R, E, C>(
    prior: &(DSL, S),
    tasks: &Vec<Task<O>>,
    recognizer: Option<R>,
    explore: E,
    compress: C,
) -> (DSL, S)
where
    R: FnOnce(&(DSL, S), &Vec<Task<O>>) -> RS,
    E: FnOnce(&(DSL, S), &Vec<Task<O>>, Option<RS>) -> HashMap<usize, Vec<Expression>>,
    C: FnOnce(&(DSL, S), &Vec<Task<O>>, HashMap<usize, Vec<Expression>>) -> (DSL, S),
{
    let recognized = recognizer.map(|r| r(prior, tasks));
    let frontiers = explore(prior, tasks, recognized);
    compress(prior, tasks, frontiers)
}
