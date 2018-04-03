//! Evaluation happens by calling primitives provided by an evaluator.
use std::collections::VecDeque;
use std::sync::Arc;
use polytype::TypeSchema;

use lambda::{Expression, Language};

pub fn eval<V, E>(dsl: &Language, expr: &Expression, evaluator: &Arc<E>, inps: &[V]) -> Option<V>
where
    V: Clone + PartialEq + Send + Sync,
    E: Evaluator<Space = V>,
{
    ReducedExpression::new(dsl, expr).eval_inps(evaluator, inps)
}

/// A specification for evaluating lambda calculus expressions in a domain.
///
/// In many simple domains, using [`SimpleEvaluator::of`] where you need an `Evaluator` should
/// suffice. A custom implementation should be done if the domain has **first-class functions**, so
/// that an [`Abstraction`] may by "lifted" into the domain's value space. What follows is
/// primarily an explanation of evaluation for value spaces with first-class functions.
///
/// The associated type [`Space`] of an `Evaluator` is the value space of the domain. It is
/// typically an enum with variants for different types of values. `Evaluator`s serve as a bridge
/// from [`Type`]s and [`Expression`]s manipulated by this library to concrete values in Rust.  It
/// is, therefore, best practice for `Space` to be an enum with as many variants as there are
/// constructed types, plus one variant for functions. It is _guaranteed_ that evaluation will
/// correspond to whatever the constraints of your types are. In other words, if `"plus"` takes two
/// numbers and `"concat"` takes two strings according to the [`Language`], then they will _never_
/// be called with arguments that aren't two numbers or two strings respectively.
///
/// When an `Abstraction` is encountered and passed into a function, a [`lift`] is attempted to
/// bring the abstraction into the domain's `Space`. For example, if `"map"` takes a function from
/// an int to an int, and it gets passed `λx.(+ 1 x)`, then an evaluator for that abstraction is
/// wrapped into a [`LiftedFunction`] and passed into `lift` to bring the function into the value
/// space, before finally being used on [`evaluate`] with the `"map"` primitive. An example below
/// features `"map"` calling a lifted function.
///
/// # Examples
///
/// An evaluator for a domain that doesn't have first-class functions:
///
/// ```
/// # #[macro_use] extern crate polytype;
/// # extern crate programinduction;
/// use programinduction::lambda::{Language, SimpleEvaluator};
///
/// # fn early_let() {
/// let dsl = Language::uniform(vec![
///     ("0", ptp!(int)),
///     ("1", ptp!(int)),
///     ("+", ptp!(@arrow[tp!(int), tp!(int), tp!(int)])),
/// ]);
/// # }
///
/// fn evaluate(primitive: &str, inps: &[i32]) -> i32 {
///     match primitive {
///         "0" => 0,
///         "1" => 1,
///         "+" => inps[0] + inps[1],
///         _ => unreachable!(),
///     }
/// }
///
/// # fn main() {
/// // Evaluator<Space = i32>
/// let eval = SimpleEvaluator::of(evaluate);
/// # }
/// ```
///
/// A slightly more complicated domain, but still without first-class functions:
///
/// ```
/// # #[macro_use] extern crate polytype;
/// # extern crate programinduction;
/// use programinduction::lambda::{Language, SimpleEvaluator};
///
/// # fn early_let() {
/// let dsl = Language::uniform(vec![
///     ("0", ptp!(int)),
///     ("1", ptp!(int)),
///     ("+", ptp!(@arrow[tp!(int), tp!(int), tp!(int)])),
///     ("eq", ptp!(@arrow[tp!(int), tp!(int), tp!(bool)])),
///     ("not", ptp!(@arrow[tp!(bool), tp!(bool)])),
/// ]);
/// # }
///
/// #[derive(Clone, PartialEq)]
/// enum ArithSpace {
///     Bool(bool),
///     Num(i32),
/// }
/// use ArithSpace::*;
///
/// fn evaluate(primitive: &str, inps: &[ArithSpace]) -> ArithSpace {
///     match primitive {
///         "0" => Num(0),
///         "1" => Num(1),
///         "+" => match (&inps[0], &inps[1]) {
///             (&Num(x), &Num(y)) => Num(x + y),
///             _ => unreachable!(),
///         }
///         "eq" => match (&inps[0], &inps[1]) {
///             (&Num(x), &Num(y)) => Bool(x == y),
///             _ => unreachable!(),
///         }
///         "not" => match inps[0] {
///             Bool(b) => Bool(!b),
///             _ => unreachable!(),
///         }
///         _ => unreachable!(),
///     }
/// }
///
/// # fn main() {
/// // Evaluator<Space = ArithSpace>
/// let eval = SimpleEvaluator::of(evaluate);
/// # }
/// ```
///
/// For a domain with first-class functions, things get more complicated:
///
/// ```
/// # #[macro_use] extern crate polytype;
/// # extern crate programinduction;
/// use programinduction::lambda::{Evaluator, Language, LiftedFunction};
///
/// # fn early_let() {
/// let dsl = Language::uniform(vec![
///     ("0", ptp!(int)),
///     ("1", ptp!(int)),
///     ("+", ptp!(@arrow[tp!(int), tp!(int), tp!(int)])),
///     ("singleton", ptp!(@arrow[tp!(int), tp!(list(tp!(int)))])),
///     ("chain", ptp!(0; @arrow[
///         tp!(list(tp!(0))),
///         tp!(list(tp!(0))),
///         tp!(list(tp!(0))),
///     ])),
///     ("map", ptp!(0, 1; @arrow[
///         tp!(@arrow[tp!(0), tp!(1)]),
///         tp!(list(tp!(0))),
///         tp!(list(tp!(0))),
///     ])),
/// ]);
/// // note: the only constructable lists in this dsl are of ints.
/// # }
///
/// #[derive(Clone)]
/// enum ListSpace {
///     Number(i32),
///     List(Vec<i32>),
///     Func(LiftedFunction<ListSpace, ListsEvaluator>),
/// }
/// use ListSpace::*;
/// impl PartialEq for ListSpace {
///     fn eq(&self, other: &Self) -> bool {
///         match (self, other) {
///             (&Number(x), &Number(y)) => x == y,
///             (&List(ref xs), &List(ref ys)) => xs == ys,
///             _ => false,
///         }
///     }
/// }
///
/// #[derive(Copy, Clone)]
/// struct ListsEvaluator;
/// impl Evaluator for ListsEvaluator {
///     type Space = ListSpace;
///     fn evaluate(&self, primitive: &str, inps: &[Self::Space]) -> Self::Space {
///         match primitive {
///             "0" => Number(0),
///             "1" => Number(1),
///             "+" => match (&inps[0], &inps[1]) {
///                 (&Number(x), &Number(y)) => Number(x + y),
///                 _ => unreachable!(),
///             },
///             "singleton" => match inps[0] {
///                 Number(x) => List(vec![x]),
///                 _ => unreachable!(),
///             },
///             "chain" => match (&inps[0], &inps[1]) {
///                 (&List(ref xs), &List(ref ys)) => {
///                     List(xs.into_iter().chain(ys).cloned().collect())
///                 },
///                 _ => unreachable!(),
///             },
///             "map" => match (&inps[0], &inps[1]) {
///                 (&Func(ref f), &List(ref xs)) => List(
///                     xs.into_iter()
///                         .cloned()
///                         .map(|x| match f.eval(&[Number(x)]) {
///                             Number(y) => y,
///                             _ => panic!("map given invalid function"),
///                         })
///                         .collect(),
///                 ),
///                 _ => unreachable!(),
///             },
///             _ => unreachable!(),
///         }
///     }
///     fn lift(&self, f: LiftedFunction<Self::Space, Self>) -> Result<Self::Space, ()> {
///         Ok(Func(f))
///     }
/// }
///
/// # fn main() {
/// // Evaluator<Space = ListSpace>
/// let eval = ListsEvaluator;
/// # }
/// ```
///
/// [`SimpleEvaluator::of`]: struct.SimpleEvaluator.html#method.of
/// [`Abstraction`]: enum.Expression.html#variant.Abstraction
/// [`Language`]: struct.Language.html
/// [`LiftedFunction`]: struct.LiftedFunction.html
/// [`Type`]: https://docs.rs/polytype
/// [`Expression`]: enum.Expression.html
/// [`Space`]: #associatedtype.Space
/// [`lift`]: #method.lift
/// [`evaluate`]: #tymethod.evaluate
pub trait Evaluator: Sized + Sync {
    /// The value space of a domain. The inputs of every primitive and the result of every
    /// evaluation must be of this type.
    type Space: Clone + PartialEq + Send + Sync;
    fn evaluate(&self, primitive: &str, inps: &[Self::Space]) -> Self::Space;
    fn lift(&self, _f: LiftedFunction<Self::Space, Self>) -> Result<Self::Space, ()> {
        Err(())
    }
}
impl<V> Evaluator for fn(&str, &[V]) -> V
where
    V: Clone + PartialEq + Send + Sync,
{
    type Space = V;
    fn evaluate(&self, primitive: &str, inps: &[Self::Space]) -> Self::Space {
        self(primitive, inps)
    }
}

/// An [`Evaluator`] defined solely by a function.
///
/// Use [`of`] to create one. Incapable of dealing with first-class functions.
///
/// [`Evaluator`]: trait.Evaluator.html
/// [`of`]: struct.SimpleEvaluator.html#method.of
pub struct SimpleEvaluator<V, F>(F, ::std::marker::PhantomData<V>);
impl<V, F> SimpleEvaluator<V, F>
where
    V: Clone + PartialEq + Send + Sync,
    F: Fn(&str, &[V]) -> V,
{
    /// Create a `SimpleEvaluator` out of a function that takes a primitive name and a list of
    /// arguments.
    ///
    /// # Examples
    ///
    /// ```
    /// use programinduction::lambda::SimpleEvaluator;
    ///
    /// fn evaluate(primitive: &str, inp: &[bool]) -> bool {
    ///     match primitive {
    ///         "nand" => !(inp[0] & inp[1]),
    ///         _ => unreachable!(),
    ///     }
    /// }
    ///
    /// let eval = SimpleEvaluator::of(evaluate);
    /// ```
    pub fn of(f: F) -> Self {
        SimpleEvaluator(f, ::std::marker::PhantomData)
    }
}
impl<V, F> Evaluator for SimpleEvaluator<V, F>
where
    V: Clone + PartialEq + Send + Sync,
    F: Fn(&str, &[V]) -> V + Sync,
{
    type Space = V;
    fn evaluate(&self, primitive: &str, inps: &[Self::Space]) -> Self::Space {
        (self.0)(primitive, inps)
    }
}

/// A function object for evaluation in domains with first-class functions.
///
/// The [`eval`] method evaluates the function. See [`Evaluator`] for more on its usage.
///
/// [`eval`]: #method.eval
/// [`Evaluator`]: trait.Evaluator.html
pub struct LiftedFunction<V: Clone + PartialEq + Send + Sync, E: Evaluator<Space = V>>(
    Arc<ReducedExpression<V>>,
    Arc<E>,
    Arc<VecDeque<ReducedExpression<V>>>,
);
impl<V, E> LiftedFunction<V, E>
where
    E: Evaluator<Space = V>,
    V: Clone + PartialEq + Send + Sync,
{
    /// Evaluate the lifted function on some values. You should determine how many values can be
    /// passed in based on the types of the [`Language`] specification.
    ///
    /// [`Language`]: struct.Language.html
    pub fn eval(&self, xs: &[V]) -> V {
        self.0
            .eval_inps_with_env(&self.1, &self.2, xs)
            .expect("nested evaluation failed")
    }
}
impl<V, E> Clone for LiftedFunction<V, E>
where
    E: Evaluator<Space = V>,
    V: Clone + PartialEq + Send + Sync,
{
    fn clone(&self) -> Self {
        LiftedFunction(self.0.clone(), self.1.clone(), self.2.clone())
    }
}

use self::ReducedExpression::*;
#[derive(Clone, PartialEq)]
pub enum ReducedExpression<V: Clone + PartialEq + Send + Sync> {
    Value(V),
    Primitive(String, TypeSchema),
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
        evaluator: &Arc<E>,
        env: &Arc<VecDeque<ReducedExpression<V>>>,
        inps: &[V],
    ) -> Option<V>
    where
        E: Evaluator<Space = V>,
    {
        let expr = self.clone().with_args(inps);
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
            Value(o) => Some(o),
            _ => None,
        }
    }
    pub fn eval_inps<E>(&self, evaluator: &Arc<E>, inps: &[V]) -> Option<V>
    where
        E: Evaluator<Space = V>,
    {
        let env = Arc::new(VecDeque::new());
        self.eval_inps_with_env(evaluator, &env, inps)
    }
    fn eval<E>(
        &self,
        evaluator: &Arc<E>,
        env: &Arc<VecDeque<ReducedExpression<V>>>,
    ) -> ReducedExpression<V>
    where
        E: Evaluator<Space = V>,
    {
        match *self {
            Application(ref xs) => {
                let f = &xs[0];
                let mut xs: Vec<_> = xs[1..].iter().map(|x| x.eval(evaluator, env)).collect();
                match *f {
                    Primitive(ref name, ref tp) => {
                        // when applying a primitive, check if all arity-many args are concrete
                        // values, try lifting abstractions, and evaluate if possible.
                        let arity = arity(tp);
                        if arity == 0 {
                            panic!("tried to apply a primitive that wasn't a function")
                        } else if arity > xs.len() || !xs.iter().take(arity).any(|x| match *x {
                            Value(_) | Abstraction(_, _) => true, // evaluatable
                            _ => false,
                        }) {
                            xs.insert(0, f.eval(evaluator, env));
                            Application(xs)
                        } else {
                            let mut args = xs;
                            let mut xs = args.split_off(arity);
                            let args: Vec<V> = args.into_iter()
                                .map(|x| match x {
                                    Value(v) => v,
                                    Abstraction(_, _) => {
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
                            let v = Value(evaluator.evaluate(name, &args));
                            if xs.is_empty() {
                                v
                            } else {
                                xs.insert(0, v);
                                Application(xs)
                            }
                        }
                    }
                    Abstraction(ref depth, ref body) => {
                        // when applying an abstraction, try to beta-reduce
                        if xs.is_empty() {
                            Abstraction(*depth, body.clone())
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
                            let v = body.eval(evaluator, &Arc::new(env));
                            if depth > 0 {
                                Abstraction(depth, Box::new(v))
                            } else if xs.is_empty() {
                                v
                            } else if let Application(mut v) = v {
                                v.extend(xs);
                                Application(v)
                            } else {
                                xs.insert(0, v);
                                Application(xs)
                            }
                        }
                    }
                    _ => {
                        xs.insert(0, f.eval(evaluator, env));
                        Application(xs)
                    }
                }
            }
            Primitive(ref name, ref tp) => {
                if is_arrow(tp) {
                    Primitive(name.clone(), tp.clone())
                } else {
                    Value(evaluator.evaluate(name, &[]))
                }
            }
            Index(i) => match env.get(i) {
                Some(x) => x.clone(),
                None => Index(i),
            },
            _ => self.clone(),
        }
    }
    fn with_args(self, inps: &[V]) -> Self {
        let mut inps: Vec<_> = inps.iter().map(|v| Value(v.clone())).collect();
        match self {
            Application(mut xs) => {
                xs.extend(inps);
                Application(xs)
            }
            _ => {
                inps.insert(0, self);
                Application(inps)
            }
        }
    }
    fn from_expr(dsl: &Language, expr: &Expression) -> Self {
        match *expr {
            Expression::Primitive(num) => {
                Primitive(dsl.primitives[num].0.clone(), dsl.primitives[num].1.clone())
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
                Application(v)
            }
            Expression::Abstraction(ref body) => {
                let mut body: &Expression = body;
                let mut depth = 1;
                while let Expression::Abstraction(ref inner_body) = *body {
                    depth += 1;
                    body = inner_body;
                }
                Abstraction(depth, Box::new(Self::from_expr(dsl, body)))
            }
            Expression::Index(i) => Index(i),
            Expression::Invented(_) => unreachable!(), // invented was stripped
        }
    }
}

fn arity(mut tp: &TypeSchema) -> usize {
    let mut tp = loop {
        match *tp {
            TypeSchema::Monotype(ref t) => break t,
            TypeSchema::Polytype { ref body, .. } => tp = body,
        }
    };
    let mut count = 0;
    while let Some((_, ret)) = tp.as_arrow() {
        count += 1;
        tp = ret;
    }
    count
}

fn is_arrow(mut tp: &TypeSchema) -> bool {
    loop {
        match *tp {
            TypeSchema::Monotype(ref t) => break t.as_arrow().is_some(),
            TypeSchema::Polytype { ref body, .. } => tp = body,
        }
    }
}
