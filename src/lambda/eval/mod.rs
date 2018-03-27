use std::collections::VecDeque;
use std::sync::Arc;

mod simple;

use self::simple::ReducedExpression;
use super::{Expression, Language};

pub fn eval<E, V>(dsl: &Language, expr: &Expression, evaluator: Arc<E>, inps: &[V]) -> Option<V>
where
    E: Evaluator<Space = V>,
    V: Clone + PartialEq + Send + Sync,
{
    ReducedExpression::new(dsl, expr).eval_inps(evaluator, inps)
}

#[derive(Clone)]
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
    pub fn eval(&self, xs: &[V]) -> V {
        self.0
            .eval_inps_with_env(self.1.clone(), self.2.clone(), xs)
            .expect("nested evaluation failed")
    }
}

/// A specification for evaluating lambda calculus expressions in a domain.
///
/// In many simple domains, using [`EvaluatorFunc::of`] where you need an `Evaluator` should
/// suffice. A custom implementation should be done if the domain has **first-class functions**, so
/// that an [`Abstraction`] may by "lifted" into the domain's value space. What follows is an
/// explanation of evaluation for value spaces with first-class functions.
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
/// When an [`Abstraction`] is encountered and attempted to be passed into a function, it attempts
/// a [`lift`] using the method defined by an `Evaluator`. For example, if `"map"` takes a function
/// from an into to an int, and it gets passed `λx.(+ 1 x)`, then an evaluator for that abstraction
/// is wrapped into a closure and passed into [`lift`] before being used on [`evaluate`].
///
/// # Examples
///
/// An evaluator for a domain that doesn't have first-class functions:
///
/// ```
/// use programinduction::lambda::{Evaluator, EvaluatorFunc};
///
/// let dsl = Language::uniform(vec![
///     ("0", tp!(int)),
///     ("1", tp!(int)),
///     ("+", arrow![tp!(int), tp!(int), tp!(int)]),
/// ]);
///
/// fn simple_evaluator(primitive: &str, inps: &[i32]) -> i32 {
///     match primitive {
///         "0" => 0,
///         "1" => 1,
///         "+" => inps[0] + inps[1],
///         _ => unreachable!(),
///     }
/// }
///
/// let eval: Evaluator = EvaluatorFunc::of(simple_evaluator);
/// ```
///
/// A slightly more complicated domain, but still without first-class functions:
///
/// ```
/// use programinduction::lambda::{Evaluator, EvaluatorFunc};
///
/// let dsl = Language::uniform(vec![
///     ("0", tp!(int)),
///     ("1", tp!(int)),
///     ("+", arrow![tp!(int), tp!(int), tp!(int)]),
///     ("eq", arrow![tp!(int), tp!(int), tp!(bool)]),
///     ("not", arrow![tp!(bool), tp!(bool)]),
/// ]);
///
/// #[derive(Clone, PartialEq)]
/// enum ArithSpace {
///     Bool(bool)
///     Num(i32),
/// }
/// fn simple_evaluator(primitive: &str, inps: &[ArithSpace]) -> ArithSpace {
///     match primitive {
///         "0" => Num(0),
///         "1" => Num(1),
///         "+" => match (&inps[0], &inps[1]) {
///             (&Num(x), &Num(y)) => Num(x + y)
///             _ => unreachable!(),
///         }
///         "eq" => match (&inps[0], &inps[1]) {
///             (&Num(x), &Num(y)) => Bool(x == y),
///             _ => unreachable!(),
///         }
///         "not" => match inps[0] {
///             Bool(b) => Bool(!b)
///         }
///         _ => unreachable!(),
///     }
/// }
///
/// let eval: Evaluator = EvaluatorFunc::of(simple_evaluator);
/// ```
///
/// For a domain with first-class functions, things get more complicated:
///
/// ```
/// use programinduction::{Evaluator, Language};
///
/// let dsl = Language::uniform(vec![
///     ("0", tp!(int)),
///     ("1", tp!(int)),
///     ("+", arrow![tp!(int), tp!(int), tp!(int)]),
///     ("singleton", arrow![tp!(int), tp!(intlist)]),
///     ("chain", arrow![tp!(intlist), tp!(intlist), tp!(intlist)]),
///     ("map", arrow![arrow![tp!(int), tp!(int)], tp!(intlist), tp!(intlist)]),
/// ]);
///
/// #[derive(Clone)]
/// enum ValueSpace {
///     Number(i32),
///     List(Vec<i32>),
///     Func(Box<Fn(&[ValueSpace]) -> ValueSpace>),
/// }
/// use ValueSpace::*;
/// impl PartialEq for ValueSpace {
///     fn eq(&self, other: &Self) -> bool {
///         match (self, other) {
///             (&Number(x), &Number(y)) => x == y,
///             (&List(ref xs), &List(ref ys)) => xs == ys,
///             _ => false,
///         }
///     }
/// }
/// impl Evaluator for ValueSpace {
///     type Space = Self;
///     fn evaluate(&self, name: &str, inps: &[Self::Space]) -> Self::Space {
///         match name {
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
///                 (&Func(f), &List(ref xs)) => List(
///                     xs.into_iter()
///                         .cloned()
///                         .map(|x| match f(&[Number(x)]) {
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
///     fn lift<F: Fn(&[Self::Space]) -> Self::Space>(f: F) -> Result<Self::Space, ()> {
///         Ok(Func(Box::new(f)))
///     }
/// }
///
/// let eval: Evaluator = ValueSpace::Number(42);
/// ```
///
/// [`EvaluatorFunc::of`]: struct.EvaluatorFunc.html#method.of
/// [`Abstraction`]: enum.Expression.html#variant.Abstraction
/// [`Language`]: struct.Language.html
/// [`Type`]: https://docs.rs/polytype
/// [`Expression`]: enum.Expression.html
/// [`Space`]: #associatedtype.Space
/// [`lift`]: #method.lift
/// [`evaluate`]: #tymethod.evaluate
pub trait Evaluator: Sized + Sync {
    /// The value space of a domain. The inputs of every primitive and the result of every
    /// evaluation must be of this type.
    type Space: Clone + PartialEq + Send + Sync;
    fn evaluate(&self, name: &str, inps: &[Self::Space]) -> Self::Space;
    fn lift(&self, _: LiftedFunction<Self::Space, Self>) -> Result<Self::Space, ()> {
        Err(())
    }
}
impl<V: Clone + PartialEq + Send + Sync> Evaluator for fn(&str, &[V]) -> V {
    type Space = V;
    fn evaluate(&self, name: &str, inps: &[Self::Space]) -> Self::Space {
        self(name, inps)
    }
}

/// An [`Evaluator`] defined solely by a function.
///
/// Use [`of`] to create one.
///
/// [`Evaluator`]: trait.Evaluator.html
/// [`of`]: struct.EvaluatorFunc.html#method.of
pub struct EvaluatorFunc<F, V>(F, ::std::marker::PhantomData<V>);
impl<F, V> EvaluatorFunc<F, V>
where
    F: Fn(&str, &[V]) -> V,
    V: PartialEq + Clone + Send + Sync,
{
    /// Create an `EvaluatorFunc` out of a function that takes a primitive name and a list of
    /// arguments.
    ///
    /// # Examples
    ///
    /// ```
    /// use programinduction::lambda::{Evaluator, EvaluatorFunc};
    ///
    /// fn simple_evaluator(primitive: &str, inp: &[bool]) -> bool {
    ///     match primitive {
    ///         "nand" => !(inp[0] & inp[1]),
    ///         _ => unreachable!(),
    ///     }
    /// }
    ///
    /// let eval: Evaluator = EvaluatorFunc::of(simple_evaluator);
    /// # let _  = eval;
    /// ```
    pub fn of(f: F) -> Self {
        EvaluatorFunc(f, ::std::marker::PhantomData)
    }
}
impl<F, V> Evaluator for EvaluatorFunc<F, V>
where
    F: Fn(&str, &[V]) -> V + Sync,
    V: PartialEq + Clone + Send + Sync,
{
    type Space = V;
    fn evaluate(&self, name: &str, inps: &[Self::Space]) -> Self::Space {
        (self.0)(name, inps)
    }
}
