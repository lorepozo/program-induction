//! Evaluation happens by calling primitives provided by an evaluator.
use polytype::TypeSchema;
use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;

use lambda::{Expression, Language};

pub fn eval<V, E>(
    dsl: &Language,
    expr: &Expression,
    evaluator: &Arc<E>,
    inps: &[V],
) -> Result<V, E::Error>
where
    V: Clone + PartialEq + Send + Sync,
    E: Evaluator<Space = V>,
{
    ReducedExpression::new(dsl, expr).eval_inps(evaluator, inps)
}

pub fn lazy_eval<V, E>(
    dsl: &Language,
    expr: &Expression,
    evaluator: &Arc<E>,
    inps: &[V],
) -> Result<V, E::Error>
where
    V: Clone + PartialEq + Send + Sync,
    E: LazyEvaluator<Space = V>,
{
    ReducedExpression::new(dsl, expr).lazy_eval_inps(evaluator, inps)
}

/// A specification for evaluating lambda calculus expressions in a domain.
///
/// Use [`Language::eval`] to invoke.
///
/// In many simple domains, using [`SimpleEvaluator::of`] where you need an `Evaluator` should
/// suffice. A custom implementation should be done if the domain has **first-class functions**, so
/// that an [`Abstraction`] may by "lifted" into the domain's value space.
///
/// The associated type [`Space`] of an `Evaluator` is the value space of the domain. `Evaluator`s
/// serve as a bridge from [`Type`]s and [`Expression`]s manipulated by this library to concrete
/// values in Rust.  It is, therefore, best practice for `Space` to be an enum with as many
/// variants as there are constructed types, plus one variant for functions. It is _guaranteed_
/// that evaluation will correspond to whatever the constraints of your types are. In other words,
/// if `"plus"` takes two numbers and `"concat"` takes two strings according to the [`Language`],
/// then they will _never_ be called with arguments that aren't two numbers or two strings
/// respectively. To help illustrate this, note that none of the this library's example evaluators
/// can panic.
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
/// fn evaluate(primitive: &str, inps: &[i32]) -> Result<i32, ()> {
///     match primitive {
///         "0" => Ok(0),
///         "1" => Ok(1),
///         "+" => Ok(inps[0] + inps[1]),
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
/// fn evaluate(primitive: &str, inps: &[ArithSpace]) -> Result<ArithSpace, ()> {
///     match primitive {
///         "0" => Ok(Num(0)),
///         "1" => Ok(Num(1)),
///         "+" => match (&inps[0], &inps[1]) {
///             (&Num(x), &Num(y)) => Ok(Num(x + y)),
///             _ => unreachable!(),
///         }
///         "eq" => match (&inps[0], &inps[1]) {
///             (&Num(x), &Num(y)) => Ok(Bool(x == y)),
///             _ => unreachable!(),
///         }
///         "not" => match inps[0] {
///             Bool(b) => Ok(Bool(!b)),
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
/// struct ListError(&'static str);
///
/// #[derive(Clone)]
/// enum ListSpace {
///     Num(i32),
///     NumList(Vec<i32>),
///     Func(LiftedFunction<ListSpace, ListsEvaluator>),
/// }
/// use ListSpace::*;
/// impl PartialEq for ListSpace {
///     fn eq(&self, other: &Self) -> bool {
///         match (self, other) {
///             (&Num(x), &Num(y)) => x == y,
///             (&NumList(ref xs), &NumList(ref ys)) => xs == ys,
///             _ => false,
///         }
///     }
/// }
///
/// #[derive(Copy, Clone)]
/// struct ListsEvaluator;
/// impl Evaluator for ListsEvaluator {
///     type Space = ListSpace;
///     type Error = ListError;
///     fn evaluate(
///         &self,
///         primitive: &str,
///         inps: &[Self::Space],
///     ) -> Result<Self::Space, Self::Error> {
///         match primitive {
///             "0" => Ok(Num(0)),
///             "1" => Ok(Num(1)),
///             "+" => match (&inps[0], &inps[1]) {
///                 (&Num(x), &Num(y)) => Ok(Num(x + y)),
///                 _ => unreachable!(),
///             },
///             "singleton" => match inps[0] {
///                 Num(x) => Ok(NumList(vec![x])),
///                 _ => unreachable!(),
///             },
///             "chain" => match (&inps[0], &inps[1]) {
///                 (&NumList(ref xs), &NumList(ref ys)) => {
///                     Ok(NumList(xs.into_iter().chain(ys).cloned().collect()))
///                 },
///                 _ => unreachable!(),
///             },
///             "map" => match (&inps[0], &inps[1]) {
///                 (&Func(ref f), &NumList(ref xs)) => {
///                     Ok(NumList(xs.into_iter()
///                         .map(|x| {
///                             f.eval(&[Num(x.clone())])
///                                 .and_then(|v| match v {
///                                     Num(y) => Ok(y),
///                                     _ => Err(ListError("map given invalid function")),
///                                 })
///                         })
///                         .collect::<Result<_, _>>()?))
///                 }
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
/// // Evaluator<Space = ListSpace, Error = ListError>
/// let eval = ListsEvaluator;
/// # }
/// ```
///
/// [`SimpleEvaluator::of`]: struct.SimpleEvaluator.html#method.of
/// [`Abstraction`]: enum.Expression.html#variant.Abstraction
/// [`Language`]: struct.Language.html
/// [`Language::eval`]: struct.Language.html#method.eval
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
    /// If evaluation should fail, this would hold an appropriate error.
    type Error: Clone + Sync;
    fn evaluate(&self, primitive: &str, inps: &[Self::Space]) -> Result<Self::Space, Self::Error>;
    fn lift(&self, _f: LiftedFunction<Self::Space, Self>) -> Result<Self::Space, ()> {
        Err(())
    }
}

/// Like [`Evaluator`], but you get to decide whether certain arguments should be evaluated.
///
/// Use [`Language::lazy_eval`] to invoke.
///
/// This should look very much like an `Evaluator`, but rather than an [`evaluate`] method which
/// takes inputs in its [`Space`], `LazyEvaluator` has a [`lazy_evaluate`] method which takes
/// inputs that are all _thunks_ — or argumentless functions — which give a value in [`Space`] upon
/// evaluation via `thunk.eval(&[])?`.
///
/// The example below covers a domain where lazy evaluation is necessary. Consider a function that
/// takes a list which should return -1 if the list is empty, otherwise returning the first item in
/// the list. This function may look like `(λ (if (empty? $0) -1 (car $0)))`. Without lazy
/// evaluation, both branches (`-1` and `(car $0)`) are evaluated before the `if` primitive is
/// evaluated — this could fail when called on an empty list becuase `(car $0)` be evaluated. With
/// lazy evaluation, the `if` primitives is responsible for calling the evaluation of its
/// arguments.
///
/// # Examples
///
/// ```
/// # #[macro_use] extern crate polytype;
/// # extern crate programinduction;
/// use programinduction::lambda::{LazyEvaluator, Language, LiftedLazyFunction};
///
/// # fn early_let() {
/// let dsl = Language::uniform(vec![
///     ("if", ptp!(0; @arrow[tp!(bool), tp!(0), tp!(0), tp!(0)])),
///     ("empty?", ptp!(0; @arrow[tp!(list(tp!(0))), tp!(bool)])),
///     ("car", ptp!(0; @arrow[tp!(list(tp!(0))), tp!(0)])),
///     ("-1", ptp!(int)),
/// ]);
/// // note: we will only have lists of ints in the example.
/// # }
///
/// #[derive(Clone, Debug, PartialEq)]
/// struct ListError(&'static str);
///
/// #[derive(Clone, Debug)]
/// enum ListSpace {
///     Bool(bool),
///     Num(i32),
///     List(Vec<i32>),
/// }
/// use ListSpace::*;
/// impl PartialEq for ListSpace {
///     fn eq(&self, other: &Self) -> bool {
///         match (self, other) {
///             (&Bool(x), &Bool(y)) => x == y,
///             (&Num(x), &Num(y)) => x == y,
///             (&List(ref xs), &List(ref ys)) => xs == ys,
///             _ => false,
///         }
///     }
/// }
///
/// #[derive(Copy, Clone)]
/// struct ListsEvaluator;
/// impl LazyEvaluator for ListsEvaluator {
///     type Space = ListSpace;
///     type Error = ListError;
///     fn lazy_evaluate(
///         &self,
///         primitive: &str,
///         inps: &[LiftedLazyFunction<Self::Space, Self>],
///     ) -> Result<Self::Space, Self::Error> {
///         match primitive {
///             "if" => match inps[0].eval(&[])? {
///                 Bool(true) => inps[1].eval(&[]),
///                 Bool(false) => inps[2].eval(&[]),
///                 _ => unreachable!(),
///             },
///             "empty?" => match inps[0].eval(&[])? {
///                 List(xs) => Ok(Bool(xs.is_empty())),
///                 _ => unreachable!(),
///             },
///             "car" => match inps[0].eval(&[])? {
///                 List(xs) => if !xs.is_empty() {
///                     Ok(Num(xs[0].clone()))
///                 } else {
///                     Err(ListError("cannot get car of empty list"))
///                 },
///                 _ => unreachable!(),
///             },
///             "-1" => Ok(Num(-1)),
///             _ => unreachable!(),
///         }
///     }
/// }
///
/// # fn main() {
/// # let dsl = Language::uniform(vec![
/// #     ("if", ptp!(0; @arrow[tp!(bool), tp!(0), tp!(0), tp!(0)])),
/// #     ("empty?", ptp!(0; @arrow[tp!(list(tp!(0))), tp!(bool)])),
/// #     ("car", ptp!(0; @arrow[tp!(list(tp!(0))), tp!(0)])),
/// #     ("-1", ptp!(int)),
/// # ]);
/// #
/// // LazyEvaluator<Space = ListSpace, Error = ListError>
/// let eval = ListsEvaluator;
///
/// let expr = dsl.parse("(λ (if (empty? $0) -1 (car $0)))").unwrap();
/// let inps = vec![List(Vec::new())];
/// // without lazy evaluation, this would be an Err because of evaluation of (car $0)
/// let evaluated = dsl.lazy_eval(&expr, eval, &inps);
/// assert_eq!(evaluated, Ok(Num(-1)));
/// # }
/// ```
///
/// [`Evaluator`]: trait.Evaluator.html
/// [`Space`]: #associatedtype.Space
/// [`evaluate`]: trait.Evaluator.html#tymethod.evaluate
/// [`lazy_evaluate`]: #tymethod.lazy_evaluate
/// [`Language::lazy_eval`]: struct.Language.html#method.lazy_eval
pub trait LazyEvaluator: Sized + Sync {
    /// The value space of a domain. The inputs of every primitive and the result of every
    /// evaluation must be of this type.
    type Space: Clone + PartialEq + Send + Sync;
    /// If evaluation should fail, this would hold an appropriate error.
    type Error: Clone + Sync;
    /// All inputs are thunks — argumentless functions — to get the underlying value.
    fn lazy_evaluate(
        &self,
        primitive: &str,
        inps: &[LiftedLazyFunction<Self::Space, Self>],
    ) -> Result<Self::Space, Self::Error>;
    fn lift(&self, _f: LiftedLazyFunction<Self::Space, Self>) -> Result<Self::Space, ()> {
        Err(())
    }
}

/// An [`Evaluator`] defined solely by a function.
///
/// Use [`of`] to create one. Incapable of dealing with first-class functions.
///
/// [`Evaluator`]: trait.Evaluator.html
/// [`of`]: #method.of
pub struct SimpleEvaluator<V, R, F>(F, ::std::marker::PhantomData<(R, V)>);
impl<V, R, F> SimpleEvaluator<V, R, F>
where
    V: Clone + PartialEq + Send + Sync,
    R: Clone + Sync,
    F: Fn(&str, &[V]) -> Result<V, R>,
{
    /// Create a `SimpleEvaluator` out of a function that takes a primitive name and a list of
    /// arguments.
    ///
    /// # Examples
    ///
    /// ```
    /// use programinduction::lambda::SimpleEvaluator;
    ///
    /// fn evaluate(primitive: &str, inp: &[bool]) -> Result<bool, ()> {
    ///     match primitive {
    ///         "nand" => Ok(!(inp[0] & inp[1])),
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
impl<V, R, F> Evaluator for SimpleEvaluator<V, R, F>
where
    V: Clone + PartialEq + Send + Sync,
    R: Clone + Sync,
    F: Fn(&str, &[V]) -> Result<V, R> + Sync,
{
    type Space = V;
    type Error = R;
    fn evaluate(&self, primitive: &str, inps: &[Self::Space]) -> Result<Self::Space, Self::Error> {
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
    pub fn eval(&self, xs: &[V]) -> Result<V, E::Error> {
        self.0.eval_inps_with_env(&self.1, &self.2, xs)
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
impl<V, E> PartialEq for LiftedFunction<V, E>
where
    E: Evaluator<Space = V>,
    V: Clone + PartialEq + Send + Sync,
{
    fn eq(&self, other: &LiftedFunction<V, E>) -> bool {
        self.0 == other.0 && self.2 == other.2
    }
}
impl<V, E> Eq for LiftedFunction<V, E>
where
    E: Evaluator<Space = V>,
    V: Clone + PartialEq + Send + Sync,
{
}

/// Like [`LiftedFunction`], but for lazy evaluation with a [`LazyEvaluator`].
///
/// The [`eval`] method evaluates the function. See [`LazyEvaluator`] for more on its usage.
///
/// [`LiftedFunction`]: struct.LiftedFunction.html
/// [`LazyEvaluator`]: trait.LazyEvaluator.html
/// [`eval`]: #method.eval
pub struct LiftedLazyFunction<V: Clone + PartialEq + Send + Sync, E: LazyEvaluator<Space = V>>(
    Arc<ReducedExpression<V>>,
    Arc<E>,
    Arc<VecDeque<ReducedExpression<V>>>,
);
impl<V, E> LiftedLazyFunction<V, E>
where
    E: LazyEvaluator<Space = V>,
    V: Clone + PartialEq + Send + Sync,
{
    /// Evaluate the lifted function on some values. You should determine how many values can be
    /// passed in based on the types of the [`Language`] specification.
    ///
    /// [`Language`]: struct.Language.html
    pub fn eval(&self, xs: &[V]) -> Result<V, E::Error> {
        self.0.lazy_eval_inps_with_env(&self.1, &self.2, xs)
    }
}
impl<V, E> Clone for LiftedLazyFunction<V, E>
where
    E: LazyEvaluator<Space = V>,
    V: Clone + PartialEq + Send + Sync,
{
    fn clone(&self) -> Self {
        LiftedLazyFunction(self.0.clone(), self.1.clone(), self.2.clone())
    }
}
impl<V, E> PartialEq for LiftedLazyFunction<V, E>
where
    E: LazyEvaluator<Space = V>,
    V: Clone + PartialEq + Send + Sync,
{
    fn eq(&self, other: &LiftedLazyFunction<V, E>) -> bool {
        self.0 == other.0 && self.2 == other.2
    }
}
impl<V, E> Eq for LiftedLazyFunction<V, E>
where
    E: LazyEvaluator<Space = V>,
    V: Clone + PartialEq + Send + Sync,
{
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
impl<V> fmt::Debug for ReducedExpression<V>
where
    V: Clone + PartialEq + Send + Sync,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Value(_) => write!(f, "Value"),
            Primitive(ref s, _) => write!(f, "Primitive({})", s),
            Application(ref xs) => write!(f, "Application({:?})", xs),
            Abstraction(depth, ref body) => write!(f, "Abstraction({}, {:?})", depth, body),
            Index(n) => write!(f, "Index({})", n),
        }
    }
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
    ) -> Result<V, E::Error>
    where
        E: Evaluator<Space = V>,
    {
        let expr = self.clone().with_args(inps);
        let mut evaluated = expr.eval(evaluator, env)?;
        loop {
            let next = evaluated.eval(evaluator, env)?;
            if next == evaluated {
                break;
            }
            evaluated = next;
        }
        match evaluated {
            Value(o) => Ok(o),
            e => panic!("tried to evaluate an irreducible expression: {:?}", e),
        }
    }
    pub fn lazy_eval_inps_with_env<E>(
        &self,
        evaluator: &Arc<E>,
        env: &Arc<VecDeque<ReducedExpression<V>>>,
        inps: &[V],
    ) -> Result<V, E::Error>
    where
        E: LazyEvaluator<Space = V>,
    {
        let expr = self.clone().with_args(inps);
        let mut evaluated = expr.lazy_eval(evaluator, env)?;
        loop {
            let next = evaluated.lazy_eval(evaluator, env)?;
            if next == evaluated {
                break;
            }
            evaluated = next;
        }
        match evaluated {
            Value(o) => Ok(o),
            e => panic!("tried to evaluate an irreducible expression {:?}", e),
        }
    }
    pub fn eval_inps<E>(&self, evaluator: &Arc<E>, inps: &[V]) -> Result<V, E::Error>
    where
        E: Evaluator<Space = V>,
    {
        let env = Arc::new(VecDeque::new());
        self.eval_inps_with_env(evaluator, &env, inps)
    }
    pub fn lazy_eval_inps<E>(&self, evaluator: &Arc<E>, inps: &[V]) -> Result<V, E::Error>
    where
        E: LazyEvaluator<Space = V>,
    {
        let env = Arc::new(VecDeque::new());
        self.lazy_eval_inps_with_env(evaluator, &env, inps)
    }
    fn eval<E>(
        &self,
        evaluator: &Arc<E>,
        env: &Arc<VecDeque<ReducedExpression<V>>>,
    ) -> Result<ReducedExpression<V>, E::Error>
    where
        E: Evaluator<Space = V>,
    {
        match *self {
            Application(ref xs) => {
                let f = &xs[0];
                let mut xs: Vec<_> = xs[1..]
                    .iter()
                    .map(|x| x.eval(evaluator, env))
                    .collect::<Result<_, _>>()?;
                match *f {
                    Primitive(ref name, ref tp) => {
                        // when applying a primitive, check if all arity-many args are concrete
                        // values, try lifting abstractions, and evaluate if possible.
                        let arity = arity(tp);
                        if arity == 0 {
                            panic!(
                                "tried to apply a primitive that wasn't a function: {}",
                                name
                            )
                        } else if xs.len() < arity || !xs.iter().take(arity).all(|x| match *x {
                            Value(_) | Abstraction(_, _) => true, // evaluatable
                            _ => false,
                        }) {
                            // not enough args or not all evaluatable.
                            xs.insert(0, f.eval(evaluator, env)?);
                            Ok(Application(xs))
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
                            let v = Value(evaluator.evaluate(name, &args)?);
                            if xs.is_empty() {
                                Ok(v)
                            } else {
                                xs.insert(0, v);
                                Ok(Application(xs))
                            }
                        }
                    }
                    Abstraction(ref depth, ref body) => {
                        // when applying an abstraction, try to beta-reduce
                        if xs.is_empty() {
                            Ok(Abstraction(*depth, body.clone()))
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
                            let v = body.eval(evaluator, &Arc::new(env))?;
                            if depth > 0 {
                                Ok(Abstraction(depth, Box::new(v)))
                            } else if xs.is_empty() {
                                Ok(v)
                            } else if let Application(mut v) = v {
                                v.extend(xs);
                                Ok(Application(v))
                            } else {
                                xs.insert(0, v);
                                Ok(Application(xs))
                            }
                        }
                    }
                    _ => {
                        xs.insert(0, f.eval(evaluator, env)?);
                        Ok(Application(xs))
                    }
                }
            }
            Primitive(ref name, ref tp) => {
                if is_arrow(tp) {
                    Ok(Primitive(name.clone(), tp.clone()))
                } else {
                    Ok(Value(evaluator.evaluate(name, &[])?))
                }
            }
            Index(i) => match env.get(i) {
                Some(x) => Ok(x.clone()),
                None => Ok(Index(i)),
            },
            _ => Ok(self.clone()),
        }
    }
    fn lazy_eval<E>(
        &self,
        evaluator: &Arc<E>,
        env: &Arc<VecDeque<ReducedExpression<V>>>,
    ) -> Result<ReducedExpression<V>, E::Error>
    where
        E: LazyEvaluator<Space = V>,
    {
        match *self {
            Application(ref exprs) => {
                let f = &exprs[0];
                let xs = &exprs[1..];
                match *f {
                    Primitive(ref name, ref tp) => {
                        // when applying a primitive, check for arity-many args
                        let arity = arity(tp);
                        if arity == 0 {
                            panic!(
                                "tried to apply a primitive that wasn't a function: {}",
                                name
                            )
                        } else if xs.len() < arity {
                            // not enough args
                            Ok(Application(exprs.clone()))
                        } else {
                            let mut args: Vec<_> = xs.into_iter().cloned().collect();
                            let mut xs = args.split_off(arity);
                            let args: Vec<_> = args.into_iter()
                                .map(|x| {
                                    if let Abstraction(_, _) = x {
                                        Value(
                                            evaluator
                                                .clone()
                                                .lift(LiftedLazyFunction(
                                                    Arc::new(x),
                                                    evaluator.clone(),
                                                    env.clone(),
                                                ))
                                                .expect("evaluator could not lift an abstraction"),
                                        )
                                    } else {
                                        x
                                    }
                                })
                                .map(|x| {
                                    LiftedLazyFunction(Arc::new(x), evaluator.clone(), env.clone())
                                })
                                .collect();
                            let v = Value(evaluator.lazy_evaluate(name, &args)?);
                            if xs.is_empty() {
                                Ok(v)
                            } else {
                                xs.insert(0, v);
                                Ok(Application(xs))
                            }
                        }
                    }
                    Abstraction(ref depth, ref body) => {
                        // when applying an abstraction, try to beta-reduce
                        if xs.is_empty() {
                            Ok(Abstraction(*depth, body.clone()))
                        } else {
                            let mut env = (**env).clone();
                            let mut depth: usize = *depth;
                            let mut xs: Vec<_> = xs.into_iter().rev().cloned().collect();
                            while !xs.is_empty() && depth > 0 {
                                let binding = xs.pop().unwrap();
                                env.push_front(binding);
                                depth -= 1;
                            }
                            xs.reverse();
                            let env = Arc::new(env);
                            let mut body = (*body).clone();
                            body.substitute_indices(&env, 0);
                            let v = body.lazy_eval(evaluator, &env)?;
                            if depth > 0 {
                                Ok(Abstraction(depth, Box::new(v)))
                            } else if xs.is_empty() {
                                Ok(v)
                            } else if let Application(mut v) = v {
                                v.extend(xs);
                                Ok(Application(v))
                            } else {
                                Ok(Application(exprs.clone()))
                            }
                        }
                    }
                    _ => Ok(Application(exprs.clone())),
                }
            }
            Primitive(ref name, ref tp) => {
                if is_arrow(tp) {
                    Ok(Primitive(name.clone(), tp.clone()))
                } else {
                    Ok(Value(evaluator.lazy_evaluate(name, &[])?))
                }
            }
            Index(i) => match env.get(i) {
                Some(x) => Ok(x.clone()),
                None => Ok(Index(i)),
            },
            _ => Ok(self.clone()),
        }
    }
    fn with_args(self, inps: &[V]) -> Self {
        if inps.is_empty() {
            self
        } else {
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
    }
    fn from_expr(dsl: &Language, expr: &Expression) -> Self {
        match *expr {
            Expression::Primitive(num) => {
                Primitive(dsl.primitives[num].0.clone(), dsl.primitives[num].1.clone())
            }
            Expression::Application(ref f, ref x) => {
                let mut v = vec![Self::from_expr(dsl, x)];
                let mut f: &Expression = f;
                while let Expression::Application(ref inner_f, ref x) = *f {
                    v.push(Self::from_expr(dsl, x));
                    f = inner_f;
                }
                v.push(Self::from_expr(dsl, f));
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
    fn substitute_indices(&mut self, env: &Arc<VecDeque<ReducedExpression<V>>>, offset: usize) {
        match *self {
            Application(ref mut xs) => for x in xs {
                x.substitute_indices(env, offset)
            },
            Abstraction(depth, ref mut body) => body.substitute_indices(env, offset + depth),
            Index(i) if i >= offset => if let Some(x) = env.get(i - offset) {
                *self = x.clone()
            },
            _ => (),
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
