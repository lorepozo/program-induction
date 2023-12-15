//! (representation) Polymorphically-typed lambda calculus.
//!
//! # Examples
//!
//! ```
//! use polytype::{ptp, tp};
//! use programinduction::{Task, lambda::{task_by_evaluation, Language, SimpleEvaluator}};
//!
//! fn evaluate(name: &str, inps: &[i32]) -> Result<i32, ()> {
//!     match name {
//!         "0" => Ok(0),
//!         "1" => Ok(1),
//!         "+" => Ok(inps[0] + inps[1]),
//!         _ => unreachable!(),
//!     }
//! }
//!
//! let dsl = Language::uniform(vec![
//!     ("0", ptp!(int)),
//!     ("1", ptp!(int)),
//!     ("+", ptp!(@arrow[tp!(int), tp!(int), tp!(int)])),
//! ]);
//!
//! // task: sum 1 with two numbers
//! let tp = ptp!(@arrow[tp!(int), tp!(int), tp!(int)]);
//! let examples = vec![(vec![2, 5], 8), (vec![1, 2], 4)];
//! let task = task_by_evaluation(SimpleEvaluator::from(evaluate), tp, &examples);
//!
//! // solution:
//! let expr = dsl.parse("(λ (+ (+ 1 $0)))").unwrap();
//! assert!(task.oracle(&dsl, &expr).is_finite())
//! ```

mod compression;
mod enumerator;
mod eval;
mod parser;
pub use self::compression::{induce, CompressionParams, RescoredFrontier};
pub use self::eval::{
    Evaluator, LazyEvaluator, LiftedFunction, LiftedLazyFunction, SimpleEvaluator,
};
pub use self::parser::ParseError;

use crossbeam_channel::bounded;
use polytype::{Context, Type, TypeScheme, UnificationError};
use rayon::spawn;
use std::collections::{HashMap, VecDeque};
use std::error::Error;
use std::ops::Index;
use std::rc::Rc;
use std::sync::Arc;

use crate::{ECFrontier, Task, EC};

const BOUND_VAR_COST: f64 = 0.1;
const FREE_VAR_COST: f64 = 0.01;

/// (representation) A Language is a registry for primitive and invented expressions in a
/// polymorphically-typed lambda calculus with corresponding production log-probabilities.
#[derive(Debug, Clone)]
pub struct Language {
    pub primitives: Vec<(String, TypeScheme, f64)>,
    pub invented: Vec<(Expression, TypeScheme, f64)>,
    pub variable_logprob: f64,
    /// Symmetry breaking prevents certain productions from being made. Specifically, an item
    /// `(f, i, a)` means that enumeration will not yield an application of `f` where the `i`th
    /// argument is `a`. This vec must be kept sorted; use via [`add_symmetry_violation`] and
    /// [`violates_symmetry`].
    ///
    /// [`add_symmetry_violation`]: #method.add_symmetry_violation
    /// [`violates_symmetry`]: #method.violates_symmetry
    pub symmetry_violations: Vec<(usize, usize, usize)>,
}
impl Language {
    /// A uniform distribution over primitives and invented expressions, as well as the abstraction
    /// operation.
    pub fn uniform(primitives: Vec<(&str, TypeScheme)>) -> Self {
        let primitives = primitives
            .into_iter()
            .map(|(s, t)| (String::from(s), t, 0f64))
            .collect();
        Language {
            primitives,
            invented: vec![],
            variable_logprob: 0f64,
            symmetry_violations: Vec::new(),
        }
    }

    /// Infer the type of an [`Expression`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use polytype::{tp, ptp};
    /// # use programinduction::lambda::{Expression, Language};
    /// let mut dsl = Language::uniform(vec![
    ///     ("singleton", ptp!(0; @arrow[tp!(0), tp!(list(tp!(0)))])),
    ///     (">=", ptp!(@arrow[tp!(int), tp!(int), tp!(bool)])),
    ///     ("+", ptp!(@arrow[tp!(int), tp!(int), tp!(int)])),
    ///     ("0", ptp!(int)),
    ///     ("1", ptp!(int)),
    /// ]);
    /// dsl.invent(
    ///     // (+ 1)
    ///     Expression::Application(
    ///         Box::new(Expression::Primitive(2)),
    ///         Box::new(Expression::Primitive(4)),
    ///     ),
    ///     0f64,
    /// );
    /// let expr = dsl.parse("(singleton ((λ (>= $0 1)) (#(+ 1) 0)))")
    ///     .unwrap();
    /// assert_eq!(dsl.infer(&expr).unwrap(), ptp!(list(tp!(bool))));
    /// ```
    ///
    /// [`Expression`]: enum.Expression.html
    pub fn infer(&self, expr: &Expression) -> Result<TypeScheme, InferenceError> {
        let mut ctx = Context::default();
        let env = VecDeque::new();
        let mut indices = HashMap::new();
        expr.infer(self, &mut ctx, &env, &mut indices)
            .map(|t| t.generalize(&[]))
    }

    /// Enumerate expressions for a request type (including log-probabilities and appropriately
    /// instantiated `Type`s):
    ///
    /// # Examples
    ///
    /// The following example can be made more effective using the approach shown with
    /// [`add_symmetry_violation`].
    ///
    /// ```
    /// use polytype::{ptp, tp};
    /// use programinduction::lambda::{Expression, Language};
    ///
    /// let dsl = Language::uniform(vec![
    ///     ("0", ptp!(int)),
    ///     ("1", ptp!(int)),
    ///     ("+", ptp!(@arrow[tp!(int), tp!(int), tp!(int)])),
    /// ]);
    /// let exprs: Vec<Expression> = dsl.enumerate(ptp!(int))
    ///     .take(8)
    ///     .map(|(expr, _log_prior)| expr)
    ///     .collect();
    ///
    /// assert_eq!(
    ///     exprs,
    ///     vec![
    ///         Expression::Primitive(0),
    ///         Expression::Primitive(1),
    ///         dsl.parse("(+ 0 0)").unwrap(),
    ///         dsl.parse("(+ 0 1)").unwrap(),
    ///         dsl.parse("(+ 1 0)").unwrap(),
    ///         dsl.parse("(+ 1 1)").unwrap(),
    ///         dsl.parse("(+ 0 (+ 0 0))").unwrap(),
    ///         dsl.parse("(+ 0 (+ 0 1))").unwrap(),
    ///     ]
    /// );
    /// ```
    ///
    /// [`add_symmetry_violation`]: #method.add_symmetry_violation
    pub fn enumerate(&self, tp: TypeScheme) -> Box<dyn Iterator<Item = (Expression, f64)>> {
        let (tx, rx) = bounded(1);
        let dsl = self.clone();
        spawn(move || {
            let tx = tx.clone();
            let termination_condition = |expr, logprior| tx.send((expr, logprior)).is_err();
            enumerator::run(&dsl, tp, termination_condition)
        });
        Box::new(rx.into_iter())
    }

    /// Update production probabilities and induce new primitives, with the guarantee that any
    /// changes to the language yield net lower prior probability for expressions in the frontier.
    ///
    /// Primitives are induced using an approach similar to Cohn et. al. in the 2010 _JMLR_ paper
    /// [Inducing Tree-Substitution Grammars] and in Tim O'Donnell's [Fragment Grammars] detailed
    /// in his 2015 _MIT Press_ book, _Productivity and Reuse in Language: A Theory of Linguistic
    /// Computation and Storage_. However, instead of using Bayesian non-parametrics, we fully
    /// evaluate posteriors under each non-trivial fragment (because we already have a tractible
    /// space of expressions — the frontiers). We repeatedly select the best fragment and
    /// re-evaluate the posteriors until the DSL does not improve.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use programinduction::domains::circuits;
    /// use programinduction::{lambda, ECParams, EC};
    /// use rand::{rngs::SmallRng, SeedableRng};
    ///
    /// let dsl = circuits::dsl();
    /// let rng = &mut SmallRng::from_seed([1u8; 32]);
    /// let tasks = circuits::make_tasks(rng, 100);
    /// let ec_params = ECParams {
    ///     frontier_limit: 10,
    ///     search_limit_timeout: None,
    ///     search_limit_description_length: Some(11.0),
    /// };
    /// let params = lambda::CompressionParams::default();
    ///
    /// // this is equivalent to one iteration of EC:
    /// let frontiers = dsl.explore(&ec_params, &tasks);
    /// let (dsl, _frontiers) = dsl.compress(&params, &tasks, frontiers);
    ///
    /// // there should have been inventions because we started with a non-expressive DSL:
    /// assert!(!dsl.invented.is_empty());
    /// ```
    ///
    /// [Inducing Tree-Substitution Grammars]: http://jmlr.csail.mit.edu/papers/volume11/cohn10b/cohn10b.pdf
    /// [Fragment Grammars]: https://dspace.mit.edu/bitstream/handle/1721.1/44963/MIT-CSAIL-TR-2009-013.pdf
    pub fn compress<Observation: ?Sized>(
        &self,
        params: &CompressionParams,
        tasks: &[impl Task<Observation, Representation = Language, Expression = Expression>],
        frontiers: Vec<ECFrontier<Expression>>,
    ) -> (Self, Vec<ECFrontier<Expression>>) {
        compression::induce_fragment_grammar(self, params, tasks, frontiers)
    }

    /// Evaluate an expressions based on an input/output pair.
    ///
    /// Inputs are given as a sequence representing sequentially applied arguments.
    ///
    /// # Examples
    ///
    /// ```
    /// use polytype::{ptp, tp};
    /// use programinduction::lambda::{Language, SimpleEvaluator};
    ///
    /// fn evaluate(name: &str, inps: &[i32]) -> Result<i32, ()> {
    ///     match name {
    ///         "0" => Ok(0),
    ///         "1" => Ok(1),
    ///         "+" => Ok(inps[0] + inps[1]),
    ///         _ => unreachable!(),
    ///     }
    /// }
    ///
    /// let dsl = Language::uniform(vec![
    ///     ("0", ptp!(int)),
    ///     ("1", ptp!(int)),
    ///     ("+", ptp!(@arrow[tp!(int), tp!(int), tp!(int)])),
    /// ]);
    /// let eval = SimpleEvaluator::from(evaluate);
    /// let expr = dsl.parse("(λ (λ (+ (+ 1 $0) $1)))").unwrap();
    /// let inps = vec![2, 5];
    /// let evaluated = dsl.eval(&expr, eval, &inps).unwrap();
    /// assert_eq!(evaluated, 8);
    /// ```
    pub fn eval<V, E>(&self, expr: &Expression, evaluator: E, inps: &[V]) -> Result<V, E::Error>
    where
        V: Clone + PartialEq + Send + Sync,
        E: Evaluator<Space = V>,
    {
        eval::eval(self, expr, &Arc::new(evaluator), inps)
    }

    /// Like [`eval`], but useful in settings with a shared evaluator.
    ///
    /// [`eval`]: #method.eval
    pub fn eval_arc<V, E>(
        &self,
        expr: &Expression,
        evaluator: &Arc<E>,
        inps: &[V],
    ) -> Result<V, E::Error>
    where
        V: Clone + PartialEq + Send + Sync,
        E: Evaluator<Space = V>,
    {
        eval::eval(self, expr, evaluator, inps)
    }

    /// Like [`eval`], but for lazy evaluation with a [`LazyEvaluator`].
    ///
    /// [`eval`]: #method.eval
    /// [`LazyEvaluator`]: trait.LazyEvaluator.html
    pub fn lazy_eval<V, E>(
        &self,
        expr: &Expression,
        evaluator: E,
        inps: &[V],
    ) -> Result<V, E::Error>
    where
        V: Clone + PartialEq + Send + Sync,
        E: LazyEvaluator<Space = V>,
    {
        eval::lazy_eval(self, expr, &Arc::new(evaluator), inps)
    }

    /// Like [`eval_arc`], but for lazy evaluation with a [`LazyEvaluator`].
    ///
    /// [`eval_arc`]: #method.eval_arc
    /// [`LazyEvaluator`]: trait.LazyEvaluator.html
    pub fn lazy_eval_arc<V, E>(
        &self,
        expr: &Expression,
        evaluator: &Arc<E>,
        inps: &[V],
    ) -> Result<V, E::Error>
    where
        V: Clone + PartialEq + Send + Sync,
        E: LazyEvaluator<Space = V>,
    {
        eval::lazy_eval(self, expr, evaluator, inps)
    }

    /// Get the log-likelihood of an expression normalized with other expressions with the given
    /// request type.
    ///
    /// # Examples
    ///
    /// ```
    /// # use polytype::{ptp, tp};
    /// # use programinduction::lambda::Language;
    /// let dsl = Language::uniform(vec![
    ///     ("0", ptp!(int)),
    ///     ("1", ptp!(int)),
    ///     ("+", ptp!(@arrow[tp!(int), tp!(int), tp!(int)])),
    /// ]);
    /// let req = ptp!(@arrow[tp!(int), tp!(int), tp!(int)]);
    ///
    /// let expr = dsl.parse("(λ (λ (+ $0 $1)))").unwrap();
    /// assert_eq!(dsl.likelihood(&req, &expr), -5.545177444479561);
    ///
    /// let expr = dsl.parse("(λ (λ (+ (+ $0 1) $1)))").unwrap();
    /// assert_eq!(dsl.likelihood(&req, &expr), -8.317766166719343);
    /// ```
    pub fn likelihood(&self, request: &TypeScheme, expr: &Expression) -> f64 {
        enumerator::likelihood(self, request, expr)
    }

    /// Register a new invented expression. If it has a valid type, this will be `Ok(num)`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use polytype::{ptp, tp};
    /// # use programinduction::lambda::{Expression, Language};
    /// let mut dsl = Language::uniform(vec![
    ///     ("0", ptp!(int)),
    ///     ("1", ptp!(int)),
    ///     ("+", ptp!(@arrow[tp!(int), tp!(int), tp!(int)])),
    /// ]);
    /// let expr = dsl.parse("(+ 1)").unwrap();
    /// dsl.invent(expr.clone(), -0.5).unwrap();
    /// assert_eq!(
    ///     dsl.invented.get(0),
    ///     Some(&(expr, ptp!(@arrow[tp!(int), tp!(int)]), -0.5))
    /// );
    /// ```
    pub fn invent(
        &mut self,
        expr: Expression,
        log_probability: f64,
    ) -> Result<usize, InferenceError> {
        let tp = self.infer(&expr)?;
        self.invented.push((expr, tp, log_probability));
        Ok(self.invented.len() - 1)
    }

    /// Introduce a symmetry-breaking pattern to the Language.
    ///
    /// # Examples
    ///
    /// ```
    /// # use polytype::{ptp, tp};
    /// # use programinduction::lambda::{Expression, Language};
    /// let mut dsl = Language::uniform(vec![
    ///     ("0", ptp!(int)),
    ///     ("1", ptp!(int)),
    ///     ("+", ptp!(@arrow[tp!(int), tp!(int), tp!(int)])),
    /// ]);
    /// // disallow (+ 0 _) and (+ _ 0)
    /// dsl.add_symmetry_violation(2, 0, 0);
    /// dsl.add_symmetry_violation(2, 1, 0);
    /// // disallow (+ (+ ..) _), so effort isn't wasted with (+ _ (+ ..))
    /// dsl.add_symmetry_violation(2, 0, 2);
    ///
    /// let exprs: Vec<Expression> = dsl.enumerate(ptp!(int))
    ///     .take(6)
    ///     .map(|(expr, _log_prior)| expr)
    ///     .collect();
    ///
    /// // enumeration can be far more effective with symmetry-breaking:
    /// assert_eq!(
    ///     exprs,
    ///     vec![
    ///         Expression::Primitive(0),
    ///         Expression::Primitive(1),
    ///         dsl.parse("(+ 1 1)").unwrap(),
    ///         dsl.parse("(+ 1 (+ 1 1))").unwrap(),
    ///         dsl.parse("(+ 1 (+ 1 (+ 1 1)))").unwrap(),
    ///         dsl.parse("(+ 1 (+ 1 (+ 1 (+ 1 1))))").unwrap(),
    ///     ]
    /// );
    /// ```
    pub fn add_symmetry_violation(&mut self, primitive: usize, arg_index: usize, arg: usize) {
        let x = (primitive, arg_index, arg);
        if let Err(i) = self.symmetry_violations.binary_search(&x) {
            self.symmetry_violations.insert(i, x)
        }
    }
    /// Check whether expressions break symmetry.
    ///
    /// # Examples
    ///
    /// ```
    /// # use polytype::{ptp, tp};
    /// # use programinduction::lambda::{Expression, Language};
    /// let mut dsl = Language::uniform(vec![
    ///     ("0", ptp!(int)),
    ///     ("1", ptp!(int)),
    ///     ("+", ptp!(@arrow[tp!(int), tp!(int), tp!(int)])),
    /// ]);
    /// dsl.add_symmetry_violation(2, 0, 0);
    /// dsl.add_symmetry_violation(2, 1, 0);
    /// dsl.add_symmetry_violation(2, 0, 2);
    ///
    /// let f = &Expression::Primitive(2); // +
    /// let x = &Expression::Primitive(0); // 0
    /// assert!(dsl.violates_symmetry(f, 0, x));
    /// let x = &dsl.parse("(+ 1 1)").unwrap();
    /// assert!(dsl.violates_symmetry(f, 0, x));
    /// assert!(!dsl.violates_symmetry(f, 1, x));
    /// ```
    pub fn violates_symmetry(&self, f: &Expression, index: usize, x: &Expression) -> bool {
        match (f, x) {
            (Expression::Primitive(f), Expression::Primitive(x)) => {
                let x = (*f, index, *x);
                self.symmetry_violations.binary_search(&x).is_ok()
            }
            (Expression::Primitive(f), Expression::Application(x, _)) => {
                let mut z: &Expression = x;
                while let Expression::Application(x, _) = z {
                    z = x
                }
                if let Expression::Primitive(x) = z {
                    let x = (*f, index, *x);
                    self.symmetry_violations.binary_search(&x).is_ok()
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// Remove all invented expressions by pulling out their underlying expressions.
    pub fn strip_invented(&self, expr: &Expression) -> Expression {
        expr.strip_invented(&self.invented)
    }

    /// A cheap function used as the objective for dsl compression. See
    /// [`lambda::CompressionParams`] for details.
    ///
    /// [`lambda::CompressionParams`]: struct.CompressionParams.html
    pub fn score(&self, joint_mdl: f64, params: &CompressionParams) -> f64 {
        let nparams = self.primitives.len() + self.invented.len();
        let structure = (self.primitives.len() as f64)
            + self
                .invented
                .iter()
                .map(|(expr, _, _)| {
                    let (leaves, free, bound) = compression::expression_count_kinds(expr, 0);
                    (leaves as f64)
                        + BOUND_VAR_COST * (bound as f64)
                        + FREE_VAR_COST * (free as f64)
                })
                .sum::<f64>();
        joint_mdl - params.aic * (nparams as f64) - params.structure_penalty * structure
    }

    /// Computes the joint minimum description length over all frontiers.
    pub fn joint_mdl(&self, frontiers: &[RescoredFrontier]) -> f64 {
        compression::joint_mdl(self, frontiers)
    }

    /// Runs a variant of the inside outside algorithm to assign production probabilities for the
    /// primitives. The joint minimum description length is returned.
    pub fn inside_outside(&mut self, frontiers: &[RescoredFrontier], pseudocounts: u64) -> f64 {
        compression::inside_outside(self, frontiers, pseudocounts)
    }

    /// The inverse of [`display`].
    ///
    /// Lambda expressions take the form `(lambda BODY)` or `(λ BODY)`, where BODY is an expression
    /// that may use a corresponding De Bruijn [`Index`].
    ///
    /// [`display`]: #method.display
    /// [`Index`]: enum.Expression.html#variant.Index
    pub fn parse(&self, inp: &str) -> Result<Expression, ParseError> {
        parser::parse(self, inp)
    }
    /// The inverse of [`parse`].
    ///
    /// [`parse`]: #method.parse
    pub fn display(&self, expr: &Expression) -> String {
        expr.show(self, false)
    }

    /// Like `display`, but in a format ready for lisp interpretation.
    pub fn lispify(&self, expr: &Expression, conversions: &HashMap<String, String>) -> String {
        expr.as_lisp(self, false, conversions, 0)
    }

    fn candidates(
        &self,
        request: &Type,
        ctx: &Context,
        env: &VecDeque<Type>,
    ) -> Vec<(f64, Expression, Type, Context)> {
        // make cands as big as possible to prevent reallocation
        let mut cands = Vec::with_capacity(self.primitives.len() + self.invented.len() + env.len());
        // primitives and inventions
        let prims = self
            .primitives
            .iter()
            .enumerate()
            .map(|(i, &(_, ref tp, p))| (p, tp, Expression::Primitive(i)));
        let invented = self
            .invented
            .iter()
            .enumerate()
            .map(|(i, &(_, ref tp, p))| (p, tp, Expression::Invented(i)));
        for (p, tp, expr) in prims.chain(invented) {
            let mut ctx = ctx.clone();
            let mut tp = tp.clone().instantiate_owned(&mut ctx);
            let unifies = {
                let ret = if let Some(ret) = tp.returns() {
                    ret
                } else {
                    &tp
                };
                ctx.unify_fast(ret.clone(), request.clone()).is_ok()
            };
            if unifies {
                tp.apply_mut(&ctx);
                cands.push((p, expr, tp, ctx))
            }
        }
        // indexed
        let indexed_start = cands.len();
        for (i, tp) in env.iter().enumerate() {
            let expr = Expression::Index(i);
            let mut ctx = ctx.clone();
            let ret = if let Some(ret) = tp.returns() {
                ret
            } else {
                tp
            };
            if ctx.unify_fast(ret.clone(), request.clone()).is_ok() {
                let mut tp = tp.clone();
                tp.apply_mut(&ctx);
                cands.push((self.variable_logprob, expr, tp, ctx))
            }
        }
        // update probabilities for indices
        let log_n_indexed = ((cands.len() - indexed_start) as f64).ln();
        for c in &mut cands[indexed_start..] {
            c.0 -= log_n_indexed
        }
        // normalize
        let p_largest = cands
            .iter()
            .take(indexed_start + 1)
            .map(|&(p, _, _, _)| p)
            .fold(f64::NEG_INFINITY, f64::max);
        let z = p_largest
            + cands
                .iter()
                .map(|&(p, _, _, _)| (p - p_largest).exp())
                .sum::<f64>()
                .ln();
        for c in &mut cands {
            c.0 -= z;
        }
        cands
    }
}
impl<Observation: ?Sized> EC<Observation> for Language {
    type Expression = Expression;
    type Params = CompressionParams;
    fn enumerate<F>(&self, tp: TypeScheme, termination_condition: F)
    where
        F: Fn(Expression, f64) -> bool + Sync,
    {
        enumerator::run(self, tp, termination_condition)
    }
    fn compress(
        &self,
        params: &Self::Params,
        tasks: &[impl Task<Observation, Representation = Self, Expression = Self::Expression>],
        frontiers: Vec<ECFrontier<Expression>>,
    ) -> (Self, Vec<ECFrontier<Expression>>) {
        self.compress(params, tasks, frontiers)
    }
}

/// Expressions of lambda calculus, which only make sense with an accompanying [`Language`].
///
/// [`Language`]: struct.Language.html
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum Expression {
    /// The number associated with a primitive is used by the Language to identify the primitive.
    Primitive(usize),
    Application(Box<Expression>, Box<Expression>),
    Abstraction(Box<Expression>),
    /// De Bruijn index referring to the nth-nearest abstraction (0-indexed).
    /// For example, the identify function is `(λ $0)` or `Abstraction(Index(0))`.
    Index(usize),
    /// The number associated with an invented expression is used by the Language to identify the
    /// invention.
    Invented(usize),
}
impl Expression {
    fn infer(
        &self,
        dsl: &Language,
        ctx: &mut Context,
        env: &VecDeque<Type>,
        indices: &mut HashMap<usize, Type>,
    ) -> Result<Type, InferenceError> {
        match *self {
            Expression::Primitive(num) => {
                if let Some(prim) = dsl.primitives.get(num) {
                    Ok(prim.1.clone().instantiate_owned(ctx))
                } else {
                    Err(InferenceError::InvalidPrimitive(num))
                }
            }
            Expression::Application(ref f, ref x) => {
                let f_tp = f.infer(dsl, ctx, env, indices)?;
                let x_tp = x.infer(dsl, ctx, env, indices)?;
                let ret_tp = ctx.new_variable();
                ctx.unify(&f_tp, &Type::arrow(x_tp, ret_tp.clone()))?;
                Ok(ret_tp.apply(ctx))
            }
            Expression::Abstraction(ref body) => {
                let arg_tp = ctx.new_variable();
                let mut env = env.clone();
                env.push_front(arg_tp.clone());
                let ret_tp = body.infer(dsl, ctx, &env, indices)?;
                let mut tp = Type::arrow(arg_tp, ret_tp);
                tp.apply_mut(ctx);
                Ok(tp)
            }
            Expression::Index(i) => {
                if i < env.len() {
                    let mut tp = env[i].clone();
                    tp.apply_mut(ctx);
                    Ok(tp)
                } else {
                    let mut tp = indices
                        .entry(i - env.len())
                        .or_insert_with(|| ctx.new_variable())
                        .clone();
                    tp.apply_mut(ctx);
                    Ok(tp)
                }
            }
            Expression::Invented(num) => {
                if let Some(inv) = dsl.invented.get(num) {
                    Ok(inv.1.clone().instantiate_owned(ctx))
                } else {
                    Err(InferenceError::InvalidInvention(num))
                }
            }
        }
    }
    /// Puts a beta-normalized expression in eta-long form. Invalid types or non-beta-normalized
    /// expression may cause this function to return `false` to indicate that an error occurred.
    ///
    /// # Examples
    ///
    /// ```
    /// # use polytype::{ptp, tp};
    /// # use programinduction::lambda::{Expression, Language};
    /// let mut dsl = Language::uniform(vec![
    ///     ("+", ptp!(@arrow[tp!(int), tp!(int), tp!(int)])),
    /// ]);
    /// let mut expr = dsl.parse("+").unwrap();
    /// expr.etalong(&dsl);
    /// assert_eq!(dsl.display(&expr), "(λ (λ (+ $1 $0)))");
    /// ```
    pub fn etalong(&mut self, dsl: &Language) -> bool {
        if let Ok(tps) = dsl.infer(self) {
            let env = Rc::new(LinkedList::default());
            let mut ctx = Context::default();
            let req = tps.instantiate(&mut ctx);
            self.etalong_internal(dsl, &env, &mut ctx, &req)
        } else {
            false
        }
    }
    fn etalong_internal(
        &mut self,
        dsl: &Language,
        env: &Rc<LinkedList<Type>>,
        ctx: &mut Context,
        req: &Type,
    ) -> bool {
        if let Expression::Abstraction(ref mut b) = *self {
            return if let Some((arg, ret)) = req.as_arrow() {
                let env = LinkedList::prepend(env, arg.clone());
                b.etalong_internal(dsl, &env, ctx, ret)
            } else {
                eprintln!(
                    "eta-long type mismatch expr={} ; tp={}",
                    dsl.display(&Expression::Abstraction(b.clone())),
                    req
                );
                false
            };
        }
        if req.as_arrow().is_some() {
            let mut x = self.clone();
            x.shift(1);
            *self = Expression::Abstraction(Box::new(Expression::Application(
                Box::new(x),
                Box::new(Expression::Index(0)),
            )));
            return self.etalong_internal(dsl, env, ctx, req);
        }
        let new_self = match *self {
            Expression::Abstraction(_) => unreachable!(),
            Expression::Application(ref f, ref x) => {
                let mut f = f;
                let mut xs: Vec<Expression> = vec![*x.clone()];
                while let Expression::Application(ref ff, ref fx) = **f {
                    f = ff;
                    xs.push(*fx.clone());
                }
                xs.reverse();
                let ft = match **f {
                    Expression::Abstraction(_) => {
                        eprintln!(
                            "eta-long called on non-beta-normalized expression {}",
                            dsl.display(self)
                        );
                        return false;
                    }
                    Expression::Application(_, _) => unreachable!(),
                    Expression::Primitive(i) => dsl.primitives[i].1.instantiate(ctx),
                    Expression::Invented(i) => dsl.invented[i].1.instantiate(ctx),
                    Expression::Index(i) => env[i].apply(ctx),
                };
                if let Err(e) = ctx.unify(req, ft.returns().unwrap_or(&ft)) {
                    eprintln!("eta-long type mismatch: {}", e);
                    return false;
                }
                let ft = ft.apply(ctx);
                let xt = ft.args().unwrap_or_default();
                if xs.len() != xt.len() {
                    eprintln!(
                        "eta-long type mismatch, {} args but type was {}",
                        xs.len(),
                        ft
                    );
                    return false;
                }
                let mut f = f.clone();
                for (mut x, t) in xs.into_iter().zip(xt) {
                    let t = t.apply(ctx);
                    if !x.etalong_internal(dsl, env, ctx, &t) {
                        return false;
                    }
                    f = Box::new(Expression::Application(f, Box::new(x)))
                }
                *f
            }
            Expression::Primitive(i) => {
                let t = dsl.primitives[i].1.instantiate(ctx);
                return ctx.unify(&t, req).is_ok();
            }
            Expression::Invented(i) => {
                let t = dsl.invented[i].1.instantiate(ctx);
                return ctx.unify(&t, req).is_ok();
            }
            Expression::Index(i) => return ctx.unify(&env[i], req).is_ok(),
        };
        *self = new_self;
        true
    }
    fn strip_invented(&self, invented: &[(Expression, TypeScheme, f64)]) -> Expression {
        match *self {
            Expression::Application(ref f, ref x) => Expression::Application(
                Box::new(f.strip_invented(invented)),
                Box::new(x.strip_invented(invented)),
            ),
            Expression::Abstraction(ref body) => {
                Expression::Abstraction(Box::new(body.strip_invented(invented)))
            }
            Expression::Invented(num) => invented[num].0.strip_invented(invented),
            _ => self.clone(),
        }
    }
    /// Shifts all free variables (indexes) in the expression. If `offset` is negative, then
    /// variables will not be changed if they are made to be negative. The return value is always
    /// `true` unless this scenario occurs.
    pub fn shift(&mut self, offset: i64) -> bool {
        self.shift_internal(offset, 0)
    }
    fn shift_internal(&mut self, offset: i64, depth: usize) -> bool {
        match *self {
            Expression::Index(ref mut i) => {
                if *i < depth {
                    true
                } else if offset >= 0 {
                    *i += offset as usize;
                    true
                } else if let Some(ni) = i.checked_sub((-offset) as usize) {
                    *i = ni;
                    true
                } else {
                    false
                }
            }
            Expression::Application(ref mut f, ref mut x) => {
                let a = f.shift_internal(offset, depth);
                let b = x.shift_internal(offset, depth);
                a && b
            }
            Expression::Abstraction(ref mut body) => body.shift_internal(offset, depth + 1),
            _ => true,
        }
    }
    fn as_lisp(
        &self,
        dsl: &Language,
        is_function: bool,
        conversions: &HashMap<String, String>,
        depth: usize,
    ) -> String {
        match *self {
            Expression::Primitive(num) => {
                let name = &dsl.primitives[num].0;
                conversions.get(name).unwrap_or(name).to_string()
            }
            Expression::Application(ref f, ref x) => {
                let f_lisp = f.as_lisp(dsl, true, conversions, depth);
                let x_lisp = x.as_lisp(dsl, false, conversions, depth);
                if is_function {
                    format!("{} {}", f_lisp, x_lisp)
                } else {
                    format!("({} {})", f_lisp, x_lisp)
                }
            }
            Expression::Abstraction(ref body) => {
                let var = (97 + depth as u8) as char;
                format!(
                    "(λ ({}) {})",
                    var,
                    body.as_lisp(dsl, false, conversions, depth + 1)
                )
            }
            Expression::Index(i) => {
                let var = (96 + (depth - i) as u8) as char;
                format!("{}", var)
            }
            Expression::Invented(num) => {
                dsl.invented[num].0.as_lisp(dsl, false, conversions, depth)
            }
        }
    }
    fn show(&self, dsl: &Language, is_function: bool) -> String {
        match *self {
            Expression::Primitive(num) => dsl.primitives[num].0.clone(),
            Expression::Application(ref f, ref x) => {
                if is_function {
                    format!("{} {}", f.show(dsl, true), x.show(dsl, false))
                } else {
                    format!("({} {})", f.show(dsl, true), x.show(dsl, false))
                }
            }
            Expression::Abstraction(ref body) => format!("(λ {})", body.show(dsl, false)),
            Expression::Index(i) => format!("${}", i),
            Expression::Invented(num) => {
                format!("#{}", dsl.invented[num].0.show(dsl, false))
            }
        }
    }
}

/// Create a task based on evaluating lambda calculus expressions on test input/output pairs.
///
/// Here we let all tasks be represented by input/output pairs that are values in the space of
/// type `V`. For example, circuits may have `V` be just `bool`, whereas string editing may
/// have `V` be an enum featuring strings, chars, and natural numbers. All inputs, outputs, and
/// evaluated expressions must be representable by `V`.
///
/// An `evaluator` takes the name of a primitive and a vector of sequential inputs to the
/// expression (so an expression with unary type will have one input in a vec of size 1).
///
/// The resulting task is "all-or-nothing": the oracle returns either `0` if all examples are
/// correctly hit or `f64::NEG_INFINITY` otherwise.
///
/// # Examples
///
/// ```
/// use polytype::{ptp, tp};
/// use programinduction::{Task, lambda::{task_by_evaluation, Language, SimpleEvaluator}};
///
/// fn evaluate(name: &str, inps: &[i32]) -> Result<i32, ()> {
///     match name {
///         "0" => Ok(0),
///         "1" => Ok(1),
///         "+" => Ok(inps[0] + inps[1]),
///         _ => unreachable!(),
///     }
/// }
///
/// let examples = vec![(vec![2, 5], 8), (vec![1, 2], 4)];
/// let tp = ptp!(@arrow[tp!(int), tp!(int), tp!(int)]);
/// let task = task_by_evaluation(SimpleEvaluator::from(evaluate), tp, &examples);
///
/// let dsl = Language::uniform(vec![
///     ("0", ptp!(int)),
///     ("1", ptp!(int)),
///     ("+", ptp!(@arrow[tp!(int), tp!(int), tp!(int)])),
/// ]);
/// let expr = dsl.parse("(λ (+ (+ 1 $0)))").unwrap();
/// assert!(task.oracle(&dsl, &expr).is_finite())
/// ```
pub fn task_by_evaluation<E, V>(
    evaluator: E,
    tp: TypeScheme,
    examples: impl AsRef<[(Vec<V>, V)]> + Sync,
) -> impl Task<[(Vec<V>, V)], Representation = Language, Expression = Expression>
where
    E: Evaluator<Space = V> + Send,
    V: PartialEq + Clone + Send + Sync,
{
    LambdaTask::<false, _, _> {
        evaluator: Arc::new(evaluator),
        tp,
        examples,
    }
}

/// Like [`task_by_evaluation`], but for use with a [`LazyEvaluator`].
///
/// [`LazyEvaluator`]: trait.LazyEvaluator.html
/// [`task_by_evaluation`]: fn.task_by_evaluation.html
pub fn task_by_lazy_evaluation<E, V>(
    evaluator: E,
    tp: TypeScheme,
    examples: impl AsRef<[(Vec<V>, V)]> + Sync,
) -> impl Task<[(Vec<V>, V)], Representation = Language, Expression = Expression>
where
    E: LazyEvaluator<Space = V> + Send,
    V: PartialEq + Clone + Send + Sync,
{
    LambdaTask::<true, _, _> {
        evaluator: Arc::new(evaluator),
        tp,
        examples,
    }
}

struct LambdaTask<const LAZY: bool, E, O: Sync> {
    evaluator: Arc<E>,
    tp: TypeScheme,
    examples: O,
}
impl<
        V: PartialEq + Clone + Send + Sync,
        E: Evaluator<Space = V> + Send,
        O: AsRef<[(Vec<V>, V)]> + Sync,
    > Task<[(Vec<V>, V)]> for LambdaTask<false, E, O>
{
    type Representation = Language;
    type Expression = Expression;

    fn oracle(&self, dsl: &Language, expr: &Expression) -> f64 {
        let success = self.examples.as_ref().iter().all(|(inps, out)| {
            let result = dsl.eval_arc(expr, &self.evaluator, inps);
            if let Ok(o) = result {
                o == *out
            } else {
                false
            }
        });
        if success {
            0f64
        } else {
            f64::NEG_INFINITY
        }
    }
    fn tp(&self) -> &TypeScheme {
        &self.tp
    }
    fn observation(&self) -> &[(Vec<V>, V)] {
        self.examples.as_ref()
    }
}
impl<
        V: PartialEq + Clone + Send + Sync,
        E: LazyEvaluator<Space = V> + Send,
        O: AsRef<[(Vec<V>, V)]> + Sync,
    > Task<[(Vec<V>, V)]> for LambdaTask<true, E, O>
{
    type Representation = Language;
    type Expression = Expression;

    fn oracle(&self, dsl: &Language, expr: &Expression) -> f64 {
        let success = self.examples.as_ref().iter().all(|(inps, out)| {
            let result = dsl.lazy_eval_arc(expr, &self.evaluator, inps);
            if let Ok(o) = result {
                o == *out
            } else {
                false
            }
        });
        if success {
            0f64
        } else {
            f64::NEG_INFINITY
        }
    }
    fn tp(&self) -> &TypeScheme {
        &self.tp
    }
    fn observation(&self) -> &[(Vec<V>, V)] {
        self.examples.as_ref()
    }
}

#[derive(Debug, Clone)]
struct LinkedList<T: Clone>(Option<(T, Rc<LinkedList<T>>)>);
impl<T: Clone> LinkedList<T> {
    fn prepend(lst: &Rc<LinkedList<T>>, v: T) -> Rc<LinkedList<T>> {
        Rc::new(LinkedList(Some((v, lst.clone()))))
    }
    fn as_vecdeque(&self) -> VecDeque<T> {
        let mut lst: &Rc<LinkedList<T>>;
        let mut out = VecDeque::new();
        if let Some((ref v, ref nlst)) = self.0 {
            out.push_back(v.clone());
            lst = nlst;
            while let Some((ref v, ref nlst)) = lst.0 {
                out.push_back(v.clone());
                lst = nlst;
            }
        }
        out
    }
    fn len(&self) -> usize {
        let mut lst: &Rc<LinkedList<T>>;
        let mut n = 0;
        if let Some((_, ref nlst)) = self.0 {
            n += 1;
            lst = nlst;
            while let Some((_, ref nlst)) = lst.0 {
                n += 1;
                lst = nlst;
            }
        }
        n
    }
}
impl<T: Clone> Default for LinkedList<T> {
    fn default() -> Self {
        LinkedList(None)
    }
}
impl<T: Clone> Index<usize> for LinkedList<T> {
    type Output = T;
    fn index(&self, i: usize) -> &Self::Output {
        let mut lst: &Rc<LinkedList<T>>;
        let mut n = 0;
        if let Some((ref v, ref nlst)) = self.0 {
            if i == n {
                return v;
            }
            n += 1;
            lst = nlst;
            while let Some((ref v, ref nlst)) = lst.0 {
                if i == n {
                    return v;
                }
                n += 1;
                lst = nlst;
            }
        }
        panic!("index out of bounds");
    }
}

#[derive(Debug, Clone)]
pub enum InferenceError {
    InvalidPrimitive(usize),
    InvalidInvention(usize),
    Unify(UnificationError),
}
impl From<UnificationError> for InferenceError {
    fn from(err: UnificationError) -> Self {
        InferenceError::Unify(err)
    }
}
impl std::fmt::Display for InferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        match self {
            InferenceError::InvalidPrimitive(n) => write!(f, "primitive {} not in Language", n),
            InferenceError::InvalidInvention(n) => write!(f, "invention {} not in Language", n),
            InferenceError::Unify(err) => write!(f, "could not unify to infer type: {}", err),
        }
    }
}
impl Error for InferenceError {
    fn description(&self) -> &str {
        "could not infer type"
    }
}
