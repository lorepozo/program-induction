use itertools::Itertools;
use polytype::{Context, Type, TypeSchema};
use rayon::join;
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};
use std::f64;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use super::{Expression, Language, LinkedList};
use utils::bounded;
use {ECFrontier, Task};

/// Parameters for grammar induction.
///
/// Proposed grammars are scored as `likelihood - aic * #primitives - structure_penalty * #nodes`.
/// Additionally, `pseudocounts` affects the likelihood calculation, and `topk` and `arity` affect
/// what fragments can be proposed.
pub struct CompressionParams {
    /// Pseudocounts are added to the observed counts associated with each primitive and invented
    /// expression.
    pub pseudocounts: u64,
    /// Rather than using every expression in the frontier for proposing fragments, only use the
    /// `topk` best expressions in each frontier.
    pub topk: usize,
    /// Structure penalty penalizes the total number of nodes in each [`Expression`] of the
    /// grammar's primitives and invented expressions.
    ///
    /// [`Expression`]: enum.Expression.html
    pub structure_penalty: f64,
    /// Determines whether to use the maximum a-posteriori value for topk evaluation, or whether to
    /// use only the likelihood. Leave this to `false` unless you know what you are doing.
    pub topk_use_only_likelihood: bool,
    /// AIC is a penalty in the number of parameters, i.e. the number of primitives and invented
    /// expressions.
    pub aic: f64,
    /// Arity is the largest applicative depth of an expression that may be manipulated to propose
    /// a fragment.
    pub arity: u32,
}
impl Default for CompressionParams {
    /// The default params prevent completely discarding of primives by having non-zero
    /// pseudocounts.
    ///
    /// ```
    /// # use programinduction::lambda::CompressionParams;
    /// CompressionParams {
    ///     pseudocounts: 5,
    ///     topk: 2,
    ///     topk_use_only_likelihood: false,
    ///     structure_penalty: 1f64,
    ///     aic: 1f64,
    ///     arity: 2,
    /// }
    /// # ;
    /// ```
    fn default() -> Self {
        CompressionParams {
            pseudocounts: 5,
            topk: 2,
            topk_use_only_likelihood: false,
            structure_penalty: 1f64,
            aic: 1f64,
            arity: 2,
        }
    }
}

/// This function makes it easier to write your own compression scheme.
/// It takes the role of a single compression step, separating it into sub-steps that are decent to
/// implement in isolation.
///
/// This is a sophisticated higher-order function — tread carefully.
/// - `state: I` can be mutated by making it `Arc<RwLock<_>>`, though it will often just be `()`
///   unless you really need it.
/// - type `X` is for a _candidate_, something which can be used to update a dsl.
/// - `proposer` pushes a bunch of candidates to the given vector.
/// - `proposal_to_dsl` adds the candidate to the dsl and returns a new joint minimum description
///   length for the dsl. For example, this may be set to:
///
///   ```ignore
///   |_state, expr, dsl, frontiers, params| {
///       if dsl.invent(expr.clone(), 0.).is_ok() {
///           Some(dsl.inside_outside(frontiers, params.pseudocounts))
///       } else {
///           None
///       }
///   }
///   ```
/// - `defragment` is most often a no-op and can be set to `|x| x`. It allows you to effectively
///   change the output of `proposal_to_expr` after scoring has been done. This is useful for
///   fragment grammar compression, because scoring with inventions that have free variables (i.e.
///   non-closed expressions) will let inside-outside capture those uses without having to rewrite
///   the frontiers.
/// - `rewrite_frontiers` finally takes the highest-scoring dsl, which is guaranteed to have a
///   single latest invention (accessible via `dsl.invented.last().unwrap()`) equal to
///   `defragment(proposal_to_expr(_, proposal))` for some generated proposal. The
///   [`lambda::Expression`] it is supplied is the non-defragmented proposal.
///   There's no need to rescore the frontiers, that's done automatically.
///
/// We recommended to make a function that adapts this into a four-argument `induce_my_algo`
/// function by filling in the higher-order functions. See the source code of this project to find
/// the particular use of this function that gives [`Language::compress`] using a
/// fragment-grammar-like compression scheme.
///
/// [`Language::compress`]: struct.Language.html#method.compress
/// [`lambda::CompressionParams`]: struct.CompressionParams.html
/// [`lambda::Expression`]: enum.Expression.html
/// [`lambda::Language`]: struct.Language.html
#[cfg_attr(feature = "cargo-clippy", allow(too_many_arguments))]
pub fn induce<O: Sync, I, X, P, D, F, R>(
    dsl: &Language,
    params: &CompressionParams,
    tasks: &[Task<Language, Expression, O>],
    mut original_frontiers: Vec<ECFrontier<Language>>,
    state: I,
    proposer: P,
    proposal_to_dsl: D,
    defragment: F,
    rewrite_frontiers: R,
) -> (Language, Vec<ECFrontier<Language>>)
where
    X: Send,
    I: Send + Sync,
    P: Fn(
            &I,
            &Language,
            &[(TypeSchema, Vec<(Expression, f64, f64)>)],
            &CompressionParams,
            &mut Vec<X>,
        )
        + Sync,
    D: Fn(&I, &X, &mut Language, &[(TypeSchema, Vec<(Expression, f64, f64)>)], &CompressionParams)
            -> Option<f64>
        + Sync,
    F: Fn(Expression) -> Expression,
    R: Fn(
        &I,
        X,
        Expression,
        &Language,
        &mut Vec<(TypeSchema, Vec<(Expression, f64, f64)>)>,
        &CompressionParams,
    ),
{
    let mut dsl = dsl.clone();
    let mut frontiers: Vec<RescoredFrontier> = tasks
        .par_iter()
        .map(|t| t.tp.clone())
        .zip(&original_frontiers)
        .filter(|&(_, f)| !f.is_empty())
        .map(|(tp, f)| (tp, f.0.clone()))
        .collect();

    let joint_mdl = dsl.inside_outside(&frontiers, params.pseudocounts);
    let mut best_score = dsl.score(joint_mdl, params);

    if cfg!(feature = "verbose") {
        eprintln!("COMPRESSION: starting score: {}", best_score)
    }
    if params.aic.is_finite() {
        loop {
            let (candidate, fragment_expr) = {
                let rescored_frontiers: Vec<_> = frontiers
                    .par_iter()
                    .cloned()
                    .map(|f| dsl.rescore_frontier(f, params.topk, params.topk_use_only_likelihood))
                    .collect();
                let mut proposals = Vec::new();
                proposer(&state, &dsl, &rescored_frontiers, params, &mut proposals);
                if cfg!(feature = "verbose") {
                    eprintln!("COMPRESSION: proposed {} fragments", proposals.len())
                }
                let best_proposal = proposals
                    .into_par_iter()
                    .filter_map(|candidate| {
                        let mut dsl = dsl.clone();
                        let joint_mdl = match proposal_to_dsl(
                            &state,
                            &candidate,
                            &mut dsl,
                            &rescored_frontiers,
                            params,
                        ) {
                            None => {
                                if cfg!(feature = "verbose") {
                                    eprintln!("COMPRESSION: dropped invalid proposal");
                                }
                                return None;
                            }
                            Some(joint_mdl) => joint_mdl,
                        };
                        let s = dsl.score(joint_mdl, params);
                        if s.is_finite() {
                            Some((dsl, candidate, s))
                        } else {
                            None
                        }
                    })
                    .max_by(|&(_, _, ref x), &(_, _, ref y)| x.partial_cmp(y).unwrap());
                if best_proposal.is_none() {
                    if cfg!(feature = "verbose") {
                        eprintln!("COMPRESSION: no sufficient proposals")
                    }
                    break;
                }
                let (new_dsl, candidate, new_score) = best_proposal.unwrap();
                if new_score <= best_score {
                    if cfg!(feature = "verbose") {
                        eprintln!("COMPRESSION: score did not improve")
                    }
                    break;
                }
                dsl = new_dsl;
                best_score = new_score;

                let (fragment_expr, _, log_prior) = dsl.invented.pop().unwrap();
                let inv = defragment(fragment_expr.clone());
                if cfg!(feature = "verbose") {
                    eprintln!(
                        "COMPRESSION: score improved to {} with invention {} (defragmented from candidate expr {})",
                        best_score,
                        dsl.display(&inv),
                        dsl.display(&fragment_expr)
                    )
                }
                dsl.invent(inv, log_prior).expect("invalid invention");
                (candidate, fragment_expr)
            };
            rewrite_frontiers(
                &state,
                candidate,
                fragment_expr,
                &dsl,
                &mut frontiers,
                &params,
            )
        }
    }
    frontiers.reverse();
    for f in &mut original_frontiers {
        if !f.is_empty() {
            f.0 = frontiers.pop().unwrap().1;
        }
    }
    (dsl, original_frontiers)
}

/// A convenient frontier representation.
pub type RescoredFrontier = (TypeSchema, Vec<(Expression, f64, f64)>);

pub fn joint_mdl(dsl: &Language, frontiers: &[RescoredFrontier]) -> f64 {
    frontiers
        .par_iter()
        .map(|(t, f)| {
            f.iter()
                .map(|e| e.2 + dsl.likelihood(t, &e.0))
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .sum::<f64>()
}

/// Runs a variant of the inside outside algorithm to assign production probabilities for all
/// primitives and invented expressions. The joint minimum description length is returned.
pub fn inside_outside(
    dsl: &mut Language,
    frontiers: &[RescoredFrontier],
    pseudocounts: u64,
) -> f64 {
    dsl.inside_outside_internal(&frontiers, pseudocounts)
}

pub fn induce_fragment_grammar<O: Sync>(
    dsl: &Language,
    params: &CompressionParams,
    tasks: &[Task<Language, Expression, O>],
    original_frontiers: Vec<ECFrontier<Language>>,
) -> (Language, Vec<ECFrontier<Language>>) {
    induce(
        dsl,
        params,
        tasks,
        original_frontiers,
        (),
        |_, dsl, rescored_frontiers, params, proposals| {
            dsl.propose_inventions(rescored_frontiers, params.arity, proposals)
        },
        |_, expr, dsl, rescored_frontiers, params| {
            if dsl.invent(expr.clone(), 0.).is_ok() {
                Some(dsl.inside_outside(rescored_frontiers, params.pseudocounts))
            } else {
                None
            }
        },
        proposals::defragment,
        |_, fragment_expr, _, dsl, frontiers, _| {
            let i = dsl.invented.len() - 1;
            for mut f in frontiers {
                dsl.rewrite_frontier_with_fragment_expression(&mut f, i, &fragment_expr);
            }
        },
    )
}

/// Extend the Language in our scope so we can do useful compression things.
impl Language {
    fn rescore_frontier(
        &self,
        f: RescoredFrontier,
        topk: usize,
        topk_use_only_likelihood: bool,
    ) -> RescoredFrontier {
        let xs = f
            .1
            .iter()
            .map(|&(ref expr, _, loglikelihood)| {
                let logprior = self.uses(&f.0, expr).0;
                (expr, logprior, loglikelihood, logprior + loglikelihood)
            })
            .sorted_by(|&(_, _, ref xl, ref xpost), &(_, _, ref yl, ref ypost)| {
                if topk_use_only_likelihood {
                    yl.partial_cmp(xl).unwrap()
                } else {
                    ypost.partial_cmp(xpost).unwrap()
                }
            })
            .into_iter()
            .take(topk)
            .map(|(expr, logprior, loglikelihood, _)| (expr.clone(), logprior, loglikelihood))
            .collect();
        (f.0, xs)
    }

    fn reset_uniform(&mut self) {
        for x in &mut self.primitives {
            x.2 = 0f64;
        }
        for x in &mut self.invented {
            x.2 = 0f64;
        }
        self.variable_logprob = 0f64;
    }

    fn inside_outside_internal(
        &mut self,
        frontiers: &[RescoredFrontier],
        pseudocounts: u64,
    ) -> f64 {
        self.reset_uniform();
        let pseudocounts = pseudocounts as f64;
        let (joint_mdl, u) = self.all_uses(frontiers);
        self.variable_logprob = (u.actual_vars + pseudocounts).ln() - u.possible_vars.ln();
        if !self.variable_logprob.is_finite() {
            self.variable_logprob = u.actual_vars.max(1f64).ln()
        }
        for (i, prim) in self.primitives.iter_mut().enumerate() {
            let obs = u.actual_prims[i] + pseudocounts;
            let pot = u.possible_prims[i];
            let pot = if pot == 0f64 { pseudocounts } else { pot };
            prim.2 = obs.ln() - pot.ln();
        }
        for (i, inv) in self.invented.iter_mut().enumerate() {
            let obs = u.actual_invented[i];
            let pot = u.possible_invented[i];
            inv.2 = obs.ln() - pot.ln();
        }
        joint_mdl
    }

    fn all_uses(&self, frontiers: &[RescoredFrontier]) -> (f64, Uses) {
        let (tx, rx) = bounded(frontiers.len());
        let u = frontiers
            .par_iter()
            .flat_map(|f| {
                let lu = f
                    .1
                    .iter()
                    .map(|&(ref expr, _logprior, loglikelihood)| {
                        let (logprior, u) = self.uses(&f.0, expr);
                        (logprior + loglikelihood, u)
                    })
                    .collect::<Vec<_>>();
                let largest = lu.iter().fold(f64::NEG_INFINITY, |acc, &(l, _)| acc.max(l));
                tx.send(largest).expect("send on closed channel");
                let z = largest
                    + lu.iter()
                        .map(|&(l, _)| (l - largest).exp())
                        .sum::<f64>()
                        .ln();
                lu.into_par_iter().map(move |(l, mut u)| {
                    u.scale((l - z).exp());
                    u
                })
            })
            .reduce(
                || Uses::new(self),
                |mut u, nu| {
                    u.merge(nu);
                    u
                },
            );
        let joint_mdl = rx.take(frontiers.len()).sum();
        (joint_mdl, u)
    }

    /// This is similar to `enumerator::likelihood` but it does a lot more work to determine
    /// _outside_ counts.
    fn uses(&self, request: &TypeSchema, expr: &Expression) -> (f64, Uses) {
        let mut ctx = Context::default();
        let tp = request.clone().instantiate_owned(&mut ctx);
        let env = Rc::new(LinkedList::default());
        self.likelihood_uses(&tp, expr, &ctx, &env)
    }

    /// This is similar to `enumerator::likelihood_internal` but it does a lot more work to
    /// determine _outside_ counts.
    fn likelihood_uses<'a>(
        &self,
        request: &Type,
        expr: &Expression,
        ctx: &'a Context,
        env: &Rc<LinkedList<Type>>,
    ) -> (f64, Uses) {
        if let Some((arg, ret)) = request.as_arrow() {
            let env = LinkedList::prepend(env, arg.clone());
            if let Expression::Abstraction(ref body) = *expr {
                self.likelihood_uses(ret, body, ctx, &env)
            } else {
                (f64::NEG_INFINITY, Uses::new(self)) // invalid expression
            }
        } else {
            let candidates = self.candidates(request, ctx, &env.as_vecdeque());
            let mut possible_vars = 0f64;
            let mut possible_prims = vec![0f64; self.primitives.len()];
            let mut possible_invented = vec![0f64; self.invented.len()];
            for &(_, ref expr, _, _) in &candidates {
                match *expr {
                    Expression::Primitive(num) => possible_prims[num] = 1f64,
                    Expression::Invented(num) => possible_invented[num] = 1f64,
                    Expression::Index(_) => possible_vars = 1f64,
                    _ => unreachable!(),
                }
            }
            let mut total_likelihood = f64::NEG_INFINITY;
            let mut weighted_uses: Vec<(f64, Uses)> = Vec::new();
            let mut f = expr;
            let mut xs: VecDeque<&Expression> = VecDeque::new();
            loop {
                // if we're dealing with an Application, we reiterate for every applicable f/xs
                // combination. (see the end of this block.)
                for &(mut l, ref expr, ref tp, ref cctx) in &candidates {
                    let mut ctx = Cow::Borrowed(cctx);
                    let mut tp = Cow::Borrowed(tp);
                    let mut bindings = HashMap::new();
                    // skip this iteration if candidate expr and f don't match:
                    if let Expression::Index(_) = *expr {
                        if expr != f {
                            continue;
                        }
                    } else if let Some(mut frag_tp) =
                        TreeMatcher::do_match(self, ctx.to_mut(), expr, f, &mut bindings, xs.len())
                    {
                        let mut template = VecDeque::with_capacity(xs.len() + 1);
                        template.push_front(request.clone());
                        for _ in 0..xs.len() {
                            template.push_front(ctx.to_mut().new_variable())
                        }
                        // unification cannot fail, so we can safely unwrap:
                        if ctx
                            .to_mut()
                            .unify(&frag_tp, &Type::from(template.clone()))
                            .is_err()
                        {
                            eprintln!(
                                "WARNING (please report to programinduction devs): likelihood unification failure against expr={} (tp={}) for f={} frag_tp={} tmpl_tp={} xs={:?}",
                                self.display(expr),
                                tp,
                                self.display(f),
                                frag_tp,
                                Type::from(template),
                                xs.iter().map(|x| self.display(x)).collect::<Vec<_>>(),
                            );
                            continue;
                        }
                        frag_tp.apply_mut(&ctx);
                        tp = Cow::Owned(frag_tp);
                    } else {
                        continue;
                    }

                    let arg_tps: VecDeque<&Type> = tp.args().unwrap_or_else(VecDeque::new);
                    if xs.len() != arg_tps.len() {
                        eprintln!(
                            "WARNING (please report to programinduction devs): xs and arg_tps did not correspond: expr={} (arg_tps={:?}) f={} xs={:?}",
                            self.display(expr),
                            arg_tps.iter().map(|t| t.to_string()).collect::<Vec<_>>(),
                            self.display(f),
                            xs.iter().map(|x| self.display(x)).collect::<Vec<_>>(),
                        );
                        continue;
                    }

                    let mut u = Uses {
                        actual_vars: 0f64,
                        actual_prims: vec![0f64; self.primitives.len()],
                        actual_invented: vec![0f64; self.invented.len()],
                        possible_vars,
                        possible_prims: possible_prims.clone(),
                        possible_invented: possible_invented.clone(),
                    };
                    match *expr {
                        Expression::Primitive(num) => u.actual_prims[num] = 1f64,
                        Expression::Invented(num) => u.actual_invented[num] = 1f64,
                        Expression::Index(_) => u.actual_vars = 1f64,
                        _ => unreachable!(),
                    }

                    for (free_tp, free_expr) in bindings
                        .iter()
                        .map(|(_, &(ref tp, ref expr))| (tp, expr))
                        .chain(arg_tps.into_iter().zip(xs.iter().map(|&x| x)))
                    {
                        let mut free_tp = free_tp.clone();
                        loop {
                            let free_tp_new = free_tp.apply(&ctx);
                            if free_tp_new != free_tp {
                                free_tp = free_tp_new;
                            } else {
                                break;
                            }
                        }
                        let n = self.likelihood_uses(&free_tp, free_expr, &ctx, env);
                        if n.0.is_infinite() {
                            l = f64::NEG_INFINITY;
                            break;
                        }
                        l += n.0;
                        u.merge(n.1);
                    }

                    if l.is_infinite() {
                        continue;
                    }
                    weighted_uses.push((l, u));
                    total_likelihood = if total_likelihood > l {
                        total_likelihood + (1f64 + (l - total_likelihood).exp()).ln()
                    } else {
                        l + (1f64 + (total_likelihood - l).exp()).ln()
                    };
                }

                if let Expression::Application(ref ff, ref x) = *f {
                    f = ff;
                    xs.push_front(&*x);
                } else {
                    break;
                }
            }

            let mut u = Uses::new(self);
            if total_likelihood.is_finite() && !weighted_uses.is_empty() {
                u.join_from(total_likelihood, weighted_uses)
            }
            (total_likelihood, u)
        }
    }

    /// returns whether the frontier was rewritten
    fn rewrite_frontier_with_fragment_expression(
        &self,
        f: &mut RescoredFrontier,
        i: usize,
        expr: &Expression,
    ) -> bool {
        let results: Vec<_> = f
            .1
            .iter_mut()
            .map(|x| self.rewrite_expression(&mut x.0, i, expr, 0))
            .collect();
        results.iter().any(|&x| x)
    }
    fn rewrite_expression(
        &self,
        expr: &mut Expression,
        inv_n: usize,
        inv: &Expression,
        n_args: usize,
    ) -> bool {
        let mut rewrote = false;
        let do_rewrite = match *expr {
            Expression::Application(ref mut f, ref mut x) => {
                rewrote |= self.rewrite_expression(f, inv_n, inv, n_args + 1);
                rewrote |= self.rewrite_expression(x, inv_n, inv, 0);
                true
            }
            Expression::Abstraction(ref mut body) => {
                rewrote |= self.rewrite_expression(body, inv_n, inv, 0);
                true
            }
            _ => false,
        };
        if do_rewrite {
            let mut bindings = HashMap::new();
            let mut ctx = Context::default();
            let matches =
                TreeMatcher::do_match(self, &mut ctx, inv, expr, &mut bindings, n_args).is_some();
            if matches {
                let mut new_expr = Expression::Invented(inv_n);
                for j in (0..bindings.len()).rev() {
                    let &(_, ref b) = &bindings[&j];
                    let inner = Box::new(new_expr);
                    new_expr = Expression::Application(inner, Box::new(b.clone()));
                }
                *expr = new_expr;
                rewrote = true
            }
        }
        rewrote
    }

    /// Yields expressions that may have free variables.
    fn propose_inventions(
        &self,
        frontiers: &[RescoredFrontier],
        arity: u32,
        proposals: &mut Vec<Expression>,
    ) {
        let (tx, rx) = bounded(100);
        join(
            move || {
                let findings = Arc::new(RwLock::new(HashMap::new()));
                frontiers
                    .par_iter()
                    .flat_map(|f| &f.1)
                    .flat_map(|&(ref expr, _, _)| proposals::from_expression(expr, arity))
                    .filter(|fragment_expr| {
                        let expr = proposals::defragment(fragment_expr.clone());
                        self.invented
                            .iter()
                            .find(|&&(ref x, _, _)| x == &expr)
                            .is_none()
                    })
                    .for_each(|fragment_expr| {
                        let res = {
                            let h = findings.read().expect("hashmap was poisoned");
                            h.get(&fragment_expr)
                                .map(|x: &AtomicUsize| x.fetch_add(1, Ordering::SeqCst))
                        };
                        match res {
                            Some(2) if self.infer(&fragment_expr).is_ok() => tx
                                .send(fragment_expr)
                                .expect("failed to send fragment proposal"),
                            None => {
                                let mut h = findings.write().expect("hashmap was poisoned");
                                let count = h
                                    .entry(fragment_expr.clone())
                                    .or_insert_with(|| AtomicUsize::new(0));
                                if 2 == count.fetch_add(1, Ordering::SeqCst)
                                    && self.infer(&fragment_expr).is_ok()
                                {
                                    tx.send(fragment_expr)
                                        .expect("failed to send fragment proposal")
                                }
                            }
                            _ => (),
                        }
                    })
            },
            move || proposals.extend(rx),
        );
    }
}

struct TreeMatcher<'a> {
    dsl: &'a Language,
    ctx: &'a mut Context,
    bindings: &'a mut HashMap<usize, (Type, Expression)>,
}
impl<'a> TreeMatcher<'a> {
    /// If the trees (`fragment` against `concrete`) match, this appropriately updates the context
    /// and gets the type for `fragment`.  Also gives bindings for indices. This may modify the
    /// context even upon failure.
    fn do_match(
        dsl: &Language,
        ctx: &mut Context,
        fragment: &Expression,
        concrete: &Expression,
        bindings: &mut HashMap<usize, (Type, Expression)>,
        n_args: usize,
    ) -> Option<Type> {
        if !Self::might_match(dsl, fragment, concrete, 0) {
            None
        } else {
            let mut tm = TreeMatcher { dsl, ctx, bindings };
            tm.execute(fragment, concrete, &Rc::new(LinkedList::default()), n_args)
        }
    }

    /// Small tree comparison that doesn't update any bindings.
    fn might_match(
        dsl: &Language,
        fragment: &Expression,
        concrete: &Expression,
        depth: usize,
    ) -> bool {
        match *fragment {
            Expression::Index(i) if i >= depth => true,
            Expression::Abstraction(ref f_body) => {
                if let Expression::Abstraction(ref e_body) = *concrete {
                    Self::might_match(dsl, f_body, e_body, depth + 1)
                } else {
                    false
                }
            }
            Expression::Application(ref f_f, ref f_x) => {
                if let Expression::Application(ref c_f, ref c_x) = *concrete {
                    Self::might_match(dsl, f_x, c_x, depth)
                        && Self::might_match(dsl, f_f, c_f, depth)
                } else {
                    false
                }
            }
            Expression::Invented(f_num) => {
                if let Expression::Invented(c_num) = *concrete {
                    f_num == c_num
                } else {
                    Self::might_match(dsl, &dsl.invented[f_num].0, concrete, depth)
                }
            }
            _ => fragment == concrete,
        }
    }

    fn execute(
        &mut self,
        fragment: &Expression,
        concrete: &Expression,
        env: &Rc<LinkedList<Type>>,
        n_args: usize,
    ) -> Option<Type> {
        match (fragment, concrete) {
            (
                &Expression::Application(ref f_f, ref f_x),
                &Expression::Application(ref c_f, ref c_x),
            ) => {
                let ft = self.execute(f_f, c_f, env, n_args)?;
                let xt = self.execute(f_x, c_x, env, n_args)?;
                let ret = self.ctx.new_variable();
                if self.ctx.unify(&ft, &Type::arrow(xt, ret.clone())).is_ok() {
                    Some(ret.apply(self.ctx))
                } else {
                    None
                }
            }
            (&Expression::Primitive(f_num), &Expression::Primitive(c_num)) if f_num == c_num => {
                let tp = self.dsl.primitives[f_num].1.clone();
                Some(tp.instantiate_owned(self.ctx))
            }
            (&Expression::Invented(f_num), &Expression::Invented(c_num)) => {
                if f_num == c_num {
                    let tp = self.dsl.invented[f_num].1.clone();
                    Some(tp.instantiate_owned(self.ctx))
                } else {
                    None
                }
            }
            (&Expression::Invented(f_num), _) => {
                let inv = &self.dsl.invented[f_num].0;
                self.execute(inv, concrete, env, n_args)
            }
            (&Expression::Abstraction(ref f_body), &Expression::Abstraction(ref c_body)) => {
                let arg = self.ctx.new_variable();
                let env = LinkedList::prepend(env, arg.clone());
                let ret = self.execute(f_body, c_body, &env, 0)?;
                Some(Type::arrow(arg, ret))
            }
            (&Expression::Index(i), _) if i < env.len() => {
                // bound variable
                if fragment == concrete {
                    let mut tp = env[i].clone();
                    tp.apply_mut(self.ctx);
                    Some(tp)
                } else {
                    None
                }
            }
            (&Expression::Index(i), _) => {
                // free variable
                let i = i - env.len();
                // make sure index bindings don't reach beyond fragment
                let mut concrete = concrete.clone();
                if concrete.shift(-(env.len() as i64)) {
                    // wrap in abstracted applications for eta-long form
                    if n_args > 0 {
                        concrete.shift(n_args as i64);
                        for j in 0..n_args {
                            concrete = Expression::Application(
                                Box::new(concrete),
                                Box::new(Expression::Index(j)),
                            );
                        }
                        for _ in 0..n_args {
                            concrete = Expression::Abstraction(Box::new(concrete));
                        }
                    }
                    // update bindings
                    if let Some(&(ref tp, ref binding)) = self.bindings.get(&i) {
                        return if binding == &concrete {
                            Some(tp.clone())
                        } else {
                            None
                        };
                    }
                    let tp = self.ctx.new_variable();
                    self.bindings.insert(i, (tp.clone(), concrete));
                    Some(tp)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
struct Uses {
    actual_vars: f64,
    possible_vars: f64,
    actual_prims: Vec<f64>,
    possible_prims: Vec<f64>,
    actual_invented: Vec<f64>,
    possible_invented: Vec<f64>,
}
impl Uses {
    fn new(dsl: &Language) -> Uses {
        let n_primitives = dsl.primitives.len();
        let n_invented = dsl.invented.len();
        Uses {
            actual_vars: 0f64,
            possible_vars: 0f64,
            actual_prims: vec![0f64; n_primitives],
            possible_prims: vec![0f64; n_primitives],
            actual_invented: vec![0f64; n_invented],
            possible_invented: vec![0f64; n_invented],
        }
    }
    fn scale(&mut self, s: f64) {
        self.actual_vars *= s;
        self.possible_vars *= s;
        self.actual_prims.iter_mut().for_each(|x| *x *= s);
        self.possible_prims.iter_mut().for_each(|x| *x *= s);
        self.actual_invented.iter_mut().for_each(|x| *x *= s);
        self.possible_invented.iter_mut().for_each(|x| *x *= s);
    }
    fn merge(&mut self, other: Uses) {
        self.actual_vars += other.actual_vars;
        self.possible_vars += other.possible_vars;
        self.actual_prims
            .iter_mut()
            .zip(other.actual_prims)
            .for_each(|(a, b)| *a += b);
        self.possible_prims
            .iter_mut()
            .zip(other.possible_prims)
            .for_each(|(a, b)| *a += b);
        self.actual_invented
            .iter_mut()
            .zip(other.actual_invented)
            .for_each(|(a, b)| *a += b);
        self.possible_invented
            .iter_mut()
            .zip(other.possible_invented)
            .for_each(|(a, b)| *a += b);
    }
    /// self must be freshly created via `Uses::new()`, `z` must be finite and `weighted_uses` must
    /// be non-empty.
    fn join_from(&mut self, z: f64, mut weighted_uses: Vec<(f64, Uses)>) {
        for &mut (l, ref mut u) in &mut weighted_uses {
            u.scale((l - z).exp());
        }
        self.actual_vars = weighted_uses
            .iter()
            .map(|&(_, ref u)| u.actual_vars)
            .sum::<f64>();
        self.possible_vars = weighted_uses
            .iter()
            .map(|&(_, ref u)| u.possible_vars)
            .sum::<f64>();
        self.actual_prims.iter_mut().enumerate().for_each(|(i, c)| {
            *c = weighted_uses
                .iter()
                .map(|&(_, ref u)| u.actual_prims[i])
                .sum::<f64>()
        });
        self.possible_prims
            .iter_mut()
            .enumerate()
            .for_each(|(i, c)| {
                *c = weighted_uses
                    .iter()
                    .map(|&(_, ref u)| u.possible_prims[i])
                    .sum::<f64>()
            });
        self.actual_invented
            .iter_mut()
            .enumerate()
            .for_each(|(i, c)| {
                *c = weighted_uses
                    .iter()
                    .map(|&(_, ref u)| u.actual_invented[i])
                    .sum::<f64>()
            });
        self.possible_invented
            .iter_mut()
            .enumerate()
            .for_each(|(i, c)| {
                *c = weighted_uses
                    .iter()
                    .map(|&(_, ref u)| u.possible_invented[i])
                    .sum::<f64>()
            });
    }
}

mod proposals {
    //! Proposals, or "fragment expressions" (written `fragment_expr` where applicable) are
    //! expressions with free variables.

    use super::super::Expression;
    use super::expression_count_kinds;
    use itertools::Itertools;
    use std::collections::HashMap;
    use std::iter;

    #[derive(Clone, Debug)]
    enum Fragment {
        Variable,
        Application(Box<Fragment>, Box<Fragment>),
        Abstraction(Box<Fragment>),
        Expression(Expression),
    }
    impl Fragment {
        fn fragvars(&self) -> usize {
            match self {
                Fragment::Expression(_) => 0,
                Fragment::Application(f, x) => f.fragvars() + x.fragvars(),
                Fragment::Abstraction(body) => body.fragvars(),
                Fragment::Variable => 1,
            }
        }
        fn n_free(&self, depth: usize) -> usize {
            match self {
                Fragment::Expression(expr) => Fragment::n_free_expr(expr, depth),
                Fragment::Application(f, x) => f.n_free(depth) + x.n_free(depth),
                Fragment::Abstraction(body) => body.n_free(depth + 1),
                Fragment::Variable => 0,
            }
        }
        fn n_free_expr(expr: &Expression, depth: usize) -> usize {
            match expr {
                Expression::Application(f, x) => {
                    Fragment::n_free_expr(f, depth) + Fragment::n_free_expr(x, depth)
                }
                Expression::Abstraction(body) => Fragment::n_free_expr(body, depth + 1),
                Expression::Index(i) if *i >= depth => 1,
                _ => 0,
            }
        }
        fn canonicalize(self) -> impl Iterator<Item = Expression> {
            let fragvars = self.fragvars();
            let n_free = self.n_free(0);
            // 000 001 010 100 011 101 110 ~111~
            iter::repeat(0..fragvars)
                .take(fragvars)
                .multi_cartesian_product()
                .filter(|xs| {
                    if let Some(x) = xs.iter().max() {
                        if *x == 0 {
                            true
                        } else {
                            (0..*x).all(|y| xs.contains(&y))
                        }
                    } else {
                        true
                    }
                })
                .pad_using(1, |_| Vec::new())
                .map(move |mut assignment| {
                    for x in &mut assignment {
                        *x += n_free
                    }
                    let mut c = Canonicalizer::new(assignment);
                    let mut frag = self.clone();
                    c.canonicalize(&mut frag, 0);
                    frag.into_expression()
                })
        }
        fn into_expression(self) -> Expression {
            match self {
                Fragment::Expression(expr) => expr,
                Fragment::Application(f, x) => Expression::Application(
                    Box::new(f.into_expression()),
                    Box::new(x.into_expression()),
                ),
                Fragment::Abstraction(body) => {
                    Expression::Abstraction(Box::new(body.into_expression()))
                }
                _ => panic!("cannot convert fragment that still has variables"),
            }
        }
    }
    /// remove free variables from an expression by introducing abstractions.
    pub fn defragment(mut fragment_expr: Expression) -> Expression {
        let reach = free_reach(&fragment_expr, 0);
        for _ in 0..reach {
            let body = Box::new(fragment_expr);
            fragment_expr = Expression::Abstraction(body);
        }
        fragment_expr
    }

    struct Canonicalizer {
        assignment: Vec<usize>,
        elapsed: usize,
        free: usize,
        mapping: HashMap<usize, usize>,
    }
    impl Canonicalizer {
        fn new(assignment: Vec<usize>) -> Canonicalizer {
            Canonicalizer {
                assignment,
                elapsed: 0,
                free: 0,
                mapping: HashMap::default(),
            }
        }
        fn canonicalize(&mut self, fr: &mut Fragment, depth: usize) {
            match *fr {
                Fragment::Expression(ref mut expr) => self.canonicalize_expr(expr, depth),
                Fragment::Application(ref mut f, ref mut x) => {
                    self.canonicalize(f, depth);
                    self.canonicalize(x, depth);
                }
                Fragment::Abstraction(ref mut body) => {
                    self.canonicalize(body, depth + 1);
                }
                Fragment::Variable => {
                    *fr = Fragment::Expression(Expression::Index(
                        self.assignment[self.elapsed] + depth,
                    ));
                    self.elapsed += 1;
                }
            }
        }
        fn canonicalize_expr(&mut self, expr: &mut Expression, depth: usize) {
            match *expr {
                Expression::Application(ref mut f, ref mut x) => {
                    self.canonicalize_expr(f, depth);
                    self.canonicalize_expr(x, depth);
                }
                Expression::Abstraction(ref mut body) => self.canonicalize_expr(body, depth + 1),
                Expression::Index(ref mut i) if *i >= depth => {
                    let j = i.checked_sub(depth).unwrap();
                    if let Some(k) = self.mapping.get(&j) {
                        *i = k + depth;
                        return;
                    }
                    self.mapping.insert(j, self.free);
                    *i = self.free + depth;
                    self.free += 1;
                }
                _ => (),
            }
        }
    }

    /// main entry point for proposals
    pub fn from_expression(expr: &Expression, arity: u32) -> Vec<Expression> {
        (0..arity + 1)
            .flat_map(move |b| from_subexpression(expr, b))
            .flat_map(Fragment::canonicalize)
            .filter(|fragment_expr| {
                // determine if nontrivial
                let (n_prims, n_free, n_bound) = expression_count_kinds(fragment_expr, 0);
                n_prims >= 1 && ((n_prims as f64) + 0.5 * ((n_free + n_bound) as f64) > 1.5)
            })
            .flat_map(to_inventions)
            .collect()
    }
    fn from_subexpression<'a>(
        expr: &'a Expression,
        arity: u32,
    ) -> impl Iterator<Item = Fragment> + 'a {
        let rst: Box<Iterator<Item = Fragment>> = match *expr {
            Expression::Application(ref f, ref x) => {
                Box::new(from_subexpression(f, arity).chain(from_subexpression(x, arity)))
            }
            Expression::Abstraction(ref body) => Box::new(from_subexpression(body, arity)),
            _ => Box::new(iter::empty()),
        };
        from_particular(expr, arity, true).chain(rst)
    }
    fn from_particular<'a>(
        expr: &'a Expression,
        arity: u32,
        toplevel: bool,
    ) -> Box<Iterator<Item = Fragment> + 'a> {
        if arity == 0 {
            return Box::new(iter::once(Fragment::Expression(expr.clone())));
        }
        let rst: Box<Iterator<Item = Fragment> + 'a> = match *expr {
            Expression::Application(ref f, ref x) => Box::new((0..arity + 1).flat_map(move |fa| {
                let xa = (arity as i32 - fa as i32) as u32;
                from_particular(f, fa, false)
                    .zip(iter::repeat(
                        from_particular(x, xa, false).collect::<Vec<_>>(),
                    ))
                    .flat_map(|(f, xs)| {
                        xs.into_iter()
                            .map(move |x| Fragment::Application(Box::new(f.clone()), Box::new(x)))
                    })
            })),
            Expression::Abstraction(ref body) if !toplevel => Box::new(
                from_particular(body, arity, false).map(|e| Fragment::Abstraction(Box::new(e))),
            ),
            _ => Box::new(iter::empty()),
        };
        Box::new(iter::once(Fragment::Variable).chain(rst))
    }
    fn to_inventions(expr: Expression) -> impl Iterator<Item = Expression> {
        // for any common subtree within the expression, replace with new index.
        let reach = free_reach(&expr, 0);
        let mut counts = HashMap::new();
        subtrees(expr.clone(), &mut counts);
        counts.remove(&expr);
        let fst = iter::once(expr.clone());
        let rst = counts
            .into_iter()
            .filter(|&(_, count)| count >= 2)
            .filter(|&(ref expr, _)| is_closed(expr))
            .map(move |(subtree, _)| {
                let mut expr = expr.clone();
                substitute(&mut expr, &subtree, &Expression::Index(reach));
                expr
            });
        fst.chain(rst)
    }

    /// How far out does the furthest reaching index go, excluding internal abstractions?
    ///
    /// For examples, the free reach of `(+ $0 (λ + 1 $0))` is 1, because there need to be one
    /// abstraction around the expression for it to make sense.
    fn free_reach(expr: &Expression, depth: usize) -> usize {
        match *expr {
            Expression::Application(ref f, ref x) => free_reach(f, depth).max(free_reach(x, depth)),
            Expression::Abstraction(ref body) => free_reach(body, depth + 1),
            Expression::Index(i) if i >= depth => 1 + i.checked_sub(depth).unwrap(),
            _ => 0,
        }
    }

    /// Counts occurrences for every subtree of expr.
    fn subtrees(expr: Expression, counts: &mut HashMap<Expression, usize>) {
        match expr.clone() {
            Expression::Application(f, x) => {
                subtrees(*f, counts);
                subtrees(*x, counts);
                counts.entry(expr).or_insert(0);
            }
            Expression::Abstraction(body) => {
                subtrees(*body, counts);
                counts.entry(expr).or_insert(0);
            }
            Expression::Index(_) => (),
            Expression::Primitive(num) => {
                counts.entry(Expression::Primitive(num)).or_insert(0);
            }
            Expression::Invented(num) => {
                counts.entry(Expression::Invented(num)).or_insert(0);
            }
        }
    }

    /// Whether every `Expression::Index` is bound within expr.
    fn is_closed(expr: &Expression) -> bool {
        free_reach(expr, 0) == 0
    }

    /// Replace all occurrences of subtree in expr with replacement.
    fn substitute(expr: &mut Expression, subtree: &Expression, replacement: &Expression) {
        if expr == subtree {
            *expr = replacement.clone()
        } else {
            match *expr {
                Expression::Application(ref mut f, ref mut x) => {
                    substitute(f, subtree, replacement);
                    substitute(x, subtree, replacement);
                }
                Expression::Abstraction(ref mut body) => substitute(body, subtree, replacement),
                _ => (),
            }
        }
    }
}

/// Counts of prims, free, bound
pub fn expression_count_kinds(expr: &Expression, abstraction_depth: usize) -> (u64, u64, u64) {
    match *expr {
        Expression::Primitive(_) | Expression::Invented(_) => (1, 0, 0),
        Expression::Index(i) => {
            if i < abstraction_depth {
                (0, 0, 1)
            } else {
                (0, 1, 0)
            }
        }
        Expression::Abstraction(ref b) => expression_count_kinds(b, abstraction_depth + 1),
        Expression::Application(ref l, ref r) => {
            let (l1, f1, b1) = expression_count_kinds(l, abstraction_depth);
            let (l2, f2, b2) = expression_count_kinds(r, abstraction_depth);
            (l1 + l2, f1 + f2, b1 + b2)
        }
    }
}
