use std::collections::{HashMap, VecDeque};
use std::f64;
use std::iter;
use std::rc::Rc;
use itertools::Itertools;
use polytype::{Context, Type};
use rayon::prelude::*;
use super::{Expression, Language, LinkedList};
use super::super::{Frontier, Task};

const BOUND_VAR_COST: f64 = 0.1;
const FREE_VAR_COST: f64 = 0.01;

/// Parameters for grammar induction.
///
/// Proposed grammars are scored as `likelihood - aic * #primitives - structure_penalty * #nodes`.
/// Additionally, `pseudocounts` affects the likelihood calculation, and `topk` and `arity` affect
/// what fragments can be proposed.
pub struct Params {
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
    /// AIC is a penalty in the number of parameters, i.e. the number of primitives and invented
    /// expressions.
    pub aic: f64,
    /// Arity is the largest applicative depth of an expression that may be manipulated to propose
    /// a fragment.
    pub arity: u32,
}
impl Default for Params {
    /// The default params prevent completely discarding of primives by having non-zero
    /// pseudocounts.
    ///
    /// ```
    /// # use programinduction::lambda::Params;
    /// Params {
    ///     pseudocounts: 1,
    ///     topk: 10,
    ///     structure_penalty: 0f64,
    ///     aic: 0f64,
    ///     arity: 3,
    /// }
    /// # ;
    /// ```
    fn default() -> Self {
        Params {
            pseudocounts: 1,
            topk: 10,
            structure_penalty: 0f64,
            aic: 0f64,
            arity: 3,
        }
    }
}

pub fn induce<O: Sync>(
    dsl: &Language,
    params: &Params,
    tasks: &[Task<Language, O>],
    frontiers: &[Frontier<Language>],
) -> Language {
    let mut g = FragmentGrammar::from(dsl);
    let mut frontiers: Vec<_> = tasks
        .iter()
        .map(|t| &t.tp)
        .zip(frontiers)
        .filter(|&(_, f)| !f.is_empty())
        .map(|(tp, fs)| {
            let fs: Vec<_> = fs.0
                .iter()
                .map(|&(ref expr, lp, ll)| (expr.clone(), lp, ll))
                .collect();
            RescoredFrontier(tp, fs)
        })
        .collect();

    let old_joint = g.joint_mdl(&frontiers, false);
    let mut best_score = g.score(
        &frontiers,
        params.pseudocounts,
        params.aic,
        params.structure_penalty,
    );

    if params.aic.is_finite() {
        loop {
            {
                eprintln!(
                    "grammar induction: attemping proposals against score {}",
                    best_score
                );
                let rescored_frontiers: Vec<_> = frontiers
                    .par_iter()
                    .map(|f| g.rescore_frontier(f, params.topk))
                    .collect();
                let best_proposal = propose_inventions(&rescored_frontiers, params.arity)
                    .zip(iter::repeat(g.clone()))
                    .filter_map(|(inv, mut g)| {
                        g.dsl.invent(inv.clone(), 0f64).unwrap();
                        let s = g.score(
                            &rescored_frontiers,
                            params.pseudocounts,
                            params.aic,
                            params.structure_penalty,
                        );
                        eprintln!(
                            "grammar induction: proposed {} with score {}",
                            g.dsl.stringify(&inv),
                            s
                        );
                        if s.is_finite() {
                            Some((g, s))
                        } else {
                            None
                        }
                    })
                    .max_by(|&(_, ref x), &(_, ref y)| x.partial_cmp(y).unwrap());
                if best_proposal.is_none() {
                    eprintln!("grammar induction: no best proposal");
                    break;
                }
                let (new_grammar, new_score) = best_proposal.unwrap();
                if new_score <= best_score {
                    eprintln!("grammar induction: score did not improve");
                    break;
                }
                g = new_grammar;
                best_score = new_score;
                let &(ref inv, ref tp, _) = g.dsl.invented.last().unwrap();
                eprintln!(
                    "grammar induction: invented with tp {} and improved score to {}: {}",
                    tp,
                    best_score,
                    dsl.stringify(inv)
                );
            }
            frontiers = frontiers
                .into_par_iter()
                .map(|mut f| {
                    g.rewrite_frontier_with_latest_invention(&mut f);
                    f
                })
                .collect();
        }
    }

    g.inside_outside(&frontiers, params.pseudocounts);
    let new_joint = g.joint_mdl(&frontiers, false);
    eprintln!(
        "grammar induction: old joint = {} ; new joint = {}",
        old_joint, new_joint
    );
    g.dsl // TODO: consider also returning new frontiers
}

struct RescoredFrontier<'a>(&'a Type, Vec<(Expression, f64, f64)>);

#[derive(Debug, Clone)]
struct FragmentGrammar {
    dsl: Language,
}
impl FragmentGrammar {
    fn rescore_frontier<'a>(&self, f: &'a RescoredFrontier, topk: usize) -> RescoredFrontier<'a> {
        let xs = f.1
            .iter()
            .map(|&(ref expr, _, loglikelihood)| {
                let logprior = self.uses(f.0, expr).0;
                (expr, logprior, loglikelihood)
            })
            .sorted_by(|&(_, _, ref x), &(_, _, ref y)| y.partial_cmp(x).unwrap())
            .into_iter()
            .take(topk)
            .map(|(expr, logprior, loglikelihood)| (expr.clone(), logprior, loglikelihood))
            .collect();
        RescoredFrontier(f.0, xs)
    }

    fn joint_mdl(&self, frontiers: &[RescoredFrontier], recompute_prior: bool) -> f64 {
        frontiers
            .par_iter()
            .map(|f| {
                f.1
                    .iter()
                    .map(|s| {
                        let loglikelihood = s.2;
                        let logprior = if recompute_prior {
                            self.dsl.likelihood(f.0, &s.0)
                        } else {
                            s.1
                        };
                        logprior + loglikelihood
                    })
                    .fold(f64::NEG_INFINITY, |acc, logposterior| acc.max(logposterior))
            })
            .sum()
    }

    fn score(
        &mut self,
        frontiers: &[RescoredFrontier],
        pseudocounts: u64,
        aic: f64,
        structure_penalty: f64,
    ) -> f64 {
        self.inside_outside(frontiers, pseudocounts);
        let likelihood = self.joint_mdl(frontiers, true);
        let structure = (self.dsl.primitives.len() as f64)
            + self.dsl
                .invented
                .iter()
                .map(|&(ref expr, _, _)| expression_structure(expr))
                .sum::<f64>();
        let nparams = self.dsl.primitives.len() + self.dsl.invented.len();
        likelihood - aic * (nparams as f64) - structure_penalty * structure
    }

    fn inside_outside(&mut self, frontiers: &[RescoredFrontier], pseudocounts: u64) {
        let pseudocounts = pseudocounts as f64;
        let u = self.all_uses(frontiers);
        self.dsl.variable_logprob =
            ((u.actual_vars + pseudocounts) as f64).ln() - 1f64.max(u.possible_vars as f64).ln();
        for (i, prim) in self.dsl.primitives.iter_mut().enumerate() {
            let obs = u.actual_prims[i] + pseudocounts;
            let pot = if u.possible_prims[i] != 0f64 {
                u.possible_prims[i]
            } else {
                pseudocounts
            };
            prim.2 = obs.ln() - pot.ln();
        }
        for (i, inv) in self.dsl.invented.iter_mut().enumerate() {
            let obs = u.actual_invented[i] + pseudocounts;
            let pot = if u.possible_invented[i] != 0f64 {
                u.possible_invented[i]
            } else {
                pseudocounts
            };
            inv.2 = obs.ln() - pot.ln();
        }
    }

    fn all_uses(&self, frontiers: &[RescoredFrontier]) -> Uses {
        let lus: Vec<_> = frontiers
            .par_iter()
            .map(|f| {
                f.1
                    .iter()
                    .map(|&(ref expr, _, loglikelihood)| {
                        let (l, u) = self.uses(f.0, expr);
                        (l + loglikelihood, u)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let zs: Vec<_> = lus.iter()
            .map(|lu_f| {
                let largest = lu_f.iter()
                    .fold(f64::NEG_INFINITY, |acc, &(l, _)| acc.max(l));
                largest
                    + lu_f.iter()
                        .map(|&(l, _)| (l - largest).exp())
                        .sum::<f64>()
                        .ln()
            })
            .collect();
        let mut u = Uses::new(&self.dsl);
        lus.into_iter()
            .zip(zs)
            .flat_map(|(lu_f, z)| {
                lu_f.into_iter().map(move |(l, mut u)| {
                    u.scale((l - z).exp());
                    u
                })
            })
            .for_each(|ou| u.merge(ou));
        u
    }

    /// This is similar to `enumerator::likelihood` but it does a lot more work to determine
    /// _outside_ counts.
    fn uses(&self, request: &Type, expr: &Expression) -> (f64, Uses) {
        let ctx = Context::default();
        let env = Rc::new(LinkedList::default());
        let (l, _, u) = self.likelihood(request, expr, &ctx, &env);
        (l, u)
    }

    /// This is similar to `enumerator::likelihood_internal` but it does a lot more work to
    /// determine _outside_ counts.
    fn likelihood(
        &self,
        request: &Type,
        expr: &Expression,
        ctx: &Context,
        env: &Rc<LinkedList<Type>>,
    ) -> (f64, Context, Uses) {
        if let Type::Arrow(ref arrow) = *request {
            let env = LinkedList::prepend(env, *arrow.arg.clone());
            if let Expression::Abstraction(ref body) = *expr {
                self.likelihood(&*arrow.ret, body, ctx, &env)
            } else {
                eprintln!("likelihood for arrow found expression that wasn't abstraction");
                (f64::NEG_INFINITY, ctx.clone(), Uses::new(&self.dsl)) // invalid expression
            }
        } else {
            let candidates = self.dsl.candidates(request, ctx, &env.as_vecdeque());
            let mut ctx = ctx.clone();
            let mut possible_vars = 0f64;
            let mut possible_prims = vec![0f64; self.dsl.primitives.len()];
            let mut possible_invented = vec![0f64; self.dsl.invented.len()];
            for &(_, ref expr, _, _) in &candidates {
                match *expr {
                    Expression::Primitive(num) => possible_prims[num] = 1f64,
                    Expression::Invented(num) => possible_invented[num] = 1f64,
                    Expression::Index(_) => possible_vars += 1f64,
                    _ => unreachable!(),
                }
            }
            let mut total_likelihood = f64::NEG_INFINITY;
            let mut weighted_uses: Vec<(f64, Uses)> = Vec::new();
            let mut f = expr;
            let mut xs = VecDeque::new();
            loop {
                // if we're dealing with an Application, we reiterate for every applicable f/xs
                // combination.
                for &(mut l, ref expr, ref tp, ref cctx) in &candidates {
                    ctx = cctx.clone();
                    let mut tp = tp.clone();
                    let mut bindings = HashMap::new();
                    // skip this iteration if candidate expr and f don't match:
                    if let Expression::Index(_) = *expr {
                        if expr != f {
                            continue;
                        }
                    } else if let Some(frag_tp) =
                        tree_match(&self.dsl, &mut ctx, expr, f, &mut bindings, xs.len())
                    {
                        let mut template: VecDeque<Type> =
                            (0..xs.len()).map(|_| ctx.new_variable()).collect();
                        template.push_back(request.clone());
                        // unification cannot fail, so we can safely unwrap:
                        ctx.unify(&frag_tp, &Type::from(template)).unwrap();
                        tp = frag_tp.apply(&ctx);
                    } else {
                        continue;
                    }

                    let arg_tps: VecDeque<Type> = if let Type::Arrow(f_tp) = tp {
                        f_tp.args().into_iter().cloned().collect()
                    } else {
                        VecDeque::new()
                    };
                    if xs.len() != arg_tps.len() {
                        eprintln!(
                            "warning: likelihood calculation xs.len() ({}) ≠ arg_tps.len() ({})",
                            xs.len(),
                            arg_tps.len()
                        );
                        continue;
                    }

                    let mut u = Uses {
                        actual_vars: 0f64,
                        actual_prims: vec![0f64; self.dsl.primitives.len()],
                        actual_invented: vec![0f64; self.dsl.invented.len()],
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
                        .chain(arg_tps.iter().zip(&xs))
                    {
                        let free_tp = free_tp.apply(&ctx);
                        let n = self.likelihood(&free_tp, free_expr, &ctx, env);
                        if n.0.is_infinite() {
                            l = f64::NEG_INFINITY;
                            break;
                        }
                        l += n.0;
                        ctx = n.1; // ctx should become any of the new ones
                        u.merge(n.2);
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
                    xs.push_front(*x.clone());
                } else {
                    break;
                }
            }

            let mut u = Uses::new(&self.dsl);
            if total_likelihood.is_finite() && !weighted_uses.is_empty() {
                u.join_from(total_likelihood, weighted_uses)
            }
            (total_likelihood, ctx, u)
        }
    }

    fn rewrite_frontier_with_latest_invention(&self, f: &mut RescoredFrontier) {
        let inv = &self.dsl.invented.last().unwrap().0;
        f.1
            .iter_mut()
            .for_each(|x| self.rewrite_expression(&mut x.0, inv, 0));
    }
    fn rewrite_expression(&self, expr: &mut Expression, inv: &Expression, n_args: usize) {
        let do_rewrite = match *expr {
            Expression::Application(ref mut f, ref mut x) => {
                self.rewrite_expression(f, inv, n_args + 1);
                self.rewrite_expression(x, inv, 0);
                true
            }
            Expression::Abstraction(ref mut body) => {
                self.rewrite_expression(body, inv, 0);
                true
            }
            _ => false,
        };
        if do_rewrite {
            let mut bindings = HashMap::new();
            let mut ctx = Context::default();
            let matches =
                tree_match(&self.dsl, &mut ctx, inv, expr, &mut bindings, n_args).is_some();
            if matches {
                assert_eq!(
                    bindings.keys().cloned().collect::<Vec<_>>(),
                    (0..bindings.len()).collect::<Vec<_>>(),
                    "fragment must not have been in canonical form"
                );
                for j in (0..bindings.len()).rev() {
                    let &(_, ref b) = &bindings[&j];
                    *expr = Expression::Application(Box::new(expr.clone()), Box::new(b.clone()));
                }
            }
        }
    }
}
impl<'a> From<&'a Language> for FragmentGrammar {
    fn from(dsl: &'a Language) -> Self {
        let dsl = dsl.clone();
        FragmentGrammar { dsl }
    }
}

fn expression_structure(expr: &Expression) -> f64 {
    let (leaves, free, bound) = expression_count_kinds(expr, 0);
    (leaves as f64) + BOUND_VAR_COST * (bound as f64) + FREE_VAR_COST * (free as f64)
}

fn expression_count_kinds(expr: &Expression, abstraction_depth: usize) -> (u64, u64, u64) {
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

/// If the trees match, this appropriately updates the context and gets the new fragment type.
/// Also gives bindings for indices. This may modify the context even upon failure.
fn tree_match(
    dsl: &Language,
    ctx: &mut Context,
    fragment: &Expression,
    expr: &Expression,
    bindings: &mut HashMap<usize, (Type, Expression)>,
    n_args: usize,
) -> Option<Type> {
    if !tree_might_match(fragment, expr, 0) {
        None
    } else {
        let env = Rc::new(LinkedList::default());
        if let Some(tp) = {
            let mut tm = TreeMatcher { dsl, ctx, bindings };
            tm.do_match(fragment, expr, &env, n_args)
        } {
            Some(tp)
        } else {
            None
        }
    }
}

fn tree_might_match(f: &Expression, e: &Expression, depth: usize) -> bool {
    match *f {
        Expression::Primitive(_) | Expression::Invented(_) => f == e,
        Expression::Index(i) => if i < depth {
            f == e
        } else {
            true
        },
        Expression::Abstraction(ref f_body) => {
            if let Expression::Abstraction(ref e_body) = *e {
                tree_might_match(f_body, e_body, depth + 1)
            } else {
                false
            }
        }
        Expression::Application(ref f_f, ref f_x) => {
            if let Expression::Application(ref e_f, ref e_x) = *e {
                tree_might_match(f_x, e_x, depth) && tree_might_match(f_f, e_f, depth)
            } else {
                false
            }
        }
    }
}

struct TreeMatcher<'a> {
    dsl: &'a Language,
    ctx: &'a mut Context,
    bindings: &'a mut HashMap<usize, (Type, Expression)>,
}
impl<'a> TreeMatcher<'a> {
    fn do_match(
        &mut self,
        fragment: &Expression,
        expr: &Expression,
        env: &Rc<LinkedList<Type>>,
        n_args: usize,
    ) -> Option<Type> {
        match (fragment, expr) {
            (
                &Expression::Application(ref f_f, ref f_x),
                &Expression::Application(ref e_f, ref e_x),
            ) => {
                let ft = self.do_match(f_f, e_f, env, n_args)?;
                let xt = self.do_match(f_x, e_x, env, n_args)?;
                let ret = self.ctx.new_variable();
                if self.ctx.unify(&ft, &arrow![xt, ret.clone()]).is_ok() {
                    Some(ret.apply(self.ctx))
                } else {
                    None
                }
            }
            (&Expression::Primitive(f_num), &Expression::Primitive(e_num)) => {
                if f_num == e_num {
                    let tp = self.dsl.primitive(f_num).unwrap().1;
                    Some(tp.instantiate_indep(self.ctx))
                } else {
                    None
                }
            }
            (&Expression::Invented(f_num), &Expression::Invented(e_num)) => {
                if f_num == e_num {
                    let tp = self.dsl.invented(f_num).unwrap().1;
                    Some(tp.instantiate_indep(self.ctx))
                } else {
                    None
                }
            }
            (&Expression::Abstraction(ref f_body), &Expression::Abstraction(ref e_body)) => {
                let arg = self.ctx.new_variable();
                let env = LinkedList::prepend(env, arg.clone());
                let ret = self.do_match(f_body, e_body, &env, 0)?;
                Some(arrow![arg, ret])
            }
            (&Expression::Index(i), _) => {
                let abstraction_depth = env.len();
                if i < abstraction_depth {
                    // bound variable
                    if fragment == expr {
                        Some(env[i].apply(self.ctx))
                    } else {
                        None
                    }
                } else {
                    // free variable
                    let i = i - abstraction_depth;
                    // make sure index bindings don't reach beyond fragment
                    let mut expr = expr.clone();
                    if expr.shift(-(abstraction_depth as i64)) {
                        // wrap in abstracted applications for eta-long form
                        if n_args > 0 {
                            expr.shift(n_args as i64);
                            for j in 0..n_args {
                                expr = Expression::Application(
                                    Box::new(expr),
                                    Box::new(Expression::Index(j)),
                                );
                            }
                            for _ in 0..n_args {
                                expr = Expression::Abstraction(Box::new(expr));
                            }
                        }
                        // update bindings
                        if let Some(&(ref tp, ref binding)) = self.bindings.get(&i) {
                            return if binding == &expr {
                                Some(tp.clone())
                            } else {
                                None
                            };
                        }
                        let tp = self.ctx.new_variable();
                        self.bindings.insert(i, (tp.clone(), expr));
                        Some(tp)
                    } else {
                        None
                    }
                }
            }
            _ => None,
        }
    }
}

#[derive(Debug)]
enum Fragment {
    Variable,
    Application(Box<Fragment>, Box<Fragment>),
    Abstraction(Box<Fragment>),
    Expression(Expression),
}
impl Fragment {
    fn canonicalize(mut self) -> Expression {
        let mut c = Canonicalizer::default();
        c.canonicalize(&mut self, 0);
        self.to_expression()
    }
    fn to_expression(self) -> Expression {
        match self {
            Fragment::Expression(expr) => expr,
            Fragment::Application(f, x) => {
                Expression::Application(Box::new(f.to_expression()), Box::new(x.to_expression()))
            }
            Fragment::Abstraction(body) => Expression::Abstraction(Box::new(body.to_expression())),
            _ => panic!("cannot convert fragment that still has variables"),
        }
    }
}

#[derive(Default)]
struct Canonicalizer {
    n_abstractions: usize,
    mapping: HashMap<usize, usize>,
}
impl Canonicalizer {
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
                *fr = Fragment::Expression(Expression::Index(self.n_abstractions + depth));
                self.n_abstractions += 1;
            }
        }
    }
    fn canonicalize_expr(&mut self, expr: &mut Expression, depth: usize) {
        match *expr {
            Expression::Application(ref mut f, ref mut x) => {
                self.canonicalize_expr(f, depth);
                self.canonicalize_expr(x, depth);
            }
            Expression::Abstraction(ref mut body) => {
                self.canonicalize_expr(body, depth + 1);
            }
            Expression::Index(ref mut i) if *i >= depth => {
                let j = i.checked_sub(depth).unwrap();
                if let Some(k) = self.mapping.get(&j) {
                    *i = k + depth;
                    return;
                }
                self.mapping.insert(j, self.n_abstractions);
                *i = self.n_abstractions + depth;
                self.n_abstractions += 1;
            }
            _ => (),
        }
    }
}

fn propose_inventions(
    frontiers: &[RescoredFrontier],
    arity: u32,
) -> Box<Iterator<Item = Expression>> {
    let mut findings = HashMap::new();
    frontiers
        .iter() // TODO figure out how to properly parallelize
        .flat_map(|f| {
            f.1.iter().flat_map(|&(ref expr, _, _)| {
                propose_from_expression(expr, arity)
                    .flat_map(propose_inventions_from_proposal)
            })
        })
        .for_each(|inv| {
            let count = findings
                .entry(inv)
                .or_insert(0u64);
            *count += 1;
        });
    Box::new(findings.into_iter().filter_map(
        |(expr, count)| {
            if count >= 2 {
                Some(expr)
            } else {
                None
            }
        },
    ))
}
fn propose_from_expression<'a>(
    expr: &'a Expression,
    arity: u32,
) -> Box<Iterator<Item = Expression> + 'a> {
    Box::new(
        (0..arity + 1)
            .flat_map(move |b| propose_fragments_from_subexpression(expr, b))
            .map(Fragment::canonicalize),
    )
}
fn propose_fragments_from_subexpression<'a>(
    expr: &'a Expression,
    arity: u32,
) -> Box<Iterator<Item = Fragment> + 'a> {
    let rst: Box<Iterator<Item = Fragment>> = match *expr {
        Expression::Application(ref f, ref x) => Box::new(
            propose_fragments_from_subexpression(f, arity)
                .chain(propose_fragments_from_subexpression(x, arity)),
        ),
        Expression::Abstraction(ref body) => {
            Box::new(propose_fragments_from_subexpression(body, arity))
        }
        _ => Box::new(iter::empty()),
    };
    Box::new(propose_fragments_from_particular(expr, arity).chain(rst))
}
fn propose_fragments_from_particular<'a>(
    expr: &'a Expression,
    arity: u32,
) -> Box<Iterator<Item = Fragment> + 'a> {
    if arity == 0 {
        return Box::new(iter::once(Fragment::Expression(expr.clone())));
    }
    let rst: Box<Iterator<Item = Fragment> + 'a> = match *expr {
        Expression::Application(ref f, ref x) => Box::new((0..arity + 1).flat_map(move |fa| {
            let fs = propose_fragments_from_particular(f, fa);
            let xs = propose_fragments_from_particular(x, (arity as i32 - fa as i32) as u32);
            fs.zip(xs)
                .map(|(f, x)| Fragment::Application(Box::new(f), Box::new(x)))
        })),
        Expression::Abstraction(ref body) => Box::new(
            propose_fragments_from_particular(body, arity)
                .map(|e| Fragment::Abstraction(Box::new(e))),
        ),
        _ => Box::new(iter::empty()),
    };
    if arity == 1 {
        Box::new(iter::once(Fragment::Variable).chain(rst))
    } else {
        rst
    }
}
fn propose_inventions_from_proposal(expr: Expression) -> Box<Iterator<Item = Expression>> {
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
            substitute(&mut expr, &subtree, &Expression::Index(reach + 1));
            expr
        });
    Box::new(fst.chain(rst))
}

/// How far out does the furthest reaching index go, excluding internal abstractions?
fn free_reach(expr: &Expression, depth: usize) -> usize {
    match *expr {
        Expression::Application(ref f, ref x) => free_reach(f, depth).max(free_reach(x, depth)),
        Expression::Abstraction(ref body) => free_reach(body, depth + 1),
        Expression::Index(i) if i >= depth => i.checked_sub(depth).unwrap(),
        _ => 0,
    }
}

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

fn is_closed(expr: &Expression) -> bool {
    free_reach(expr, 0) == 0
}

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

#[derive(Clone)]
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
