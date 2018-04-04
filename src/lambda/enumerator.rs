use polytype::{Context, Type, TypeSchema};
use std::collections::VecDeque;
use std::f64;
use std::rc::Rc;

use super::{Expression, Language, LinkedList};

const BUDGET_INCREMENT: f64 = 1.0;
const MAX_DEPTH: u32 = 8192;

fn budget_interval(n: u32) -> (f64, f64) {
    let offset = BUDGET_INCREMENT * f64::from(n);
    (offset, offset + BUDGET_INCREMENT)
}

pub fn run<F>(dsl: &Language, request: TypeSchema, mut termination_condition: F)
where
    F: FnMut(Expression, f64) -> bool,
{
    let mut ctx = Context::default();
    let tp = request.instantiate_owned(&mut ctx);
    let env = Rc::new(LinkedList::default());
    let cb = &mut |expr, logprior, _| !termination_condition(expr, logprior);
    (0..)
        .map(budget_interval)
        .all(|budget| enumerate(dsl, &ctx, &tp, &env, budget, 0, cb));
}

pub fn likelihood<'a>(dsl: &'a Language, request: &TypeSchema, expr: &Expression) -> f64 {
    let mut ctx = Context::default();
    let env = Rc::new(LinkedList::default());
    let t = request.clone().instantiate_owned(&mut ctx);
    likelihood_internal(dsl, &t, &ctx, &env, expr).0
}
fn likelihood_internal<'a>(
    dsl: &'a Language,
    request: &Type,
    ctx: &Context,
    env: &Rc<LinkedList<Type>>,
    mut expr: &Expression,
) -> (f64, Context) {
    if let Some((arg, ret)) = request.as_arrow() {
        let env = LinkedList::prepend(env, arg.clone());
        if let Expression::Abstraction(ref body) = *expr {
            likelihood_internal(dsl, ret, ctx, &env, body)
        } else {
            (f64::NEG_INFINITY, ctx.clone()) // invalid expression
        }
    } else {
        let mut xs: Vec<&Expression> = vec![];
        while let Expression::Application(ref l, ref r) = *expr {
            expr = l;
            xs.push(r);
        }
        xs.reverse();
        match dsl.candidates(request, ctx, &env.as_vecdeque())
            .into_iter()
            .find(|&(_, ref c_expr, _, _)| expr == c_expr)
        {
            Some((f_l, _, f_tp, ctx)) => {
                if let Some(arg_tps) = f_tp.args() {
                    xs.into_iter()
                        .zip(arg_tps)
                        .fold((f_l, ctx), |(l, ctx), (x, x_tp)| {
                            let (x_l, ctx) = likelihood_internal(dsl, x_tp, &ctx, env, x);
                            (l + x_l, ctx)
                        })
                } else {
                    (f_l, ctx)
                }
            }
            None => {
                let s = dsl.display(expr);
                panic!(
                    "expression {} (with type {}) is not in candidates for request type {}",
                    s,
                    dsl.infer(expr)
                        .expect(&format!("could not infer type for {}", s)),
                    request,
                );
            }
        }
    }
}

/// returns whether the caller should continue enumerating (i.e. whether the termination condition
/// from `cb` has been met)
fn enumerate(
    dsl: &Language,
    ctx: &Context,
    request: &Type,
    env: &Rc<LinkedList<Type>>,
    budget: (f64, f64),
    depth: u32,
    cb: &mut FnMut(Expression, f64, Context) -> bool,
) -> bool {
    if budget.1 <= 0f64 || depth > MAX_DEPTH {
        true
    } else if let Some((arg, ret)) = request.as_arrow() {
        let env = LinkedList::prepend(env, arg.clone());
        let cb = &mut |body, ll, ctx| cb(Expression::Abstraction(Box::new(body)), ll, ctx);
        enumerate(dsl, ctx, ret, &env, budget, depth, cb)
    } else {
        dsl.candidates(request, ctx, &env.as_vecdeque())
            .into_iter()
            .filter(|&(ll, _, _, _)| -ll <= budget.1)
            .all(|(p, expr, tp, ctx)| {
                let arg_tps: VecDeque<Type> = tp.args()
                    .map(|args| args.into_iter().cloned().collect())
                    .unwrap_or_else(VecDeque::new);
                let budget = (budget.0 + p, budget.1 + p);
                let depth = depth + 1;
                let idx = (0, &expr);
                let f = expr.clone();
                enumerate_many(dsl, &ctx, env, f, idx, arg_tps, budget, p, depth, cb)
            })
    }
}

#[cfg_attr(feature = "cargo-clippy", allow(too_many_arguments))]
fn enumerate_many(
    dsl: &Language,
    ctx: &Context,
    env: &Rc<LinkedList<Type>>,
    f: Expression,
    idx: (usize, &Expression),
    mut arg_tps: VecDeque<Type>,
    budget: (f64, f64),
    offset: f64,
    depth: u32,
    cb: &mut FnMut(Expression, f64, Context) -> bool,
) -> bool {
    if budget.1 <= 0f64 {
        true
    } else if let Some(mut arg_tp) = arg_tps.pop_front() {
        arg_tp.apply_mut(ctx);
        let cb_arg = &mut |arg, ll, ctx| {
            if dsl.violates_symmetry(idx.1, idx.0, &arg) {
                return true;
            }
            let idx = (idx.0 + 1, idx.1);
            let f = Expression::Application(Box::new(f.clone()), Box::new(arg));
            let arg_tps = arg_tps.clone();
            let budget = (budget.0 + ll, budget.1 + ll);
            let offset = offset + ll;
            enumerate_many(dsl, &ctx, env, f, idx, arg_tps, budget, offset, depth, cb)
        };
        enumerate(dsl, ctx, &arg_tp, env, (0f64, budget.1), depth, cb_arg)
    } else if budget.0 < 0f64 {
        cb(f.clone(), offset, ctx.clone())
    } else {
        true
    }
}
