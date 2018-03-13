use std::collections::VecDeque;
use std::iter;
use std::f64;
use std::rc::Rc;
use std::sync::mpsc::{self, channel};
use rayon::prelude::*;
use rayon;
use polytype::{Context, Type};

use super::{Expression, Language, LinkedList};

const BUDGET_INCREMENT: f64 = 1.0;
const MAX_DEPTH: u32 = 256;

pub fn new<'a>(dsl: &'a Language, request: Type) -> Box<Iterator<Item = (Expression, f64)> + 'a> {
    let to_budget = |offset: f64| (offset, offset + BUDGET_INCREMENT);
    Box::new(
        (0..)
            .map(|n| BUDGET_INCREMENT * f64::from(n))
            .map(to_budget)
            .zip(iter::repeat((dsl.clone(), request)))
            .flat_map(|(budget, (dsl, request))| new_par(dsl, request, budget)),
    )
}

/// enumerate expressions in parallel within the budget interval
fn new_par(dsl: Language, request: Type, budget: (f64, f64)) -> mpsc::IntoIter<(Expression, f64)> {
    let (tx, rx) = channel();
    rayon::spawn(move || {
        rayon::iter::repeat(tx)
            .zip(exponential_decay(budget))
            .for_each(|(tx, budget)| {
                let ctx = Context::default();
                let env = Rc::new(LinkedList::default());
                let e = enumerate(&dsl, request.clone(), &ctx, env.clone(), budget, 0)
                    .map(|(log_prior, _, expr)| (expr, log_prior));
                for (expr, logprior) in e {
                    if tx.send((expr, logprior)).is_err() {
                        // receiving end was dropped
                        break;
                    }
                }
            })
    });
    rx.into_iter()
}

fn exponential_decay(budget: (f64, f64)) -> Vec<(f64, f64)> {
    // because depth values correspond to description length in nats, we
    // assume that for pieces to have the same total description coverage
    // (which should correspond roughly to enumerate time), their budget
    // widths follow exponential decay.
    let pieces = match budget.1 {
        x if x < 4. => 2,
        x if x < 8. => 4,
        x if x < 10. => 8,
        x if x < 12. => 16,
        x if x < 15. => 32,
        x if x < 17. => 64,
        _ => 256,
    };
    let step = (budget.1.exp() - budget.0.exp()) / (f64::from(pieces));
    let mut v = Vec::new();
    let mut prev = budget.0;
    for x in (1..(pieces + 1))
        .map(|i| budget.0.exp() + (f64::from(i)) * step)
        .map(f64::ln)
    {
        v.push((prev, x));
        prev = x;
    }
    v
}

pub fn likelihood<'a>(dsl: &'a Language, request: &Type, expr: &Expression) -> f64 {
    let ctx = Context::default();
    let env = Rc::new(LinkedList::default());
    likelihood_internal(dsl, request, &ctx, &env, expr).0
}
fn likelihood_internal<'a>(
    dsl: &'a Language,
    request: &Type,
    ctx: &Context,
    env: &Rc<LinkedList<Type>>,
    mut expr: &Expression,
) -> (f64, Context) {
    if let Type::Arrow(ref arrow) = *request {
        let env = LinkedList::prepend(env, *arrow.arg.clone());
        if let Expression::Abstraction(ref body) = *expr {
            likelihood_internal(dsl, &*arrow.ret, ctx, &env, body)
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
                if let Type::Arrow(f_tp) = f_tp {
                    let arg_tps = f_tp.args();
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

fn enumerate<'a>(
    dsl: &'a Language,
    request: Type,
    ctx: &Context,
    env: Rc<LinkedList<Type>>,
    budget: (f64, f64),
    depth: u32,
) -> Box<Iterator<Item = (f64, Context, Expression)> + 'a> {
    if budget.1 <= 0f64 || depth > MAX_DEPTH {
        Box::new(iter::empty())
    } else if let Type::Arrow(arrow) = request {
        let env = LinkedList::prepend(&env, *arrow.arg);
        let it = enumerate(dsl, *arrow.ret, ctx, env, budget, depth)
            .map(|(ll, ctx, body)| (ll, ctx, Expression::Abstraction(Box::new(body))));
        Box::new(it)
    } else {
        Box::new(
            dsl.candidates(&request, ctx, &env.as_vecdeque())
                .into_iter()
                .filter(move |&(ll, _, _, _)| -ll <= budget.1)
                .flat_map(move |(ll, expr, tp, ctx)| {
                    let arg_tps: VecDeque<Type> = if let Type::Arrow(f_tp) = tp {
                        f_tp.args().into_iter().cloned().collect()
                    } else {
                        VecDeque::new()
                    };
                    let budget = (budget.0 + ll, budget.1 + ll);
                    enumerate_application(dsl, &ctx, env.clone(), expr, arg_tps, budget, depth + 1)
                        .map(move |(l, c, x)| (l + ll, c, x))
                }),
        )
    }
}

fn enumerate_application<'a>(
    dsl: &'a Language,
    ctx: &Context,
    env: Rc<LinkedList<Type>>,
    f: Expression,
    mut arg_tps: VecDeque<Type>,
    budget: (f64, f64),
    depth: u32,
) -> Box<Iterator<Item = (f64, Context, Expression)> + 'a> {
    if let Some(arg_tp) = arg_tps.pop_front() {
        let arg_tp = arg_tp.apply(ctx);
        Box::new(
            enumerate(dsl, arg_tp, ctx, env.clone(), (0f64, budget.1), depth).flat_map(
                move |(arg_ll, ctx, arg)| {
                    let f_next = Expression::Application(Box::new(f.clone()), Box::new(arg));
                    let budget = (budget.0 + arg_ll, budget.1 + arg_ll);
                    enumerate_application(
                        dsl,
                        &ctx,
                        env.clone(),
                        f_next,
                        arg_tps.clone(),
                        budget,
                        depth,
                    ).map(move |(l, c, x)| (arg_ll + l, c, x))
                },
            ),
        )
    } else if budget.0 < 0f64 && 0f64 <= budget.1 {
        Box::new(iter::once((0f64, ctx.clone(), f)))
    } else {
        Box::new(iter::empty())
    }
}
