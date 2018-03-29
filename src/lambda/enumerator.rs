use polytype::{Context, Type, TypeSchema};
use std::collections::VecDeque;
use std::f64;
use std::iter;
use std::rc::Rc;

use super::{Expression, Language, LinkedList};

#[cfg(feature = "par_enum")]
use crossbeam_channel::{self, bounded};
#[cfg(feature = "par_enum")]
use num_cpus;
#[cfg(feature = "par_enum")]
use rayon;
#[cfg(feature = "par_enum")]
use rayon::prelude::*;
#[cfg(feature = "par_enum")]
use std::env;
#[cfg(feature = "par_enum")]
lazy_static! {
    static ref PIECES: u32 = match env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
    {
        Some(x) if x > 0 => x,
        _ => num_cpus::get_physical() as u32,
    };
}

#[cfg(feature = "par_enum")]
const PAR_BUFFER_SIZE: usize = 512;
const BUDGET_INCREMENT: f64 = 1.0;
const MAX_DEPTH: u32 = 256;

fn budget_interval(n: u32) -> (f64, f64) {
    let offset = BUDGET_INCREMENT * f64::from(n);
    (offset, offset + BUDGET_INCREMENT)
}

#[cfg(not(feature = "par_enum"))]
pub fn new<'a>(
    dsl: &'a Language,
    request: TypeSchema,
) -> Box<Iterator<Item = (Expression, f64)> + 'a> {
    let mut ctx = Context::default();
    let tp = request.instantiate_owned(&mut ctx);
    let env = Rc::new(LinkedList::default());
    Box::new(
        (0..)
            .map(budget_interval)
            .flat_map(move |budget| {
                enumerate(dsl, &tp, &ctx, env.clone(), budget, 0)
            })
            // .map(move |(log_prior, log_likelihood, expr)| {
            //     println!("{} {}", log_prior, dsl.display(&expr));
            //     (log_prior, log_likelihood, expr)
            // })
            .map(|(log_prior, _, expr)| (expr, log_prior)),
    )
}
#[cfg(feature = "par_enum")]
pub fn new<'a>(
    dsl: &'a Language,
    request: TypeSchema,
) -> Box<Iterator<Item = (Expression, f64)> + 'a> {
    let mut ctx = Context::default();
    let request = request.instantiate_owned(&mut ctx);
    Box::new(
        (0..)
            .map(budget_interval)
            .zip(iter::repeat((dsl.clone(), request, ctx)))
            .flat_map(move |(budget, (dsl, request, ctx))| new_par(dsl, request, ctx, budget)),
    )
}

/// enumerate expressions in parallel within the budget interval
#[cfg(feature = "par_enum")]
fn new_par(
    dsl: Language,
    request: Type,
    ctx: Context,
    budget: (f64, f64),
) -> crossbeam_channel::IntoIter<(Expression, f64)> {
    let (tx, rx) = bounded(PAR_BUFFER_SIZE);
    rayon::spawn(move || {
        rayon::iter::repeat(tx)
            .zip(exponential_decay(budget))
            .for_each(|(tx, budget)| {
                let env = Rc::new(LinkedList::default());
                let e = enumerate(&dsl, &request, &ctx, env.clone(), budget, 0)
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

#[cfg(feature = "par_enum")]
fn exponential_decay(budget: (f64, f64)) -> Vec<(f64, f64)> {
    // because depth values correspond to description length in nats, we
    // assume that for pieces to have the same total description coverage
    // (which should correspond roughly to enumerate time), their budget
    // widths follow exponential decay.
    let step = (budget.1.exp() - budget.0.exp()) / (f64::from(*PIECES));
    let mut v = Vec::new();
    let mut prev = budget.0;
    for x in (1..(*PIECES + 1))
        .map(|i| budget.0.exp() + (f64::from(i)) * step)
        .map(f64::ln)
    {
        v.push((prev, x));
        prev = x;
    }
    v
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

fn enumerate<'a>(
    dsl: &'a Language,
    request: &Type,
    ctx: &Context,
    env: Rc<LinkedList<Type>>,
    budget: (f64, f64),
    depth: u32,
) -> Box<Iterator<Item = (f64, Context, Expression)> + 'a> {
    if budget.1 <= 0f64 || depth > MAX_DEPTH {
        Box::new(iter::empty())
    } else if let Some((arg, ret)) = request.as_arrow() {
        let env = LinkedList::prepend(&env, arg.clone());
        let it = enumerate(dsl, ret, ctx, env, budget, depth)
            .map(|(ll, ctx, body)| (ll, ctx, Expression::Abstraction(Box::new(body))));
        Box::new(it)
    } else {
        Box::new(
            dsl.candidates(request, ctx, &env.as_vecdeque())
                .into_iter()
                .filter(move |&(ll, _, _, _)| -ll <= budget.1)
                .flat_map(move |(ll, expr, tp, ctx)| {
                    let arg_tps: VecDeque<Type> = tp.args()
                        .map(|args| args.into_iter().cloned().collect())
                        .unwrap_or_else(VecDeque::new);
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
    if let Some(mut arg_tp) = arg_tps.pop_front() {
        arg_tp.apply_mut(ctx);
        Box::new(
            enumerate(dsl, &arg_tp, ctx, env.clone(), (0f64, budget.1), depth).flat_map(
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
