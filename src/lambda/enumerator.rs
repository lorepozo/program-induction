use std::collections::VecDeque;
use std::iter;
use std::f64;
use std::rc::Rc;
use polytype::{Context, Type};
use super::{Expression, Language};

const BUDGET_INCREMENT: f64 = 1.0;
const MAX_DEPTH: u32 = 256;

pub fn new<'a>(dsl: &'a Language, request: Type) -> Box<Iterator<Item = (Expression, f64)> + 'a> {
    let budget = |offset: f64| (offset, offset + BUDGET_INCREMENT);
    let ctx = Context::default();
    let env = Rc::new(LinkedList::default());
    let depth = 0;
    Box::new(
        (0..)
            .map(|n| BUDGET_INCREMENT * f64::from(n))
            .flat_map(move |offset| {
                enumerate(
                    dsl,
                    request.clone(),
                    &ctx,
                    env.clone(),
                    budget(offset),
                    depth,
                ).map(move |(log_prior, _, expr)| (expr, log_prior))
            }),
    )
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
            candidates(dsl, &request, ctx, &env.as_vecdeque())
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

fn candidates(
    dsl: &Language,
    request: &Type,
    ctx: &Context,
    env: &VecDeque<Type>,
) -> Vec<(f64, Expression, Type, Context)> {
    let mut cands = Vec::new();
    let prims = dsl.primitives
        .iter()
        .zip(&dsl.primitives_logprob)
        .enumerate()
        .map(|(i, (&(_, ref tp), &p))| (p, tp, true, Expression::Primitive(i)));
    let invented = dsl.invented
        .iter()
        .zip(&dsl.invented_logprob)
        .enumerate()
        .map(|(i, (&(_, ref tp), &p))| (p, tp, true, Expression::Invented(i)));
    let indices = env.iter()
        .enumerate()
        .map(|(i, tp)| (dsl.variable_logprob, tp, false, Expression::Index(i)));
    for (p, tp, instantiate, expr) in prims.chain(invented).chain(indices) {
        let mut ctx = ctx.clone();
        let itp;
        let tp = if instantiate {
            itp = tp.instantiate_indep(&mut ctx);
            &itp
        } else {
            tp
        };
        let ret = if let Type::Arrow(ref arrow) = *tp {
            arrow.returns()
        } else {
            &tp
        };
        if ctx.unify(ret, request).is_ok() {
            let tp = tp.apply(&ctx);
            cands.push((p, expr, tp, ctx))
        }
    }
    // update probabilities for variables (indices)
    let n_indexed = cands
        .iter()
        .filter(|&&(_, ref expr, _, _)| match *expr {
            Expression::Index(_) => true,
            _ => false,
        })
        .count() as f64;
    for mut c in &mut cands {
        if let Expression::Index(_) = c.1 {
            c.0 -= n_indexed.ln()
        }
    }
    // normalize
    let p_largest = cands
        .iter()
        .map(|&(p, _, _, _)| p)
        .fold(f64::NEG_INFINITY, |acc, p| acc.max(p));
    let z = p_largest
        + cands
            .iter()
            .map(|&(p, _, _, _)| (p - p_largest).exp())
            .sum::<f64>()
            .ln();
    for mut c in &mut cands {
        c.0 -= z;
    }
    cands
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
}
impl<T: Clone> Default for LinkedList<T> {
    fn default() -> Self {
        LinkedList(None)
    }
}
