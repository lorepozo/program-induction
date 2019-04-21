use polytype::{Context, Type, TypeSchema};
use std::collections::VecDeque;
use std::f64;
use std::rc::Rc;

use super::{Expression, Language, LinkedList};

const MAX_DEPTH: u32 = 8192;

fn budget_interval(n: u32) -> (f64, f64) {
    match n / 6 {
        0 => {
            let offset = f64::from(n) * 2.0;
            (offset, offset + 2.0)
        }
        1 => {
            let offset = 12. + f64::from(n - 6) * 1.0;
            (offset, offset + 1.0)
        }
        _ => {
            let offset = 18. + f64::from(n - 12) * 0.5;
            (offset, offset + 0.5)
        }
    }
}

pub fn run<F>(dsl: &Language, request: TypeSchema, termination_condition: F)
where
    F: Fn(Expression, f64) -> bool + Send + Sync,
{
    let mut ctx = Context::default();
    let tp = request.instantiate_owned(&mut ctx);
    if ::rayon::current_num_threads() == 1 {
        // dfs
        let env = Rc::new(LinkedList::default());
        let cb = &mut |expr, logprior, _| !termination_condition(expr, logprior);
        (0..).map(budget_interval).all(|budget| {
            if cfg!(feature = "verbose") {
                eprintln!(
                    "ENUMERATION: starting budget {:?} for request {}",
                    budget, &tp
                );
            }
            enumerate(dsl, &ctx, &tp, &env, budget, 0, cb)
        });
    } else {
        // partial bfs then dfs
        let cb = move |expr, logprior, _| !termination_condition(expr, logprior);
        (0..).map(budget_interval).all(|budget| {
            if cfg!(feature = "verbose") {
                eprintln!(
                    "ENUMERATION: starting budget {:?} for request {}",
                    budget, &tp
                );
            }
            self::par::enumerate(dsl, &ctx, &tp, budget, &cb)
        });
    }
    if cfg!(feature = "verbose") {
        eprintln!("ENUMERATION: finished for request {}", &tp);
    }
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
        match dsl
            .candidates(request, ctx, &env.as_vecdeque())
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
                        .unwrap_or_else(|_| panic!("could not infer type for {}", s)),
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
    cb: &mut dyn FnMut(Expression, f64, Context) -> bool,
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
                let arg_tps: VecDeque<Type> = tp
                    .args()
                    .map(|args| args.into_iter().cloned().collect())
                    .unwrap_or_else(VecDeque::new);
                let budget = (budget.0 + p, budget.1 + p);
                let depth = depth + 1;
                let idx = (0, &expr);
                enumerate_many(dsl, &ctx, env, &expr, idx, arg_tps, budget, p, depth, cb)
            })
    }
}

#[allow(clippy::too_many_arguments)]
fn enumerate_many(
    dsl: &Language,
    ctx: &Context,
    env: &Rc<LinkedList<Type>>,
    f: &Expression,
    idx: (usize, &Expression),
    mut arg_tps: VecDeque<Type>,
    budget: (f64, f64),
    offset: f64,
    depth: u32,
    cb: &mut dyn FnMut(Expression, f64, Context) -> bool,
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
            enumerate_many(dsl, &ctx, env, &f, idx, arg_tps, budget, offset, depth, cb)
        };
        enumerate(dsl, ctx, &arg_tp, env, (0f64, budget.1), depth, cb_arg)
    } else if budget.0 < 0f64 {
        cb(f.clone(), offset, ctx.clone())
    } else {
        true
    }
}

mod par {
    use polytype::{Context, Type};
    use rayon::prelude::*;

    use super::super::{Expression, Language};

    const ENUMERATE_LOAD: usize = 4;

    /// Serial entry point for parallel enumeration
    pub fn enumerate<F>(
        dsl: &Language,
        ctx: &Context,
        request: &Type,
        budget: (f64, f64),
        cb: F,
    ) -> bool
    where
        F: Fn(Expression, f64, Context) -> bool + Send + Sync,
    {
        let shards = ::rayon::current_num_threads() * ENUMERATE_LOAD;
        let (items, bfss) = super::bfs::search(dsl, ctx, request, shards);
        if items
            .into_iter()
            .filter(|&(_, ll, _)| ll >= budget.0 && ll < budget.1)
            .all(|(expr, ll, ctx)| cb(expr, ll, ctx))
        {
            bfss.into_par_iter()
                .map(|bfs| bfs.enumerate_dfs(&dsl, budget, &cb))
                .all(|b| b)
        } else {
            false
        }
    }
}

mod bfs {
    use polytype::{Context, Type};
    use std::collections::{BinaryHeap, VecDeque};
    use std::f64;
    use std::ops::{Index, IndexMut};
    use std::rc::Rc;

    use super::super::{Expression, Language, LinkedList};

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum Turn {
        Abs(Type),
        Left,
        Right,
    }
    /// Specifies how to find a `Hole` in an `HoleExpression`.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Path(Vec<Turn>);
    impl Path {
        fn environment(&self) -> VecDeque<Type> {
            self.0
                .iter()
                .rev()
                .filter_map(|t| match *t {
                    Turn::Abs(ref tp) => Some(tp.clone()),
                    _ => None,
                })
                .collect()
        }
        /// Replaces the innermost left-turn and subsequent turns with a right-turn.
        fn unwind(&self) -> Path {
            self.0
                .iter()
                .rev()
                .position(|t| match *t {
                    Turn::Left => true,
                    _ => false,
                })
                .map(|last_left| {
                    let len = self.0.len() - last_left;
                    let mut turns = self.0[..len].to_vec();
                    turns[len - 1] = Turn::Right;
                    Path(turns)
                })
                .unwrap_or_else(|| self.clone())
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    enum HoleExpression {
        Hole(Type),
        Primitive(usize),
        Application(Box<HoleExpression>, Box<HoleExpression>),
        Abstraction(Box<HoleExpression>),
        Index(usize),
        Invented(usize),
    }
    impl HoleExpression {
        fn as_expression(&self) -> Option<Expression> {
            match *self {
                HoleExpression::Hole(_) => None,
                HoleExpression::Primitive(n) => Some(Expression::Primitive(n)),
                HoleExpression::Application(ref f, ref x) => {
                    let f = f.as_expression()?;
                    let x = x.as_expression()?;
                    Some(Expression::Application(Box::new(f), Box::new(x)))
                }
                HoleExpression::Abstraction(ref body) => {
                    let body = body.as_expression()?;
                    Some(Expression::Abstraction(Box::new(body)))
                }
                HoleExpression::Index(i) => Some(Expression::Index(i)),
                HoleExpression::Invented(n) => Some(Expression::Invented(n)),
            }
        }
        fn fill(&self, path: &Path, replacement: HoleExpression) -> HoleExpression {
            let mut expr = self.clone();
            expr[&path.0] = replacement;
            expr
        }
        fn free(&self) -> bool {
            match *self {
                HoleExpression::Hole(_) => true,
                HoleExpression::Application(ref f, ref x) if f.free() || x.free() => true,
                HoleExpression::Abstraction(ref body) if body.free() => true,
                _ => false,
            }
        }
        #[allow(clippy::too_many_arguments)]
        fn enumerate_dfs(
            &self,
            dsl: &Language,
            ctx: &Context,
            env: &VecDeque<Type>,
            budget: (f64, f64),
            idx: Option<(usize, &Expression)>,
            abs_depth: usize,
            cb: &mut dyn FnMut(Expression, f64, Context) -> bool,
        ) -> bool {
            match *self {
                HoleExpression::Hole(ref req) => {
                    let mut nenv = Rc::new(LinkedList::default());
                    for tp in env.iter().rev().take(abs_depth) {
                        nenv = LinkedList::prepend(&nenv, tp.clone());
                    }
                    let cb = &mut |expr, ll, ctx| {
                        if let Some((i, e)) = idx {
                            if dsl.violates_symmetry(e, i, &expr) {
                                return true;
                            }
                        }
                        cb(expr, ll, ctx)
                    };
                    super::enumerate(dsl, ctx, req, &nenv, budget, 0, cb)
                }
                HoleExpression::Application(ref hf, ref hx) if !hf.free() => {
                    let f = hf.as_expression().unwrap();
                    let idx = Some(idx.map(|(i, e)| (i + 1, e)).unwrap_or((0, &f)));
                    let cb_arg = &mut |arg, ll, ctx| {
                        if let Some((i, e)) = idx {
                            if dsl.violates_symmetry(e, i, &arg) {
                                return true;
                            }
                        }
                        cb(
                            Expression::Application(Box::new(f.clone()), Box::new(arg)),
                            ll,
                            ctx,
                        )
                    };
                    hx.enumerate_dfs(dsl, ctx, env, budget, idx, abs_depth, cb_arg)
                }
                HoleExpression::Application(ref hf, ref hx) => {
                    // both hf and hx are free
                    // TODO: consider not duplicating some work
                    let cb_f = &mut |f: Expression, f_ll, ctx| {
                        let idx = Some(idx.map(|(i, e)| (i + 1, e)).unwrap_or((0, &f)));
                        let cb_arg = &mut |arg, arg_ll, ctx| {
                            if let Some((i, e)) = idx {
                                if dsl.violates_symmetry(e, i, &arg) {
                                    return true;
                                }
                            }
                            cb(
                                Expression::Application(Box::new(f.clone()), Box::new(arg)),
                                f_ll + arg_ll,
                                ctx,
                            )
                        };
                        let budget = (budget.0 + f_ll, budget.1 + f_ll);
                        hx.enumerate_dfs(dsl, &ctx, env, budget, idx, abs_depth, cb_arg)
                    };
                    hf.enumerate_dfs(dsl, ctx, env, (0.0, budget.1), None, abs_depth, cb_f)
                }
                HoleExpression::Abstraction(ref body) => {
                    let cb =
                        &mut |body, ll, ctx| cb(Expression::Abstraction(Box::new(body)), ll, ctx);
                    body.enumerate_dfs(dsl, ctx, env, budget, None, abs_depth + 1, cb)
                }
                _ => unreachable!(/* only holes, applications, and abstractions can be free */),
            }
        }
    }
    impl From<Expression> for HoleExpression {
        fn from(e: Expression) -> HoleExpression {
            match e {
                Expression::Primitive(n) => HoleExpression::Primitive(n),
                Expression::Application(f, x) => {
                    let f = HoleExpression::from(*f);
                    let x = HoleExpression::from(*x);
                    HoleExpression::Application(Box::new(f), Box::new(x))
                }
                Expression::Abstraction(body) => {
                    let body = HoleExpression::from(*body);
                    HoleExpression::Abstraction(Box::new(body))
                }
                Expression::Index(i) => HoleExpression::Index(i),
                Expression::Invented(n) => HoleExpression::Invented(n),
            }
        }
    }
    impl<'a> Index<&'a [Turn]> for HoleExpression {
        type Output = HoleExpression;
        fn index(&self, path: &'a [Turn]) -> &HoleExpression {
            if path.is_empty() {
                self
            } else {
                let t = &path[0];
                match (t, self) {
                    (&Turn::Abs(_), &HoleExpression::Abstraction(ref body)) => {
                        body.index(&path[1..])
                    }
                    (&Turn::Left, &HoleExpression::Application(ref f, _)) => f.index(&path[1..]),
                    (&Turn::Right, &HoleExpression::Application(_, ref x)) => x.index(&path[1..]),
                    _ => panic!("invalid path for expression: {:?} against {:?}", t, self),
                }
            }
        }
    }
    impl<'a> IndexMut<&'a [Turn]> for HoleExpression {
        fn index_mut(&mut self, path: &'a [Turn]) -> &mut HoleExpression {
            if path.is_empty() {
                self
            } else {
                let t = &path[0];
                match (t, self) {
                    (&Turn::Abs(_), &mut HoleExpression::Abstraction(ref mut body)) => {
                        body.index_mut(&path[1..])
                    }
                    (&Turn::Left, &mut HoleExpression::Application(ref mut f, _)) => {
                        f.index_mut(&path[1..])
                    }
                    (&Turn::Right, &mut HoleExpression::Application(_, ref mut x)) => {
                        x.index_mut(&path[1..])
                    }
                    (t, e) => panic!("invalid path for expression: {:?} against {:?}", t, e),
                }
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct BestFirstState {
        expr: HoleExpression,
        path: Path,
        ctx: Context,
        cost: f64,
    }
    impl BestFirstState {
        fn new(ctx: &Context, request: &Type) -> BestFirstState {
            let ctx = ctx.clone();
            BestFirstState {
                expr: HoleExpression::Hole(request.clone()),
                path: Path(Vec::new()),
                ctx,
                cost: 0f64,
            }
        }
        fn successors(&self, dsl: &Language) -> Vec<BestFirstState> {
            let req = match self.expr[&self.path.0] {
                HoleExpression::Hole(ref t) => t,
                ref e => panic!(
                    "invalid path for expression (successor): {:?} against {:?} yielded {:?}",
                    self.path, self.expr, e
                ),
            };
            if let Some((arg, ret)) = req.as_arrow() {
                let expr = self.expr.fill(
                    &self.path,
                    HoleExpression::Abstraction(Box::new(HoleExpression::Hole(ret.clone()))),
                );
                let mut path = self.path.clone();
                path.0.push(Turn::Abs(arg.clone()));
                let ctx = self.ctx.clone();
                let cost = self.cost;
                vec![BestFirstState {
                    expr,
                    path,
                    ctx,
                    cost,
                }]
            } else {
                let env = self.path.environment();
                dsl.candidates(req, &self.ctx, &env)
                    .into_iter()
                    .filter(|&(_, ref expr, _, _)| match self.expr {
                        // FIXME: this only checks for symmetry violations starting from the root
                        HoleExpression::Application(ref f, _) => {
                            let mut i = 0;
                            let mut f: &HoleExpression = &*f;
                            let applied = loop {
                                match *f {
                                    HoleExpression::Application(ref ff, _) => {
                                        f = &*ff;
                                        i += 1;
                                    }
                                    _ => break f.as_expression().map(|f| (i, f)),
                                }
                            };
                            if let Some((i, f)) = applied {
                                !dsl.violates_symmetry(&f, i, expr)
                            } else {
                                true
                            }
                        }
                        _ => true,
                    })
                    .map(|(p, expr, tp, ctx)| {
                        if let Some(arg_tps) = tp.args() {
                            let tpl = arg_tps.iter().fold(expr.into(), |e, &arg_tp| {
                                HoleExpression::Application(
                                    Box::new(e),
                                    Box::new(HoleExpression::Hole(arg_tp.clone())),
                                )
                            });
                            let mut path = self.path.clone();
                            path.0.extend(vec![Turn::Left; arg_tps.len() - 1]);
                            path.0.push(Turn::Right);
                            BestFirstState {
                                ctx,
                                cost: self.cost - p,
                                path,
                                expr: self.expr.fill(&self.path, tpl),
                            }
                        } else {
                            BestFirstState {
                                ctx,
                                cost: self.cost - p,
                                path: self.path.unwind(),
                                expr: self.expr.fill(&self.path, expr.into()),
                            }
                        }
                    })
                    .collect()
            }
        }
        /// Should only be called after search, so the underlying expression is ensured to have
        /// holes.
        pub fn enumerate_dfs<F>(&self, dsl: &Language, budget: (f64, f64), mut cb: F) -> bool
        where
            F: FnMut(Expression, f64, Context) -> bool,
        {
            let env = self.path.environment();
            let budget = (budget.0 - self.cost, budget.1 - self.cost);
            let cb = &mut |body, ll, ctx| cb(body, ll - self.cost, ctx);
            self.expr
                .enumerate_dfs(dsl, &self.ctx, &env, budget, None, 0, cb)
        }
    }
    impl PartialEq for BestFirstState {
        fn eq(&self, other: &BestFirstState) -> bool {
            self.cost == other.cost && self.expr == other.expr
        }
    }
    impl Eq for BestFirstState {}
    impl PartialOrd for BestFirstState {
        fn partial_cmp(&self, other: &BestFirstState) -> Option<::std::cmp::Ordering> {
            self.cost.partial_cmp(&other.cost)
        }
    }
    impl Ord for BestFirstState {
        fn cmp(&self, other: &BestFirstState) -> ::std::cmp::Ordering {
            self.partial_cmp(other).unwrap()
        }
    }

    pub fn search(
        dsl: &Language,
        ctx: &Context,
        request: &Type,
        at_least: usize,
    ) -> (Vec<(Expression, f64, Context)>, Vec<BestFirstState>) {
        let mut exprs = Vec::new();
        let mut queue = BinaryHeap::with_capacity(at_least);
        queue.push(BestFirstState::new(ctx, request));
        while queue.len() < at_least {
            let s = queue.pop().unwrap();
            for s in s.successors(dsl) {
                if let Some(expr) = s.expr.as_expression() {
                    exprs.push((expr, s.cost, s.ctx))
                } else {
                    queue.push(s)
                }
            }
        }
        (exprs, queue.into_vec())
    }
}
