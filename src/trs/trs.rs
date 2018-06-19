///! Hindley-Milner Typing for First-Order Term Rewriting Systems (no abstraction)
///!
///! Much thanks to:
///! - https://github.com/rob-smallshire/hindley-milner-python
///! - https://en.wikipedia.org/wiki/Hindley%E2%80%93Milner_type_system
///! - (TAPL; Pierce, 2002, ch. 22)
use itertools::Itertools;
use polytype::{self, Context, Type, TypeSchema, Variable as TypeVar};
use rand::{thread_rng, Rng};
use std::f64::NEG_INFINITY;
use std::fmt;
use std::iter::{once, repeat};
use term_rewriting::{Atom, MergeStrategy, Operator, Rule, Signature, Term, Variable, TRS as UntypedTRS};

use super::trace::Trace;
use utils::logsumexp;
use GP;

#[derive(Debug, Clone)]
pub enum TypeError {
    Unification(polytype::UnificationError),
    OpNotFound,
    VarNotFound,
}
impl From<polytype::UnificationError> for TypeError {
    fn from(e: polytype::UnificationError) -> TypeError {
        TypeError::Unification(e)
    }
}
impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            TypeError::Unification(ref e) => write!(f, "unification error: {}", e),
            TypeError::OpNotFound => write!(f, "operator not found"),
            TypeError::VarNotFound => write!(f, "variable not found"),
        }
    }
}
impl ::std::error::Error for TypeError {
    fn description(&self) -> &'static str {
        "type error"
    }
}

#[derive(Debug, Clone)]
pub enum SampleError {
    TypeError(TypeError),
    DepthExceeded(usize, usize),
    OptionsExhausted,
    Subterm,
}
impl From<TypeError> for SampleError {
    fn from(e: TypeError) -> SampleError {
        SampleError::TypeError(e)
    }
}
impl From<polytype::UnificationError> for SampleError {
    fn from(e: polytype::UnificationError) -> SampleError {
        SampleError::TypeError(TypeError::Unification(e))
    }
}
impl fmt::Display for SampleError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            SampleError::TypeError(ref e) => write!(f, "type error: {}", e),
            SampleError::DepthExceeded(depth, max_depth) => {
                write!(f, "depth {} exceeded maximum of {}", depth, max_depth)
            }
            SampleError::OptionsExhausted => write!(f, "failed to sample (options exhausted)"),
            SampleError::Subterm => write!(f, "cannot sample subterm"),
        }
    }
}
impl ::std::error::Error for SampleError {
    fn description(&self) -> &'static str {
        "sample error"
    }
}

/// A first-order [Term Rewriting System][0] (TRS) with a [Hindley-Milner type system][1].
///
/// [0]: https://wikipedia.org/wiki/Hindley–Milner_type_system
///      "Wikipedia - Hindley-Milner Type System"
/// [1]: https://en.wikipedia.org/wiki/Rewriting#Term_rewriting_systems
///      "Wikipedia - Term Rewriting Systems"
#[derive(Debug, PartialEq, Clone)]
pub struct TRS {
    // TODO: may also want to track background knowledge here.
    ops: Vec<TypeSchema>,
    vars: Vec<TypeSchema>,
    pub signature: Signature,
    pub trs: UntypedTRS,
    pub ctx: Context,
}
impl TRS {
    /// The size of the TRS (the sum over the size of the rules in the underlying [`TRS`])
    ///
    /// [`TRS`]: ../../term_rewriting/struct.TRS.html
    pub fn size(&self) -> usize {
        self.trs.size()
    }
    /// All the free type variables in the lexicon.
    pub fn free_vars(&self) -> Vec<TypeVar> {
        let vars_fvs = self.vars.iter().flat_map(|x| x.free_vars());
        let ops_fvs = self.ops.iter().flat_map(|x| x.free_vars());
        vars_fvs.chain(ops_fvs).unique().collect()
    }
    fn var_tp(&self, v: &Variable) -> Result<TypeSchema, TypeError> {
        if let Some(idx) = self.signature.variables().iter().position(|x| x == v) {
            Ok(self.vars[idx].clone())
        } else {
            Err(TypeError::VarNotFound)
        }
    }
    fn op_tp(&self, o: &Operator) -> Result<TypeSchema, TypeError> {
        if let Some(idx) = self.signature.operators().iter().position(|x| x == o) {
            Ok(self.ops[idx].clone())
        } else {
            Err(TypeError::OpNotFound)
        }
    }
    /// Infer (lookup) the [`TypeSchema`] of an [`Operator`] or [`Variable`].
    ///
    /// [`TypeSchema`]: ../../polytype/struct.TypeSchema.html
    /// [`Operator`]: ../../term_rewriting/enum.Operator.html
    /// [`Variable`]: ../../term_rewriting/type.Variable.html
    pub fn infer_atom(&self, atom: &Atom) -> Result<TypeSchema, TypeError> {
        match *atom {
            Atom::Operator(ref o) => self.op_tp(o),
            Atom::Variable(ref v) => self.var_tp(v),
        }
    }
    fn instantiate_atom(&self, atom: &Atom, ctx: &mut Context) -> Result<Type, TypeError> {
        let mut tp = self.infer_atom(atom)?.instantiate_owned(ctx);
        tp.apply_mut(ctx);
        Ok(tp)
    }
    /// Infer the [`TypeSchema`] of a [`Term`].
    ///
    /// [`Term`]: ../../term_rewriting/enum.Term.html
    /// [`TypeSchema`]: ../../polytype/struct.TypeSchema.html
    pub fn infer_term(&self, term: &Term, ctx: &mut Context) -> Result<TypeSchema, TypeError> {
        let tp = self.infer_internal(term, ctx)?;
        // Get the variables bound by the lexicon
        let bound_vs = self.free_vars()
            .iter()
            .flat_map(|x| Type::Variable(*x).apply(ctx).vars())
            .unique()
            .collect::<Vec<u16>>();
        Ok(tp.generalize(&bound_vs))
    }
    fn infer_internal(&self, term: &Term, ctx: &mut Context) -> Result<Type, TypeError> {
        if let Term::Application { op, ref args } = *term {
            if op.arity(&self.signature) > 0 {
                let head_type = self.instantiate_atom(&Atom::from(op), ctx)?;
                let body_type = {
                    let mut pre_types = Vec::with_capacity(args.len() + 1);
                    for a in args {
                        pre_types.push(self.infer_internal(a, ctx)?);
                    }
                    pre_types.push(ctx.new_variable());
                    Type::from(pre_types)
                };
                ctx.unify(&head_type, &body_type)?;
            }
        }
        self.instantiate_atom(&term.head(), ctx)
    }
    /// Infer the [`TypeSchema`] of a [`Rule`], along with schemas for the left-hand-side and
    /// right-hand-sides of the rule.
    ///
    /// [`Rule`]: ../../term_rewriting/struct.Rule.html
    /// [`TypeSchema`]: ../../polytype/struct.TypeSchema.html
    pub fn infer_rule(
        &self,
        r: &Rule,
        ctx: &mut Context,
    ) -> Result<(TypeSchema, TypeSchema, Vec<TypeSchema>), TypeError> {
        let lhs_schema = self.infer_term(&r.lhs, ctx)?;
        let lhs_type = lhs_schema.instantiate(ctx);
        let mut rhs_types = Vec::with_capacity(r.rhs.len());
        let mut rhs_schemas = Vec::with_capacity(r.rhs.len());
        for rhs in &r.rhs {
            let rhs_schema = self.infer_term(&rhs, ctx)?;
            rhs_types.push(rhs_schema.instantiate(ctx));
            rhs_schemas.push(rhs_schema);
        }
        for rhs_type in rhs_types {
            ctx.unify(&lhs_type, &rhs_type)?;
        }
        // Get the variables bound by the lexicon
        let bound_vs = self.free_vars()
            .iter()
            .flat_map(|x| Type::Variable(*x).apply(&ctx).vars())
            .unique()
            .collect::<Vec<u16>>();
        let rule_schema = lhs_type.apply(ctx).generalize(&bound_vs);
        Ok((rule_schema, lhs_schema, rhs_schemas))
    }
    /// Infer the [`Context`] where every [`Rule`] in a [`TRS`] typechecks.
    ///
    /// [`Context`]: ../../polytype/struct.Context.html
    /// [`Rule`]: ../../term_rewriting/struct.Rule.html
    /// [`TRS`]: ../../term_rewriting/struct.TRS.html
    pub fn infer_trs(&self, ctx: &mut Context) -> Result<(), TypeError> {
        // TODO: Right now, this assumes the variables already exist in the signature. Is that sensible?
        for rule in &self.trs.rules {
            self.infer_rule(rule, ctx)?;
        }
        Ok(())
    }
    /// Sample a `Term`.
    pub fn sample_term(
        &mut self,
        schema: &TypeSchema,
        ctx: &mut Context,
        invent: bool,
        max_d: usize,
    ) -> Result<Term, SampleError> {
        self.sample_term_internal(schema, ctx, invent, max_d, 0)
    }
    fn sample_term_internal(
        &mut self,
        schema: &TypeSchema,
        ctx: &mut Context,
        invent: bool,
        max_d: usize,
        d: usize,
    ) -> Result<Term, SampleError> {
        // fail if we've gone too deep
        if d > max_d {
            return Err(SampleError::DepthExceeded(d, max_d));
        }

        let tp = schema.instantiate(ctx);

        let mut options: Vec<Option<Atom>> = self.signature.atoms().into_iter().map(Some).collect();
        if invent {
            options.push(None);
        }
        thread_rng().shuffle(&mut options);
        for option in options {
            let atom = option.unwrap_or_else(|| Atom::Variable(self.invent_variable(&tp)));
            let arg_types = self.fit_atom(&atom, &tp, ctx)?;
            let result = self.place_atom(&atom, arg_types, ctx, invent, max_d, d);
            if result.is_ok() {
                return result;
            }
        }
        Err(SampleError::OptionsExhausted)
    }
    /// Sample a `Rule`.
    pub fn sample_rule(
        &mut self,
        schema: &TypeSchema,
        ctx: &mut Context,
        invent: bool,
        max_d: usize,
    ) -> Result<Rule, SampleError> {
        self.sample_rule_internal(schema, ctx, invent, max_d, 0)
    }
    fn sample_rule_internal(
        &mut self,
        schema: &TypeSchema,
        ctx: &mut Context,
        invent: bool,
        max_d: usize,
        d: usize,
    ) -> Result<Rule, SampleError> {
        let orig_self = self.clone();
        let orig_ctx = ctx.clone();
        loop {
            let lhs = self.sample_term_internal(schema, ctx, invent, max_d, d)?;
            let rhs = self.sample_term_internal(schema, ctx, false, max_d, d)?;
            if let Some(rule) = Rule::new(lhs, vec![rhs]) {
                return Ok(rule);
            } else {
                *self = orig_self.clone();
                *ctx = orig_ctx.clone();
            }
        }
    }
    /// Give the log probability of sampling a Term.
    pub fn logprior_term(
        &self,
        term: &Term,
        schema: &TypeSchema,
        ctx: &mut Context,
        invent: bool,
    ) -> Result<f64, SampleError> {
        // instantiate the typeschema
        let tp = schema.instantiate(ctx);
        // setup the existing options
        let mut options = Vec::new();
        let atoms = self.signature.atoms();
        for atom in &atoms {
            if let Ok(arg_types) = self.fit_atom(atom, &tp, ctx) {
                options.push((Some(*atom), arg_types))
            }
        }
        // add a variable if needed
        if invent {
            if let Term::Variable(v) = *term {
                if !atoms.contains(&Atom::Variable(v)) {
                    options.push((Some(Atom::Variable(v)), Vec::new()));
                }
            } else {
                options.push((None, Vec::new()));
            }
        }
        // compute the log probability of selecting this head
        let mut lp = -(options.len() as f64).ln(); // unused if undefined
        match options
            .into_iter()
            .find(|&(ref o, _)| o == &Some(term.head()))
            .map(|(o, arg_types)| (o.unwrap(), arg_types))
        {
            Some((Atom::Variable(_), _)) => Ok(lp),
            Some((Atom::Operator(_), arg_types)) => {
                for (subterm, mut arg_tp) in term.args().iter().zip(arg_types) {
                    arg_tp.apply_mut(ctx);
                    let arg_schema = TypeSchema::Monotype(arg_tp.clone());
                    lp += self.logprior_term(subterm, &arg_schema, ctx, invent)?;
                    let final_type = self.infer_term(subterm, ctx)?.instantiate_owned(ctx);
                    if ctx.unify(&arg_tp, &final_type).is_err() {
                        return Ok(NEG_INFINITY);
                    }
                }
                Ok(lp)
            }
            None => Ok(NEG_INFINITY),
        }
    }
    /// Give the log probability of sampling a Rule.
    pub fn logprior_rule(
        &self,
        rule: &Rule,
        schema: &TypeSchema,
        ctx: &mut Context,
        invent: bool,
    ) -> Result<f64, SampleError> {
        let mut lp = 0.0;
        let lp_lhs = self.logprior_term(&rule.lhs, schema, ctx, invent)?;
        for rhs in &rule.rhs {
            let tmp_lp = self.logprior_term(&rhs, schema, ctx, false)?;
            lp += tmp_lp + lp_lhs;
        }
        Ok(lp)
    }
    /// Give the log probability of sampling a TRS.
    pub fn logprior_trs(
        &self,
        trs: &UntypedTRS,
        schemas: &[TypeSchema],
        p_rule: f64,
        ctx: &mut Context,
        invent: bool,
    ) -> Result<f64, SampleError> {
        // TODO: this might not be numerically stable
        // geometric distribution over number of rules
        let p_n_rules = p_rule.ln() * (trs.clauses().len() as f64);
        let mut p_rules = 0.0;
        for rule in &trs.rules {
            let mut rule_ps = vec![];
            for schema in schemas {
                let tmp_lp = self.logprior_rule(&rule, schema, ctx, invent)?;
                rule_ps.push(tmp_lp);
            }
            p_rules += logsumexp(&rule_ps);
        }
        Ok(p_n_rules + p_rules)
    }
    fn fit_atom(
        &self,
        atom: &Atom,
        tp: &Type,
        ctx: &mut Context,
    ) -> Result<Vec<Type>, SampleError> {
        let atom_tp = self.instantiate_atom(atom, ctx)?;
        ctx.unify(&atom_tp, tp)?;
        Ok(atom_tp
            .args()
            .map(|o| o.into_iter().cloned().collect())
            .unwrap_or_else(Vec::new))
    }
    fn place_atom(
        &mut self,
        atom: &Atom,
        arg_types: Vec<Type>,
        ctx: &mut Context,
        invent: bool,
        max_d: usize,
        d: usize,
    ) -> Result<Term, SampleError> {
        match *atom {
            Atom::Variable(v) => Ok(Term::Variable(v)),
            Atom::Operator(op) => {
                let orig_ctx = ctx.clone(); // for "undo" semantics
                let mut args = Vec::with_capacity(arg_types.len());
                for (i, arg_tp) in arg_types.into_iter().enumerate() {
                    let subtype = arg_tp.apply(ctx);
                    let arg_schema = TypeSchema::Monotype(arg_tp);
                    let d_i = (d + 1) * ((i == 0) as usize);
                    let result = self.sample_term_internal(&arg_schema, ctx, invent, max_d, d_i)
                        .map_err(|_| SampleError::Subterm)
                        .and_then(|subterm| {
                            let tp = self.infer_term(&subterm, ctx)?.instantiate_owned(ctx);
                            ctx.unify_fast(subtype, tp)?;
                            Ok(subterm)
                        });
                    match result {
                        Ok(subterm) => args.push(subterm),
                        Err(e) => {
                            *ctx = orig_ctx;
                            return Err(e);
                        }
                    }
                }
                Ok(Term::Application { op, args })
            }
        }
    }
    /// Create a brand-new variable.
    fn invent_variable(&mut self, tp: &Type) -> Variable {
        let var = self.signature.new_var(None);
        self.vars.push(TypeSchema::Monotype(tp.clone()));
        var
    }

    pub fn pseudo_log_prior(&self, temp: f64, prior_temp: f64) -> f64 {
        let raw_prior = -(self.size() as f64);
        raw_prior / ((temp + 1.0) * prior_temp)
    }

    pub fn log_likelihood(&self, data: &[Rule], p_partial: f64, temp: f64, ll_temp: f64) -> f64 {
        data.iter()
            .map(|x| self.single_log_likelihood(x, p_partial, temp) / ll_temp)
            .sum()
    }

    fn single_log_likelihood(&self, datum: &Rule, p_partial: f64, temp: f64) -> f64 {
        let p_observe = 0.0;
        let max_steps = 50;
        let max_size = 500;
        let mut trace = Trace::new(&self.trs, &datum.lhs, p_observe, max_steps, max_size);
        trace.run();

        let ll = if let Some(ref rhs) = datum.rhs() {
            trace.rewrites_to(rhs)
        } else {
            NEG_INFINITY
        };

        if ll == NEG_INFINITY {
            (p_partial + temp).ln()
        } else {
            (1.0 - p_partial + temp).ln() + ll
        }
    }

    pub fn posterior(
        &self,
        data: &[Rule],
        p_partial: f64,
        temperature: f64,
        prior_temperature: f64,
        ll_temperature: f64,
    ) -> f64 {
        let prior = self.pseudo_log_prior(temperature, prior_temperature);
        if prior == NEG_INFINITY {
            NEG_INFINITY
        } else {
            prior + self.log_likelihood(data, p_partial, temperature, ll_temperature)
        }
    }

    /// Sample a rule and add it to the TRS.
    pub fn add_rule<R: Rng>(&self, rng: &mut R) -> Result<TRS, SampleError> {
        let mut new_trs = self.clone();

        let var = new_trs.ctx.new_variable();
        let schema = TypeSchema::Monotype(var);

        new_trs.ctx.reset();
        let mut ctx = new_trs.ctx.clone();

        let rule = new_trs.sample_rule(&schema, &mut ctx, true, 4)?;
        self.infer_rule(&rule, &mut ctx)?;
        new_trs.trs.push(rule);
        new_trs.ctx = ctx;

        Ok(new_trs)
    }
    /// Delete a rule from the TRS if possible.
    pub fn delete_rule<R: Rng>(&self, rng: &mut R) -> Result<TRS, SampleError> {
        let mut new_trs = self.clone();
        if !self.trs.is_empty() {
            let idx = rng.gen_range(0, self.trs.len());
            new_trs.trs.rules.remove(idx);
            Ok(new_trs)
        } else {
            Err(SampleError::OptionsExhausted)
        }
    }
    /// merge two `TRS` into a single `TRS`.
    pub fn combine(trs1: &TRS, trs2: &TRS) -> TRS {
        // merge the Signatures
        let mut signature = trs1.signature.clone();
        let sig2 = trs2.signature.clone();
        let sig_change = signature.merge(sig2, MergeStrategy::SameOperators);
        // merge the Contexts
        let mut ctx = trs1.ctx.clone();
        let ctx2 = trs2.ctx.clone();
        let p2_op_free_typevars = trs1
            .ops
            .iter()
            .flat_map(TypeSchema::free_vars)
            .unique()
            .collect();
        let ctx_change = ctx.merge(ctx2, p2_op_free_typevars);
        // merge/reify the variable types
        let mut vars = trs1.vars.clone();
        let mut vars2 = trs2
            .vars
            .clone()
            .into_iter()
            .map(|mut x| {
                ctx_change.reify_typeschema(&mut x);
                x
            })
            .collect();
        vars.append(&mut vars2);
        // merge/reify the operator types -- assuming they're the same makes this easy
        let ops = trs1.ops.clone();
        // merge/reify the ruleset
        let mut trs = trs1.trs.clone();
        trs.rules.append(&mut sig_change.reify_trs(trs2.trs.clone()).rules);
        // create the TRS
        TRS {
            signature,
            ctx,
            trs,
            vars,
            ops,
        }
    }
}
impl fmt::Display for TRS {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.trs.display(&self.signature))
    }
}

#[derive(Debug, Clone)]
pub struct TRSParams {
    h0: TRS,
    n_crosses: usize,
}

#[derive(Debug, Clone)]
pub struct TRSSpace;
impl GP for TRSSpace {
    type Expression = TRS;
    type Params = TRSParams;
    fn genesis<R: Rng>(
        &self,
        params: &Self::Params,
        rng: &mut R,
        pop_size: usize,
        tp: &TypeSchema,
    ) -> Vec<Self::Expression> {
        repeat(params.h0.clone()).take(pop_size).collect()
    }
    fn mutate<R: Rng>(
        &self,
        params: &Self::Params,
        rng: &mut R,
        prog: &Self::Expression,
    ) -> Self::Expression {
        loop {
            if rng.gen_bool(0.5) {
                if let Ok(new_expr) = prog.add_rule(rng) {
                    return new_expr;
                }
            } else if let Ok(new_expr) = prog.delete_rule(rng) {
                return new_expr;
            }
        }
    }
    fn crossover<R: Rng>(
        &self,
        params: &Self::Params,
        rng: &mut R,
        parent1: &Self::Expression,
        parent2: &Self::Expression,
    ) -> Vec<Self::Expression> {
        let trs = TRS::combine(parent1, parent2);
        let trss = repeat(trs.clone()).take(params.n_crosses).map(|mut x| {
            x.trs.rules = x
                .trs
                .rules
                .into_iter()
                .filter(|_| rng.gen_bool(0.5))
                .collect();
            x
        });
        once(trs).chain(trss).collect()
    }
}
