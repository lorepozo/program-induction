use itertools::Itertools;
use polytype::{Context, Type, TypeSchema, Variable as TypeVar};
use rand::{thread_rng, Rng};
use std::f64::NEG_INFINITY;
use std::fmt;
use std::iter::{once, repeat};
use std::sync::{Arc, RwLock};
use term_rewriting::{Atom, Operator, Rule, Signature, Term, Variable, TRS};

use super::{RewriteSystem, SampleError, TypeError};
use utils::logsumexp;
use GP;

#[derive(Debug, Clone)]
pub struct GeneticParams {
    pub h0: RewriteSystem,
    pub n_crosses: usize,
}

/// Manages the syntax of a term rewriting system.
#[derive(Clone)]
pub struct Lexicon(pub(crate) Arc<RwLock<Lex>>);
impl Lexicon {
    /// All the free type variables in the lexicon.
    pub fn free_vars(&self) -> Vec<TypeVar> {
        self.0.read().expect("poisoned lexicon").free_vars()
    }
    /// Sample a `Term`.
    pub fn sample_term(
        &mut self,
        schema: &TypeSchema,
        ctx: &mut Context,
        invent: bool,
        max_d: usize,
    ) -> Result<Term, SampleError> {
        let mut lex = self.0.write().expect("poisoned lexicon");
        lex.sample_term(schema, ctx, invent, max_d, 0)
    }
    /// Sample a `Rule`.
    pub fn sample_rule(
        &mut self,
        schema: &TypeSchema,
        ctx: &mut Context,
        invent: bool,
        max_d: usize,
    ) -> Result<Rule, SampleError> {
        let mut lex = self.0.write().expect("poisoned lexicon");
        lex.sample_rule(schema, ctx, invent, max_d, 0)
    }
    /// Give the log probability of sampling a Term.
    pub fn logprior_term(
        &self,
        term: &Term,
        schema: &TypeSchema,
        ctx: &mut Context,
        invent: bool,
    ) -> Result<f64, SampleError> {
        let lex = self.0.read().expect("poisoned lexicon");
        lex.logprior_term(term, schema, ctx, invent)
    }
    /// Give the log probability of sampling a Rule.
    pub fn logprior_rule(
        &self,
        rule: &Rule,
        schema: &TypeSchema,
        ctx: &mut Context,
        invent: bool,
    ) -> Result<f64, SampleError> {
        let lex = self.0.read().expect("poisoned lexicon");
        lex.logprior_rule(rule, schema, ctx, invent)
    }
    /// Give the log probability of sampling a TRS.
    pub fn logprior_trs(
        &self,
        trs: &TRS,
        schemas: &[TypeSchema],
        p_rule: f64,
        ctx: &mut Context,
        invent: bool,
    ) -> Result<f64, SampleError> {
        let lex = self.0.read().expect("poisoned lexicon");
        lex.logprior_trs(trs, schemas, p_rule, ctx, invent)
    }

    /// merge two `TRS` into a single `TRS`.
    pub fn combine(
        &self,
        rs1: &RewriteSystem,
        rs2: &RewriteSystem,
    ) -> Result<RewriteSystem, TypeError> {
        assert_eq!(rs1.lex, rs2.lex);
        let mut new_rs = rs1.clone();
        // because the lexicon is the same, we only need to transfer rules
        let mut new_rules = rs2.trs.rules.clone();
        new_rs.trs.rules.append(&mut new_rules);
        // update the context
        let mut ctx = Context::default();
        {
            let lex = rs1.lex.0.read().expect("poisoned lexicon");
            lex.infer_trs(&new_rs.trs, &mut ctx)?;
        }
        new_rs.ctx = ctx;
        Ok(new_rs)
    }
}
impl fmt::Debug for Lexicon {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let lex = self.0.read();
        write!(f, "Lexicon({:?})", lex)
    }
}
impl PartialEq for Lexicon {
    fn eq(&self, other: &Lexicon) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Lex {
    ops: Vec<TypeSchema>,
    vars: Vec<TypeSchema>,
    pub(crate) signature: Signature,
}
impl Lex {
    fn free_vars(&self) -> Vec<TypeVar> {
        let vars_fvs = self.vars.iter().flat_map(TypeSchema::free_vars);
        let ops_fvs = self.ops.iter().flat_map(TypeSchema::free_vars);
        vars_fvs.chain(ops_fvs).unique().collect()
    }
    fn free_vars_applied(&self, ctx: &Context) -> Vec<TypeVar> {
        self.free_vars()
            .iter()
            .flat_map(|x| Type::Variable(*x).apply(ctx).vars())
            .unique()
            .collect::<Vec<_>>()
    }
    fn invent_variable(&mut self, tp: &Type) -> Variable {
        let var = self.signature.new_var(None);
        self.vars.push(TypeSchema::Monotype(tp.clone()));
        var
    }
    fn var_tp(&self, v: Variable) -> Result<TypeSchema, TypeError> {
        if let Some(idx) = self.signature.variables().iter().position(|&x| x == v) {
            Ok(self.vars[idx].clone())
        } else {
            Err(TypeError::VarNotFound)
        }
    }
    fn op_tp(&self, o: Operator) -> Result<TypeSchema, TypeError> {
        if let Some(idx) = self.signature.operators().iter().position(|&x| x == o) {
            Ok(self.ops[idx].clone())
        } else {
            Err(TypeError::OpNotFound)
        }
    }
    fn infer_atom(&self, atom: &Atom) -> Result<TypeSchema, TypeError> {
        match *atom {
            Atom::Operator(o) => self.op_tp(o),
            Atom::Variable(v) => self.var_tp(v),
        }
    }
    fn instantiate_atom(&self, atom: &Atom, ctx: &mut Context) -> Result<Type, TypeError> {
        let mut tp = self.infer_atom(atom)?.instantiate_owned(ctx);
        tp.apply_mut(ctx);
        Ok(tp)
    }
    pub fn infer_term(&self, term: &Term, ctx: &mut Context) -> Result<TypeSchema, TypeError> {
        let tp = self.infer_term_internal(term, ctx)?;
        let lex_vars = self.free_vars_applied(ctx);
        Ok(tp.generalize(&lex_vars))
    }
    fn infer_term_internal(&self, term: &Term, ctx: &mut Context) -> Result<Type, TypeError> {
        if let Term::Application { op, ref args } = *term {
            if op.arity(&self.signature) > 0 {
                let head_type = self.instantiate_atom(&Atom::from(op), ctx)?;
                let body_type = {
                    let mut pre_types = Vec::with_capacity(args.len() + 1);
                    for a in args {
                        pre_types.push(self.infer_term_internal(a, ctx)?);
                    }
                    pre_types.push(ctx.new_variable());
                    Type::from(pre_types)
                };
                ctx.unify(&head_type, &body_type)?;
            }
        }
        self.instantiate_atom(&term.head(), ctx)
    }
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
        let lex_vars = self.free_vars_applied(&ctx);
        let rule_schema = lhs_type.apply(ctx).generalize(&lex_vars);
        Ok((rule_schema, lhs_schema, rhs_schemas))
    }
    pub fn infer_trs(&self, trs: &TRS, ctx: &mut Context) -> Result<(), TypeError> {
        // TODO: Right now, this assumes the variables already exist in the signature. Is that sensible?
        for rule in &trs.rules {
            self.infer_rule(rule, ctx)?;
        }
        Ok(())
    }

    pub fn sample_term(
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
    pub fn sample_rule(
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
            let lhs = self.sample_term(schema, ctx, invent, max_d, d)?;
            let rhs = self.sample_term(schema, ctx, false, max_d, d)?;
            if let Some(rule) = Rule::new(lhs, vec![rhs]) {
                return Ok(rule);
            } else {
                *self = orig_self.clone();
                *ctx = orig_ctx.clone();
            }
        }
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
                    let result = self.sample_term(&arg_schema, ctx, invent, max_d, d_i)
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

    fn logprior_term(
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
    fn logprior_rule(
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
    fn logprior_trs(
        &self,
        trs: &TRS,
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
}
impl GP for Lexicon {
    type Expression = RewriteSystem;
    type Params = GeneticParams;
    fn genesis<R: Rng>(
        &self,
        params: &Self::Params,
        _rng: &mut R,
        pop_size: usize,
        _tp: &TypeSchema,
    ) -> Vec<Self::Expression> {
        repeat(params.h0.clone()).take(pop_size).collect()
    }
    fn mutate<R: Rng>(
        &self,
        _params: &Self::Params,
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
        let trs = self.combine(parent1, parent2)
            .expect("poorly-typed TRS in crossover");
        let trss = repeat(trs.clone()).take(params.n_crosses).map(|mut x| {
            x.trs.rules = x.trs
                .rules
                .into_iter()
                .filter(|_| rng.gen_bool(0.5))
                .collect();
            x
        });
        once(trs).chain(trss).collect()
    }
}
