use super::trace::Trace;
use super::utils::{log_n_of, logsumexp};
///! Hindley-Milner Typing for First-Order Term Rewriting Systems (no abstraction)
///!
///! Much thanks to:
///! - https://github.com/rob-smallshire/hindley-milner-python
///! - https://en.wikipedia.org/wiki/Hindley%E2%80%93Milner_type_system
///! - (TAPL; Pierce, 2002, ch. 22)
use itertools::Itertools;
use polytype::{Context, Type, TypeSchema, Variable as TypeVar};
use rand::{thread_rng, Rng};
use std::f64::NEG_INFINITY;
use std::iter::repeat;
use term_rewriting::{Atom, Operator, Rule, Signature, Term, Variable, TRS};

#[derive(Debug, Clone, Copy)]
pub struct TypeError;

#[derive(Debug, Clone)]
pub struct SampleError(String);

/// A first-order [Term Rewriting System][0] (TRS) with a [Hindley-Milner type system][1].
///
/// [0]: https://wikipedia.org/wiki/Hindley–Milner_type_system
///      "Wikipedia - Hindley-Milner Type System"
/// [1]: https://en.wikipedia.org/wiki/Rewriting#Term_rewriting_systems
///      "Wikipedia - Term Rewriting Systems"
#[derive(Debug, PartialEq, Clone)]
pub struct HMTRS {
    // TODO: may also want to track background knowledge here.
    ops: Vec<TypeSchema>,
    vars: Vec<TypeSchema>,
    pub signature: Signature,
    pub trs: TRS,
    pub ctx: Context,
}
impl HMTRS {
    /// the size of the HMTRS (the sum over the size of the rules in the [`TRS`])
    ///
    /// [`TRS`]: ../../term_rewriting/struct.TRS.html
    pub fn size(&self) -> usize {
        self.trs.size()
    }
    /// Convert the `TRS` to a `String`.
    pub fn display(&self) -> String {
        self.trs.display(&self.signature)
    }
    /// All the free variables in the lexicon.
    pub fn free_vars(&self) -> Vec<TypeVar> {
        let vars_fvs = self.vars.iter().flat_map(|x| x.free_vars());
        let ops_fvs = self.ops.iter().flat_map(|x| x.free_vars());
        vars_fvs.chain(ops_fvs).unique().collect()
    }
    /// Infer the `Type` of a `Variable`.
    ///
    /// No [`Context`] is necessary for inference here; `self` either knows the correct [`TypeSchema`] or generates a [`TypeError`].
    ///
    /// [`Context`]: ../../polytype/struct.Context.html
    /// [`TypeSchema`]: ../../polytype/struct.TypeSchema.html
    /// [`TypeError`]: struct.TypeError.html
    pub fn infer_var(&self, v: &Variable) -> Result<TypeSchema, TypeError> {
        if let Some(idx) = self.signature.variables().iter().position(|x| x == v) {
            Ok(self.vars[idx].clone())
        } else {
            Err(TypeError)
        }
    }
    /// Infer the `Type` of a `Variable`.
    ///
    /// No [`Context`] is necessary for inference here; `self` either knows the correct [`TypeSchema`] or generates a [`TypeError`].
    ///
    /// [`Context`]: ../../polytype/struct.Context.html
    /// [`TypeSchema`]: ../../polytype/struct.TypeSchema.html
    /// [`TypeError`]: struct.TypeError.html
    pub fn infer_op(&self, o: &Operator) -> Result<TypeSchema, TypeError> {
        if let Some(idx) = self.signature.operators().iter().position(|x| x == o) {
            Ok(self.ops[idx].clone())
        } else {
            Err(TypeError)
        }
    }
    /// Infer the [`TypeSchema`] of a [`Term`] and the corresponding [`Context`] or generate a [`TypeError`].
    ///
    /// [`Context`]: ../../polytype/struct.Context.html
    /// [`Term`]: ../../term_rewriting/enum.Term.html
    /// [`TypeSchema`]: ../../polytype/struct.TypeSchema.html
    /// [`TypeError`]: struct.TypeError.html
    pub fn infer_term(&self, term: &Term, ctx: &mut Context) -> Result<TypeSchema, TypeError> {
        let tp = self.infer_internal(term, ctx)?;
        // Get the variables bound by the lexicon
        let bound_vs = self.free_vars()
            .iter()
            .flat_map(|x| {
                ctx.substitution()
                    .get(x)
                    .unwrap_or(&Type::Variable(*x))
                    .vars()
            })
            .unique()
            .collect::<Vec<u16>>();
        Ok(tp.generalize(&bound_vs))
    }
    // half the internal recursive workhorse of type inference.
    fn infer_internal(&self, term: &Term, ctx: &mut Context) -> Result<Type, TypeError> {
        match term {
            Term::Application { op, .. } if op.arity(&self.signature) == 0 => {
                Ok(self.infer_op(op)?.instantiate(ctx).apply(ctx))
            }
            Term::Variable(v) => Ok(self.infer_var(v)?.instantiate(ctx).apply(ctx)),
            Term::Application { op, args } => {
                let body_type = self.infer_args(args, ctx)?;
                let head_type = self.infer_op(op)?.instantiate(ctx).apply(ctx);
                if ctx.unify(&head_type, &body_type).is_ok() {
                    Ok(self.infer_op(op)?.instantiate(ctx).apply(ctx))
                } else {
                    Err(TypeError)
                }
            }
        }
    }
    // half the internal recursive workhorse of type inference.
    fn infer_args(&self, args: &[Term], ctx: &mut Context) -> Result<Type, TypeError> {
        let mut pre_types = vec![];
        for a in args {
            pre_types.push(self.infer_internal(a, ctx)?);
        }
        pre_types.push(ctx.new_variable());
        Ok(Type::from(pre_types))
    }
    /// Infer the [`TypeSchema`] of a [`Rule`] and the corresponding [`Context`] or generate a [`TypeError`].
    ///
    /// [`Context`]: ../../polytype/struct.Context.html
    /// [`Rule`]: ../../term_rewriting/struct.Rule.html
    /// [`TypeSchema`]: ../../polytype/struct.TypeSchema.html
    /// [`TypeError`]: struct.TypeError.html
    pub fn infer_rule(
        &self,
        r: &Rule,
        ctx: &mut Context,
    ) -> Result<(TypeSchema, TypeSchema, Vec<TypeSchema>), TypeError> {
        let lhs_schema = self.infer_term(&r.lhs, ctx)?;
        let lhs_type = lhs_schema.instantiate(ctx);
        let mut rhs_types = vec![];
        let mut rhs_schemas = vec![];
        for rhs in &r.rhs {
            let rhs_schema = self.infer_term(&rhs, ctx)?;
            rhs_types.push(rhs_schema.instantiate(ctx));
            rhs_schemas.push(rhs_schema);
        }
        for rhs_type in rhs_types {
            ctx.unify(&lhs_type, &rhs_type).or(Err(TypeError))?;
        }
        // Get the variables bound by the lexicon
        let bound_vs = self.free_vars()
            .iter()
            .flat_map(|x| {
                ctx.substitution()
                    .get(x)
                    .unwrap_or(&Type::Variable(*x))
                    .vars()
            })
            .unique()
            .collect::<Vec<u16>>();
        let rule_schema = lhs_type.apply(ctx).generalize(&bound_vs);
        Ok((rule_schema, lhs_schema, rhs_schemas))
    }
    /// Infer the [`Context`] allowing every [`Rule`] in a [`TRS`] to typecheck or generate a [`TypeError`].
    ///
    /// [`Context`]: ../../polytype/struct.Context.html
    /// [`Rule`]: ../../term_rewriting/struct.Rule.html
    /// [`TRS`]: ../../term_rewriting/struct.TRS.html
    /// [`TypeError`]: struct.TypeError.html
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
        d: usize,
    ) -> Result<Term, SampleError> {
        // fail if we've gone too deep
        if d > max_d {
            return Err(SampleError(format!("depth bound {} < {}", max_d, d)));
        }

        let tp = schema.instantiate(ctx);

        let mut options: Vec<Option<Atom>> = self.signature.atoms().into_iter().map(Some).collect();
        if invent {
            options.push(None);
        }
        let mut rng = thread_rng();
        rng.shuffle(&mut options);
        for option in options {
            let atom = option.unwrap_or_else(|| Atom::Variable(self.invent_variable(&tp)));
            let body_types = self.check_option(&atom, &tp, ctx)?;
            let result = self.try_option(&atom, body_types, ctx, invent, max_d, d);
            if result.is_ok() {
                return result;
            }
        }
        Err(SampleError("failed to sample term".to_string()))
    }
    pub fn lp_term(
        &self,
        term: &Term,
        schema: &TypeSchema,
        ctx: &mut Context,
        invent: bool,
    ) -> Result<f64, SampleError> {
        // instantiate the typeschema
        let tp = schema.instantiate(ctx);
        // setup the existing options
        let mut options = vec![];
        let atoms = self.signature.atoms();
        for atom in &atoms {
            if let Ok(bts) = self.check_option(atom, &tp, ctx) {
                options.push((Some(*atom), bts))
            }
        }
        // add a variable if needed
        if invent {
            if let Term::Variable(v) = *term {
                if !atoms.contains(&Atom::Variable(v)) {
                    options.push((Some(Atom::Variable(v)), vec![tp]));
                }
            } else {
                options.push((None, vec![tp]));
            }
        }
        // compute the log probability of selecting this head
        let mut lp = log_n_of(&options, 1, 0.0);
        // filter the matches
        let matches: Vec<(Option<Atom>, Vec<Type>)> = options
            .into_iter()
            .filter(|(o, _)| o == &Some(term.head()))
            .collect();
        // process the match
        match matches.get(0) {
            Some((Some(Atom::Variable(_)), _)) => Ok(lp),
            Some((Some(Atom::Operator(_)), ref bts)) => {
                for (subterm, body_type) in term.args().iter().zip(bts) {
                    let subtype = body_type.apply(ctx);
                    let subschema = TypeSchema::Monotype(subtype.clone());
                    let tmp_lp = self.lp_term(subterm, &subschema, ctx, invent)?;
                    lp += tmp_lp;
                    let final_schema = self.infer_term(subterm, ctx)
                        .or_else(|_| Err(SampleError("untypable".to_string())))?;
                    let final_type = final_schema.instantiate(ctx);
                    if ctx.unify(&subtype, &final_type).is_err() {
                        return Ok(NEG_INFINITY);
                    }
                }
                Ok(lp)
            }
            Some(_) => Err(SampleError("Should never happen -- FIXME!".to_string())),
            None => Ok(NEG_INFINITY),
        }
    }
    /// Sample a Rule.
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
    /// Give the log probability of sampling a Rule.
    pub fn lp_rule(
        &self,
        rule: &Rule,
        schema: &TypeSchema,
        ctx: &mut Context,
        invent: bool,
    ) -> Result<f64, SampleError> {
        let mut lp = 0.0;
        let lp_lhs = self.lp_term(&rule.lhs, schema, ctx, invent)?;
        for rhs in &rule.rhs {
            let tmp_lp = self.lp_term(&rhs, schema, ctx, false)?;
            lp += tmp_lp + lp_lhs;
        }
        Ok(lp)
    }
    /// Give the log probability of sampling a TRS.
    pub fn lp_trs(
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
                let tmp_lp = self.lp_rule(&rule, schema, ctx, invent)?;
                rule_ps.push(tmp_lp);
            }
            p_rules += logsumexp(&rule_ps);
        }
        Ok(p_n_rules + p_rules)
    }
    fn fast_update(&self, atom: &Atom, ctx: &mut Context) -> Result<Type, TypeError> {
        let schema = match atom {
            Atom::Operator(o) => self.infer_op(o)?,
            Atom::Variable(v) => self.infer_var(v)?,
        };
        Ok(schema.instantiate(ctx).apply(ctx))
    }
    fn check_option(
        &self,
        atom: &Atom,
        tp: &Type,
        ctx: &mut Context,
    ) -> Result<Vec<Type>, SampleError> {
        let lexicon_type = self.fast_update(atom, ctx)
            .or_else(|_| Err(SampleError("could not find item in lexicon".to_string())))?;
        match atom {
            Atom::Operator(o) => {
                let mut arg_types: Vec<Type> = repeat(0)
                    .take(o.arity(&self.signature) as usize)
                    .map(|_| ctx.new_variable())
                    .collect();
                let result_type = ctx.new_variable();
                arg_types.push(result_type);
                let structural_type = Type::from(arg_types.clone());

                ctx.unify(&lexicon_type, &structural_type)
                    .or_else(|_| Err(SampleError("failed unification".to_string())))?;

                let result_type = arg_types.pop().unwrap();
                ctx.unify(&result_type, tp)
                    .or_else(|_| Err(SampleError("failed unification".to_string())))?;
                Ok(arg_types)
            }
            Atom::Variable(_) => {
                ctx.unify(&lexicon_type, tp)
                    .or_else(|_| Err(SampleError("failed unification".to_string())))?;
                Ok(vec![])
            }
        }
    }
    fn try_option(
        &mut self,
        atom: &Atom,
        body_types: Vec<Type>,
        ctx: &mut Context,
        invent: bool,
        max_d: usize,
        d: usize,
    ) -> Result<Term, SampleError> {
        match atom {
            Atom::Variable(v) => Ok(Term::Variable(*v)),
            Atom::Operator(o) => {
                let mut subterms = vec![];
                let orig_ctx = ctx.clone();
                for (i, bt) in body_types.into_iter().enumerate() {
                    let subtype = bt.apply(ctx);
                    let d_i = (d + 1) * ((i == 0) as usize);
                    let result =
                        self.sample_term(&TypeSchema::Monotype(bt), ctx, invent, max_d, d_i)
                            .or_else(|_| Err(SampleError("cannot sample subterm".to_string())))
                            .and_then(|subterm| {
                                self.infer_term(&subterm, ctx)
                                    .map(|schema| (subterm, schema))
                                    .or_else(|_| Err(SampleError("untypable term".to_string())))
                            })
                            .and_then(|(subterm, schema)| {
                                let tp = schema.instantiate(ctx);
                                ctx.unify(&subtype, &tp)
                                    .map(|_| subterm)
                                    .or_else(|_| Err(SampleError("type mismatch".to_string())))
                            });
                    if let Ok(subterm) = result {
                        subterms.push(subterm);
                    } else {
                        *ctx = orig_ctx;
                        return result;
                    }
                }
                Ok(Term::Application {
                    op: *o,
                    args: subterms,
                })
            }
        }
    }
    /// Create a brand-new variable.
    fn invent_variable(&mut self, tp: &Type) -> Variable {
        let var = self.signature.new_var(None);
        self.vars.push(TypeSchema::Monotype(tp.clone()));
        var
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

    pub fn pseudo_log_prior(&self, temp: f64, prior_temp: f64) -> f64 {
        let raw_prior = -(self.size() as f64);
        raw_prior / ((temp + 1.0) * prior_temp)
    }

    pub fn log_likelihood(&self, data: &[Rule], p_partial: f64, temp: f64, ll_temp: f64) -> f64 {
        data.iter()
            .map(|x| self.single_log_likelihood(x, p_partial, temp) / ll_temp)
            .sum()
    }

    pub fn single_log_likelihood(&self, datum: &Rule, p_partial: f64, temp: f64) -> f64 {
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
}
