use itertools::Itertools;
use polytype::{Context as TypeContext, Type, TypeSchema, Variable as TypeVar};
use rand::Rng;
use std::collections::HashMap;
use std::f64::NEG_INFINITY;
use std::fmt;
use std::iter;
use std::sync::{Arc, RwLock};
use term_rewriting::{
    Atom, Context, Operator, Place, Rule, RuleContext, Signature, Term, Variable, TRS as UntypedTRS,
};

use super::{SampleError, TypeError, TRS};
use utils::{logsumexp, weighted_permutation};
use GP;

#[derive(Debug, Clone)]
/// Parameters for [`Lexicon`] genetic programming ([`GP`]).
///
/// [`Lexicon`]: struct.Lexicon.html
/// [`GP`]: ../trait.GP.html
pub struct GeneticParams {
    /// The number of hypotheses crossover should generate.
    pub n_crosses: usize,
    /// The maximum number of non-progressing recursive steps sampling can make without failing.
    pub max_sample_depth: usize,
    /// The probability of adding (vs. deleting) a rule during mutation.
    pub p_add: f64,
    /// The probability of keeping a rule during crossover.
    pub p_keep: f64,
    /// Rule templates to use when sampling rules.
    pub templates: Vec<RuleContext>,
    /// The weight to assign variables, constants, and non-constant operators, respectively.
    pub atom_weights: (f64, f64, f64),
}

/// (representation) Manages the syntax of a term rewriting system.
#[derive(Clone)]
pub struct Lexicon(pub(crate) Arc<RwLock<Lex>>);
impl Lexicon {
    /// Construct a `Lexicon` with only background [`term_rewriting::Operator`]s.
    ///
    /// # Example
    ///
    /// See [`polytype::ptp`] for details on constructing [`polytype::TypeSchema`]s.
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # use programinduction::trs::Lexicon;
    /// # fn main() {
    /// let operators = vec![
    ///     (2, Some("PLUS".to_string()), ptp![@arrow[tp!(int), tp!(int), tp!(int)]]),
    ///     (1, Some("SUCC".to_string()), ptp![@arrow[tp!(int), tp!(int)]]),
    ///     (0, Some("ZERO".to_string()), ptp![int]),
    /// ];
    /// let deterministic = false;
    ///
    /// let lexicon = Lexicon::new(operators, deterministic);
    /// # }
    /// ```
    ///
    /// [`polytype::ptp`]: https://docs.rs/polytype/~6.0/polytype/macro.ptp.html
    /// [`polytype::TypeSchema`]: https://docs.rs/polytype/~6.0/polytype/enum.TypeSchema.html
    /// [`term_rewriting::Operator`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Operator.html
    pub fn new(operators: Vec<(u32, Option<String>, TypeSchema)>, deterministic: bool) -> Lexicon {
        let mut signature = Signature::default();
        let mut ops = Vec::with_capacity(operators.len());
        for (id, name, tp) in operators {
            signature.new_op(id, name);
            ops.push(tp)
        }
        Lexicon(Arc::new(RwLock::new(Lex {
            ops,
            vars: Vec::new(),
            signature,
            background: vec![],
            deterministic,
        })))
    }
    /// Construct a `Lexicon` with a set of background
    /// [`term_rewriting::Operator`]s, [`term_rewriting::Variable`]s, and
    /// [`term_rewriting::Rule`]s.
    ///
    /// # Example
    ///
    /// See [`polytype::ptp`] for details on constructing [`polytype::TypeSchema`]s.
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate term_rewriting;
    /// # use programinduction::trs::Lexicon;
    /// # use term_rewriting::{Signature, parse_rule};
    /// # fn main() {
    /// let mut sig = Signature::default();
    ///
    /// let vars = vec![];
    ///
    /// let mut ops = vec![];
    /// sig.new_op(2, Some("PLUS".to_string()));
    /// ops.push(ptp![@arrow[tp!(int), tp!(int), tp!(int)]]);
    /// sig.new_op(1, Some("SUCC".to_string()));
    /// ops.push(ptp![@arrow[tp!(int), tp!(int)]]);
    /// sig.new_op(0, Some("ZERO".to_string()));
    /// ops.push(ptp![int]);
    ///
    /// let background = vec![
    ///     parse_rule(&mut sig, "PLUS(x_ ZERO) = x_").expect("parsed rule"),
    ///     parse_rule(&mut sig, "PLUS(x_ SUCC(y_)) = SUCC(PLUS(x_ y_))").expect("parsed rule"),
    /// ];
    ///
    /// let deterministic = false;
    ///
    /// let lexicon = Lexicon::from_signature(sig, ops, vars, background, deterministic);
    /// # }
    /// ```
    ///
    /// [`polytype::ptp`]: https://docs.rs/polytype/~6.0/polytype/macro.ptp.html
    /// [`polytype::TypeSchema`]: https://docs.rs/polytype/~6.0/polytype/enum.TypeSchema.html
    /// [`term_rewriting::Operator`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Operator.html
    /// [`term_rewriting::Rule`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Rule.html
    /// [`term_rewriting::Variable`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.Variable.html
    pub fn from_signature(
        signature: Signature,
        ops: Vec<TypeSchema>,
        vars: Vec<TypeSchema>,
        background: Vec<Rule>,
        deterministic: bool,
    ) -> Lexicon {
        Lexicon(Arc::new(RwLock::new(Lex {
            ops,
            vars,
            signature,
            background,
            deterministic,
        })))
    }
    /// All the free type variables in the lexicon.
    pub fn free_vars(&self) -> Vec<TypeVar> {
        self.0.read().expect("poisoned lexicon").free_vars()
    }
    /// Infer the [`polytype::TypeSchema`] associated with a [`term_rewriting::Context`].
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate term_rewriting;
    /// # use programinduction::trs::Lexicon;
    /// # use term_rewriting::{Context, Signature, parse_rule};
    /// # use polytype::Context as TypeContext;
    /// # fn main() {
    /// let mut sig = Signature::default();
    ///
    /// let vars = vec![];
    ///
    /// let mut ops = vec![];
    /// sig.new_op(2, Some("PLUS".to_string()));
    /// ops.push(ptp![@arrow[tp!(int), tp!(int), tp!(int)]]);
    /// let succ = sig.new_op(1, Some("SUCC".to_string()));
    /// ops.push(ptp![@arrow[tp!(int), tp!(int)]]);
    /// sig.new_op(0, Some("ZERO".to_string()));
    /// ops.push(ptp![int]);
    ///
    /// let background = vec![];
    ///
    /// let deterministic = false;
    ///
    /// let lexicon = Lexicon::from_signature(sig, ops, vars, background, deterministic);
    ///
    /// let context = Context::Application {
    ///     op: succ,
    ///     args: vec![Context::Hole]
    /// };
    /// let mut ctx = TypeContext::default();
    ///
    /// let inferred_schema = lexicon.infer_context(&context, &mut ctx).unwrap();
    ///
    /// assert_eq!(inferred_schema, ptp![int]);
    /// # }
    /// ```
    ///
    /// [`polytype::TypeSchema`]: https://docs.rs/polytype/~6.0/polytype/enum.TypeSchema.html
    /// [`term_rewriting::Context`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/enum.Context.html
    pub fn infer_context(
        &self,
        context: &Context,
        ctx: &mut TypeContext,
    ) -> Result<TypeSchema, TypeError> {
        self.0
            .write()
            .expect("poisoned lexicon")
            .infer_context(context, ctx)
    }
    /// Infer the `TypeSchema` associated with a `RuleContext`.
    pub fn infer_rule_context(
        &self,
        context: &RuleContext,
        ctx: &mut TypeContext,
    ) -> Result<TypeSchema, TypeError> {
        self.0
            .write()
            .expect("poisoned lexicon")
            .infer_rule_context(context, ctx)
    }
    /// Sample a [`term_rewriting::Term`].
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # use programinduction::trs::Lexicon;
    /// # use polytype::Context as TypeContext;
    /// # fn main() {
    /// let operators = vec![
    ///     (2, Some("PLUS".to_string()), ptp![@arrow[tp!(int), tp!(int), tp!(int)]]),
    ///     (1, Some("SUCC".to_string()), ptp![@arrow[tp!(int), tp!(int)]]),
    ///     (0, Some("ZERO".to_string()), ptp![int]),
    /// ];
    /// let deterministic = false;
    /// let mut lexicon = Lexicon::new(operators, deterministic);
    ///
    /// let schema = ptp![int];
    /// let mut ctx = TypeContext::default();
    /// let invent = true;
    /// let atom_weights = (0.5, 0.25, 0.25);
    /// let max_d = 4;
    ///
    /// let term = lexicon.sample_term(&schema, &mut ctx, atom_weights, invent, max_d).unwrap();
    /// # }
    /// ```
    ///
    /// [`term_rewriting::Term`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/enum.Term.html
    pub fn sample_term(
        &mut self,
        schema: &TypeSchema,
        ctx: &mut TypeContext,
        atom_weights: (f64, f64, f64),
        invent: bool,
        max_d: usize,
    ) -> Result<Term, SampleError> {
        self.0.write().expect("poisoned lexicon").sample_term(
            schema,
            ctx,
            atom_weights,
            invent,
            max_d,
            0,
        )
    }
    /// Sample a `Term` conditioned on a `Context` rather than a `TypeSchema`.
    pub fn sample_term_from_context(
        &mut self,
        context: &Context,
        ctx: &mut TypeContext,
        atom_weights: (f64, f64, f64),
        invent: bool,
        max_d: usize,
    ) -> Result<Term, SampleError> {
        self.0
            .write()
            .expect("poisoned lexicon")
            .sample_term_from_context(context, ctx, atom_weights, invent, max_d, 0)
    }
    /// Sample a `Rule`.
    pub fn sample_rule(
        &mut self,
        schema: &TypeSchema,
        ctx: &mut TypeContext,
        atom_weights: (f64, f64, f64),
        invent: bool,
        max_d: usize,
    ) -> Result<Rule, SampleError> {
        self.0.write().expect("poisoned lexicon").sample_rule(
            schema,
            ctx,
            atom_weights,
            invent,
            max_d,
            0,
        )
    }
    /// Sample a `Rule` conditioned on a `Context` rather than a `TypeSchema`.
    pub fn sample_rule_from_context(
        &mut self,
        context: &RuleContext,
        ctx: &mut TypeContext,
        atom_weights: (f64, f64, f64),
        invent: bool,
        max_d: usize,
    ) -> Result<Rule, SampleError> {
        self.0
            .write()
            .expect("poisoned lexicon")
            .sample_rule_from_context(context, ctx, atom_weights, invent, max_d, 0)
    }
    /// Give the log probability of sampling a Term.
    pub fn logprior_term(
        &self,
        term: &Term,
        schema: &TypeSchema,
        ctx: &mut TypeContext,
        atom_weights: (f64, f64, f64),
        invent: bool,
    ) -> Result<f64, SampleError> {
        self.0.read().expect("poisoned lexicon").logprior_term(
            term,
            schema,
            ctx,
            atom_weights,
            invent,
        )
    }
    /// Give the log probability of sampling a Rule.
    pub fn logprior_rule(
        &self,
        rule: &Rule,
        schema: &TypeSchema,
        ctx: &mut TypeContext,
        atom_weights: (f64, f64, f64),
        invent: bool,
    ) -> Result<f64, SampleError> {
        self.0.read().expect("poisoned lexicon").logprior_rule(
            rule,
            schema,
            ctx,
            atom_weights,
            invent,
        )
    }
    /// Give the log probability of sampling a TRS.
    pub fn logprior_utrs(
        &self,
        utrs: &UntypedTRS,
        schemas: &[TypeSchema],
        p_rule: f64,
        ctx: &mut TypeContext,
        atom_weights: (f64, f64, f64),
        invent: bool,
    ) -> Result<f64, SampleError> {
        self.0.read().expect("poisoned lexicon").logprior_utrs(
            utrs,
            schemas,
            p_rule,
            ctx,
            atom_weights,
            invent,
        )
    }

    /// merge two `TRS` into a single `TRS`.
    pub fn combine(&self, trs1: &TRS, trs2: &TRS) -> Result<TRS, TypeError> {
        assert_eq!(trs1.lex, trs2.lex);
        let background_size = trs1
            .lex
            .0
            .read()
            .expect("poisoned lexicon")
            .background
            .len();
        let mut rules1 = trs1.utrs.rules[..trs1.utrs.len() - background_size].to_vec();
        let mut rules2 = trs2.utrs.rules.clone(); // includes background
        rules1.append(&mut rules2);
        TRS::new(&trs1.lex, rules1)
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
    pub(crate) background: Vec<Rule>,
    /// If `true`, then the `TRS`s should be deterministic.
    pub(crate) deterministic: bool,
}
impl Lex {
    fn free_vars(&self) -> Vec<TypeVar> {
        let vars_fvs = self.vars.iter().flat_map(TypeSchema::free_vars);
        let ops_fvs = self.ops.iter().flat_map(TypeSchema::free_vars);
        vars_fvs.chain(ops_fvs).unique().collect()
    }
    fn free_vars_applied(&self, ctx: &TypeContext) -> Vec<TypeVar> {
        self.free_vars()
            .into_iter()
            .flat_map(|x| Type::Variable(x).apply(ctx).vars())
            .unique()
            .collect::<Vec<_>>()
    }
    fn invent_variable(&mut self, tp: &Type) -> Variable {
        let var = self.signature.new_var(None);
        self.vars.push(TypeSchema::Monotype(tp.clone()));
        var
    }
    fn fit_atom(
        &self,
        atom: &Atom,
        tp: &Type,
        ctx: &mut TypeContext,
    ) -> Result<Vec<Type>, SampleError> {
        let atom_tp = self.instantiate_atom(atom, ctx)?;
        ctx.unify(atom_tp.returns().or_else(|| Some(&atom_tp)).unwrap(), tp)?;
        Ok(atom_tp
            .args()
            .map(|o| o.into_iter().cloned().collect())
            .unwrap_or_else(Vec::new))
    }
    #[cfg_attr(feature = "cargo-clippy", allow(too_many_arguments))]
    fn place_atom(
        &mut self,
        atom: &Atom,
        arg_types: Vec<Type>,
        ctx: &mut TypeContext,
        atom_weights: (f64, f64, f64),
        invent: bool,
        max_d: usize,
        d: usize,
        vars: &mut Vec<Variable>,
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
                    let result =
                        self.sample_term_internal(
                            &arg_schema,
                            ctx,
                            atom_weights,
                            invent,
                            max_d,
                            d_i,
                            vars,
                        ).map_err(|_| SampleError::Subterm)
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
    fn instantiate_atom(&self, atom: &Atom, ctx: &mut TypeContext) -> Result<Type, TypeError> {
        let mut tp = self.infer_atom(atom)?.instantiate_owned(ctx);
        tp.apply_mut(ctx);
        Ok(tp)
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
    pub fn infer_term(&self, term: &Term, ctx: &mut TypeContext) -> Result<TypeSchema, TypeError> {
        let tp = self.infer_term_internal(term, ctx)?;
        let lex_vars = self.free_vars_applied(ctx);
        Ok(tp.apply(ctx).generalize(&lex_vars))
    }
    fn infer_term_internal(&self, term: &Term, ctx: &mut TypeContext) -> Result<Type, TypeError> {
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
                return Ok(head_type.returns().unwrap_or(&head_type).apply(ctx));
            }
        }
        self.instantiate_atom(&term.head(), ctx)
    }
    pub fn infer_context(
        &self,
        context: &Context,
        ctx: &mut TypeContext,
    ) -> Result<TypeSchema, TypeError> {
        let tp = self.infer_context_internal(context, ctx, vec![], &mut HashMap::new())?;
        let lex_vars = self.free_vars_applied(ctx);
        Ok(tp.apply(ctx).generalize(&lex_vars))
    }
    fn infer_context_internal(
        &self,
        context: &Context,
        ctx: &mut TypeContext,
        place: Place,
        tps: &mut HashMap<Place, Type>,
    ) -> Result<Type, TypeError> {
        let tp = match *context {
            Context::Hole => ctx.new_variable(),
            Context::Variable(v) => self.instantiate_atom(&Atom::from(v), ctx)?,
            Context::Application { op, ref args } => {
                let head_type = self.instantiate_atom(&Atom::from(op), ctx)?;
                let body_type = {
                    let mut pre_types = Vec::with_capacity(args.len() + 1);
                    for (i, a) in args.iter().enumerate() {
                        let mut new_place = place.clone();
                        new_place.push(i);
                        pre_types.push(self.infer_context_internal(a, ctx, new_place, tps)?);
                    }
                    pre_types.push(ctx.new_variable());
                    Type::from(pre_types)
                };
                ctx.unify(&head_type, &body_type)?;
                head_type.returns().unwrap_or(&head_type).apply(ctx)
            }
        };
        tps.insert(place, tp.clone());
        Ok(tp)
    }
    pub fn infer_rule(
        &self,
        r: &Rule,
        ctx: &mut TypeContext,
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
    pub fn infer_rule_context(
        &self,
        context: &RuleContext,
        ctx: &mut TypeContext,
    ) -> Result<TypeSchema, TypeError> {
        let tp = self.infer_rule_context_internal(context, ctx, &mut HashMap::new())?;
        let lex_vars = self.free_vars_applied(&ctx);
        Ok(tp.apply(ctx).generalize(&lex_vars))
    }
    fn infer_rule_context_internal(
        &self,
        context: &RuleContext,
        ctx: &mut TypeContext,
        tps: &mut HashMap<Place, Type>,
    ) -> Result<Type, TypeError> {
        let lhs_type = self.infer_context_internal(&context.lhs, ctx, vec![0], tps)?;
        let rhs_types = context
            .rhs
            .iter()
            .enumerate()
            .map(|(i, rhs)| self.infer_context_internal(&rhs, ctx, vec![i + 1], tps))
            .collect::<Result<Vec<Type>, _>>()?;
        // unify to introduce rule-level constraints
        for rhs_type in rhs_types {
            ctx.unify(&lhs_type, &rhs_type)?;
        }
        Ok(lhs_type.apply(ctx))
    }
    pub fn infer_utrs(&self, utrs: &UntypedTRS, ctx: &mut TypeContext) -> Result<(), TypeError> {
        // TODO: we assume the variables already exist in the signature. Is that sensible?
        for rule in &utrs.rules {
            self.infer_rule(rule, ctx)?;
        }
        Ok(())
    }

    pub fn sample_term(
        &mut self,
        schema: &TypeSchema,
        ctx: &mut TypeContext,
        atom_weights: (f64, f64, f64),
        invent: bool,
        max_d: usize,
        d: usize,
    ) -> Result<Term, SampleError> {
        self.sample_term_internal(schema, ctx, atom_weights, invent, max_d, d, &mut vec![])
    }
    #[cfg_attr(feature = "cargo-clippy", allow(too_many_arguments))]
    pub fn sample_term_internal(
        &mut self,
        schema: &TypeSchema,
        ctx: &mut TypeContext,
        atom_weights: (f64, f64, f64),
        invent: bool,
        max_d: usize,
        d: usize,
        vars: &mut Vec<Variable>,
    ) -> Result<Term, SampleError> {
        if d > max_d {
            return Err(SampleError::DepthExceeded(d, max_d));
        }

        let tp = schema.instantiate(ctx);

        for option in self.prepare_options(vars, atom_weights, invent) {
            if let Some(atom) = option {
                if let Ok(arg_types) = self.fit_atom(&atom, &tp, ctx) {
                    let result = self.place_atom(
                        &atom,
                        arg_types,
                        ctx,
                        atom_weights,
                        invent,
                        max_d,
                        d,
                        vars,
                    );
                    if result.is_ok() {
                        return result;
                    }
                }
            } else {
                let new_var = self.invent_variable(&tp);
                vars.push(new_var);
                return Ok(Term::Variable(new_var));
            }
        }
        Err(SampleError::OptionsExhausted)
    }
    fn prepare_options(
        &self,
        vars: &mut Vec<Variable>,
        (vw, cw, ow): (f64, f64, f64),
        invent: bool,
    ) -> Vec<Option<Atom>> {
        // create various types of options
        let mut var_options = if invent { vec![None] } else { vec![] };
        var_options.extend(vars.to_vec().into_iter().map(|v| Some(Atom::Variable(v))));
        let (const_options, op_options): (Vec<_>, Vec<_>) = self
            .signature
            .operators()
            .into_iter()
            .partition(|o| o.arity(&self.signature) == 0);
        let const_options = const_options.into_iter().map(|o| Some(Atom::Operator(o)));
        let op_options = op_options.into_iter().map(|o| Some(Atom::Operator(o)));

        // combine into a single list
        let weights: Vec<_> = iter::repeat(vw)
            .take(var_options.len())
            .chain(iter::repeat(cw).take(const_options.len()))
            .chain(iter::repeat(ow).take(op_options.len()))
            .collect();
        let options: Vec<_> = var_options
            .into_iter()
            .chain(const_options)
            .chain(op_options)
            .collect();
        weighted_permutation(&options, &weights)
    }
    pub fn sample_term_from_context(
        &mut self,
        context: &Context,
        ctx: &mut TypeContext,
        atom_weights: (f64, f64, f64),
        invent: bool,
        max_d: usize,
        d: usize,
    ) -> Result<Term, SampleError> {
        let mut map = HashMap::new();
        let context = context.clone();
        let hole_places = context.holes();
        self.infer_context_internal(&context, ctx, vec![], &mut map)?;
        let lex_vars = self.free_vars_applied(&ctx);
        let mut context_vars = context.variables();
        for p in &hole_places {
            let schema = &map[p].apply(ctx).generalize(&lex_vars);
            let subterm = self.sample_term_internal(
                &schema,
                ctx,
                atom_weights,
                invent,
                max_d,
                d,
                &mut context_vars,
            )?;
            context.replace(&p, Context::from(subterm));
        }
        context.to_term().or(Err(SampleError::Subterm))
    }
    pub fn sample_rule(
        &mut self,
        schema: &TypeSchema,
        ctx: &mut TypeContext,
        atom_weights: (f64, f64, f64),
        invent: bool,
        max_d: usize,
        d: usize,
    ) -> Result<Rule, SampleError> {
        let orig_self = self.clone();
        let orig_ctx = ctx.clone();
        loop {
            let mut vars = vec![];
            let lhs =
                self.sample_term_internal(schema, ctx, atom_weights, invent, max_d, d, &mut vars)?;
            let rhs =
                self.sample_term_internal(schema, ctx, atom_weights, false, max_d, d, &mut vars)?;
            if let Some(rule) = Rule::new(lhs, vec![rhs]) {
                return Ok(rule);
            } else {
                *self = orig_self.clone();
                *ctx = orig_ctx.clone();
            }
        }
    }
    pub fn sample_rule_from_context(
        &mut self,
        context: &RuleContext,
        ctx: &mut TypeContext,
        atom_weights: (f64, f64, f64),
        invent: bool,
        max_d: usize,
        d: usize,
    ) -> Result<Rule, SampleError> {
        let mut map = HashMap::new();
        let mut context = context.clone();
        let hole_places = context.holes();
        let mut context_vars = context.variables();
        self.infer_rule_context_internal(&context, ctx, &mut map)?;
        for p in &hole_places {
            let schema = TypeSchema::Monotype(map[p].apply(ctx));
            let subterm = self.sample_term_internal(
                &schema,
                ctx,
                atom_weights,
                p[0] == 0 && invent,
                max_d,
                d,
                &mut context_vars,
            )?;
            context = context
                .replace(&p, Context::from(subterm))
                .ok_or(SampleError::Subterm)?;
        }
        context.to_rule().or(Err(SampleError::Subterm))
    }

    fn logprior_term(
        &self,
        term: &Term,
        schema: &TypeSchema,
        ctx: &mut TypeContext,
        (vw, cw, ow): (f64, f64, f64),
        invent: bool,
    ) -> Result<f64, SampleError> {
        // instantiate the typeschema
        let tp = schema.instantiate(ctx);
        // setup the existing options
        let (mut vs, mut cs, mut os) = (vec![], vec![], vec![]);
        let atoms = self.signature.atoms();
        for atom in &atoms {
            if let Ok(arg_types) = self.fit_atom(atom, &tp, ctx) {
                match *atom {
                    Atom::Variable(_) => vs.push((Some(*atom), arg_types)),
                    Atom::Operator(o) if o.arity(&self.signature) == 0 => {
                        cs.push((Some(*atom), arg_types))
                    }
                    Atom::Operator(_) => os.push((Some(*atom), arg_types)),
                }
            }
        }
        if invent {
            if let Term::Variable(v) = *term {
                if !atoms.contains(&Atom::Variable(v)) {
                    vs.push((Some(Atom::Variable(v)), vec![]));
                }
            } else {
                vs.push((None, vec![]));
            }
        }
        // compute the log probability of selecting this head
        let z = vw + cw + ow;
        let (vw, cw, ow) = (vw / z, cw / z, ow / z);
        let vlp = vw.ln() - (vs.len() as f64).ln();
        let clp = cw.ln() - (cs.len() as f64).ln();
        let olp = ow.ln() - (os.len() as f64).ln();
        let mut options = vec![];
        options.append(&mut vs);
        options.append(&mut cs);
        options.append(&mut os);
        match options
            .into_iter()
            .find(|&(ref o, _)| o == &Some(term.head()))
            .map(|(o, arg_types)| (o.unwrap(), arg_types))
        {
            Some((Atom::Variable(_), _)) => Ok(vlp),
            Some((Atom::Operator(_), ref arg_types)) if arg_types.is_empty() => Ok(clp),
            Some((Atom::Operator(_), arg_types)) => {
                let mut lp = olp;
                for (subterm, mut arg_tp) in term.args().iter().zip(arg_types) {
                    arg_tp.apply_mut(ctx);
                    let arg_schema = TypeSchema::Monotype(arg_tp.clone());
                    lp += self.logprior_term(subterm, &arg_schema, ctx, (vw, cw, ow), invent)?;
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
        ctx: &mut TypeContext,
        atom_weights: (f64, f64, f64),
        invent: bool,
    ) -> Result<f64, SampleError> {
        let mut lp = 0.0;
        let lp_lhs = self.logprior_term(&rule.lhs, schema, ctx, atom_weights, invent)?;
        for rhs in &rule.rhs {
            let tmp_lp = self.logprior_term(&rhs, schema, ctx, atom_weights, false)?;
            lp += tmp_lp + lp_lhs;
        }
        Ok(lp)
    }
    fn logprior_utrs(
        &self,
        utrs: &UntypedTRS,
        schemas: &[TypeSchema],
        p_rule: f64,
        ctx: &mut TypeContext,
        atom_weights: (f64, f64, f64),
        invent: bool,
    ) -> Result<f64, SampleError> {
        // TODO: this might not be numerically stable
        // geometric distribution over number of rules
        let p_n_rules = p_rule.ln() * (utrs.clauses().len() as f64);
        let mut p_rules = 0.0;
        for rule in &utrs.rules {
            if self.background.contains(rule) {
                // TODO: if we can guarantee that UntypedTRS has self.background at the end,
                // then we wouldn't need to do this slow check
                continue;
            }
            let mut rule_ps = vec![];
            for schema in schemas {
                let tmp_lp = self.logprior_rule(&rule, schema, ctx, atom_weights, invent)?;
                rule_ps.push(tmp_lp);
            }
            p_rules += logsumexp(&rule_ps);
        }
        Ok(p_n_rules + p_rules)
    }
}
impl GP for Lexicon {
    type Expression = TRS;
    type Params = GeneticParams;
    fn genesis<R: Rng>(
        &self,
        _params: &Self::Params,
        rng: &mut R,
        pop_size: usize,
        _tp: &TypeSchema,
    ) -> Vec<Self::Expression> {
        match TRS::new(self, Vec::new()) {
            Ok(mut trs) => {
                if self.0.read().expect("poisoned lexicon").deterministic {
                    trs.utrs.make_deterministic(rng);
                }
                iter::repeat(trs).take(pop_size).collect()
            }
            Err(err) => {
                let lex = self.0.read().expect("poisoned lexicon");
                let background_trs = UntypedTRS::new(lex.background.clone());
                panic!(
                    "invalid background knowledge {}: {}",
                    background_trs.display(&lex.signature),
                    err
                )
            }
        }
    }
    fn mutate<R: Rng>(
        &self,
        params: &Self::Params,
        rng: &mut R,
        trs: &Self::Expression,
    ) -> Self::Expression {
        loop {
            if rng.gen_bool(params.p_add) {
                if let Ok(new_trs) = trs.add_rule(
                    &params.templates,
                    params.atom_weights,
                    params.max_sample_depth,
                    rng,
                ) {
                    return new_trs;
                }
            } else if let Ok(new_trs) = trs.delete_rule(rng) {
                return new_trs;
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
        let trs = self
            .combine(parent1, parent2)
            .expect("poorly-typed TRS in crossover");
        iter::repeat(trs)
            .take(params.n_crosses)
            .update(|trs| trs.utrs.rules.retain(|_| rng.gen_bool(params.p_keep)))
            .collect()
    }
}
