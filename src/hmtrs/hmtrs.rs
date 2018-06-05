///! Hindley-Milner Typing for First-Order Term Rewriting Systems (no abstraction)
///!
///! Much thanks to:
///! - https://github.com/rob-smallshire/hindley-milner-python
///! - https://en.wikipedia.org/wiki/Hindley%E2%80%93Milner_type_system
///! - (TAPL; Pierce, 2002, ch. 22)
use itertools::Itertools;
use polytype::{Context, Type, TypeSchema, Variable as pVar};
use term_rewriting::{Operator, Rule, Signature, Term, Variable as tVar, TRS};

/// A Hindley-Milner Term Rewriting System (HMTRS): a first-order [term rewriting system][0] with a [Hindley-Milner type system][1].
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
    rules: Vec<(TypeSchema, TypeSchema, Vec<TypeSchema>)>,
    pub signature: Signature,
    pub trs: TRS,
}
impl HMTRS {
    /// the size of the HMTRS (the sum over the size of the rules in the [`TRS`])
    ///
    /// [`TRS`]: ../../term_rewriting/struct.TRS.html
    pub fn size(&self) -> usize {
        self.trs.size()
    }
    /// Convert the `HMTRS` to a `String`.
    pub fn display(&self) -> String {
        self.trs.display(&self.signature)
    }
    /// All the free variables in the lexicon.
    pub fn free_vars(&self) -> Vec<pVar> {
        let vars_fvs = self.vars.iter().flat_map(|x| x.free_vars());
        let ops_fvs = self.ops.iter().flat_map(|x| x.free_vars());
        vars_fvs.chain(ops_fvs).unique().collect()
    }
    /// Infer the `Type` of a `Variable`.
    pub fn infer_var(&self, v: &tVar) -> Result<TypeSchema, TypeError> {
        if let Some(idx) = self.signature.variables().iter().position(|x| x == v) {
            Ok(self.vars[idx].clone())
        } else {
            Err(TypeError)
        }
    }
    /// Infer the `Type` of a `Variable`.
    pub fn infer_op(&self, o: &Operator) -> Result<TypeSchema, TypeError> {
        if let Some(idx) = self.signature.operators().iter().position(|x| x == o) {
            Ok(self.ops[idx].clone())
        } else {
            Err(TypeError)
        }
    }
    /// Infer the `TypeSchema` of a `Term` and the corresponding `Context`.
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
    pub fn infer_args(&self, args: &[Term], ctx: &mut Context) -> Result<Type, TypeError> {
        let mut pre_types = vec![];
        for a in args {
            pre_types.push(self.infer_internal(a, ctx)?);
        }
        pre_types.push(ctx.new_variable());
        if let Ok(body_type) = Type::multi_arrow(pre_types) {
            Ok(body_type)
        } else {
            Err(TypeError)
        }
    }
    pub fn infer_rule(&self, r: &Rule, ctx: &mut Context) -> Result<TypeSchema, TypeError> {
        let lhs_schema = self.infer_term(&r.lhs, ctx)?;
        let lhs_type = lhs_schema.instantiate(ctx);
        let mut rhs_types = vec![];
        for rhs in r.rhs.iter() {
            let rhs_schema = self.infer_term(&rhs, ctx)?;
            rhs_types.push(rhs_schema.instantiate(ctx));
        }
        for rhs_type in rhs_types {
            ctx.unify(&lhs_type, &rhs_type);
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
        Ok(lhs_type.apply(ctx).generalize(&bound_vs))
    }
    pub fn infer_trs(&self, t: &TRS, ctx: &mut Context) -> Result<(), TypeError> {
        // TODO: Right now, this assumes the variables already exist in the signature. Is that sensible?
        for rule in self.trs.rules.iter() {
            self.infer_rule(rule, ctx)?;
        }
        Ok(())
    }
}

pub struct TypeError;
