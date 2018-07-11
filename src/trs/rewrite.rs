//! Polymorphically typed (Hindley-Milner) First-Order Term Rewriting Systems (no abstraction)
//!
//! Much thanks to:
//! - https://github.com/rob-smallshire/hindley-milner-python
//! - https://en.wikipedia.org/wiki/Hindley%E2%80%93Milner_type_system
//! - (TAPL; Pierce, 2002, ch. 22)

use polytype::Context as TypeContext;
use rand::seq::sample_iter;
use rand::Rng;
use std::f64::NEG_INFINITY;
use std::fmt;
use term_rewriting::trace::Trace;
use term_rewriting::{Rule, RuleContext, TRS as UntypedTRS};

use super::{Lexicon, ModelParams, SampleError, TypeError};

/// Manages the semantics of a term rewriting system.
#[derive(Debug, PartialEq, Clone)]
pub struct TRS {
    // TODO: may also want to track background knowledge here.
    pub(crate) lex: Lexicon,
    // INVARIANT: UntypedTRS.rules ends with lex.background
    pub(crate) utrs: UntypedTRS,
    pub(crate) ctx: TypeContext,
}
impl TRS {
    /// Create a new `TRS` under the given `Lexicon`. Any background knowledge will be appended to
    /// the given ruleset.
    pub fn new(lex: &Lexicon, mut rules: Vec<Rule>) -> Result<TRS, TypeError> {
        let lex = lex.clone();
        let mut ctx = TypeContext::default();
        let utrs = {
            let lex = lex.0.read().expect("poisoned lexicon");
            rules.append(&mut lex.background.clone());
            let utrs = UntypedTRS::new(rules);
            lex.infer_utrs(&utrs, &mut ctx)?;
            utrs
        };
        Ok(TRS { lex, utrs, ctx })
    }

    /// The size of the TRS (the sum over the size of the rules in the underlying [`TRS`])
    ///
    /// [`TRS`]: ../../term_rewriting/struct.TRS.html
    pub fn size(&self) -> usize {
        self.utrs.size()
    }

    pub fn pseudo_log_prior(&self) -> f64 {
        -(self.size() as f64)
    }

    pub fn log_likelihood(&self, data: &[Rule], params: ModelParams) -> f64 {
        data.iter()
            .map(|x| self.single_log_likelihood(x, params))
            .sum()
    }

    fn single_log_likelihood(&self, datum: &Rule, params: ModelParams) -> f64 {
        let ll = if let Some(ref rhs) = datum.rhs() {
            let mut trace = Trace::new(&self.utrs, &datum.lhs, params.p_observe, params.max_size);
            trace.rewrites_to(params.max_steps, rhs)
        } else {
            NEG_INFINITY
        };

        if ll == NEG_INFINITY {
            params.p_partial.ln()
        } else {
            (1.0 - params.p_partial).ln() + ll
        }
    }

    pub fn posterior(&self, data: &[Rule], params: ModelParams) -> f64 {
        let prior = self.pseudo_log_prior();
        if prior == NEG_INFINITY {
            NEG_INFINITY
        } else {
            prior + self.log_likelihood(data, params)
        }
    }

    /// Sample a rule and add it to the rewrite system.
    pub fn add_rule<R: Rng>(
        &self,
        contexts: &[RuleContext],
        atom_weights: (f64, f64, f64),
        max_depth: usize,
        rng: &mut R,
    ) -> Result<TRS, SampleError> {
        let mut trs = self.clone();
        let context = sample_iter(rng, contexts, 1)?[0];
        let rule = trs.lex.sample_rule_from_context(
            &context,
            &mut trs.ctx,
            atom_weights,
            true,
            max_depth,
        )?;
        trs.lex
            .0
            .write()
            .expect("poisoned lexicon")
            .infer_rule(&rule, &mut trs.ctx)?;
        trs.utrs.push(rule)?;
        Ok(trs)
    }
    /// Delete a rule from the rewrite system if possible. Background knowledge cannot be deleted.
    pub fn delete_rule<R: Rng>(&self, rng: &mut R) -> Result<TRS, SampleError> {
        let background = &self.lex.0.read().expect("poisoned lexicon").background;
        let clauses = self.utrs.clauses();
        let deletable: Vec<_> = clauses.iter().filter(|c| !background.contains(c)).collect();
        if deletable.is_empty() {
            //println!("  Err: no rules to delete.");
            Err(SampleError::OptionsExhausted)
        } else {
            let mut trs = self.clone();
            trs.utrs.remove_clauses(sample_iter(rng, deletable, 1)?[0])?;
            //println!("Success");
            Ok(trs)
        }
    }
}
impl fmt::Display for TRS {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let sig = &self.lex.0.read().expect("poisoned lexicon").signature;
        write!(f, "{}", self.utrs.display(sig))
    }
}
