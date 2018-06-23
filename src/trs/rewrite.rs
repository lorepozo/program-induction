//! Polymorphically typed (Hindley-Milner) First-Order Term Rewriting Systems (no abstraction)
//!
//! Much thanks to:
//! - https://github.com/rob-smallshire/hindley-milner-python
//! - https://en.wikipedia.org/wiki/Hindley%E2%80%93Milner_type_system
//! - (TAPL; Pierce, 2002, ch. 22)

use polytype::{Context as TypeContext, TypeSchema};
use rand::Rng;
use std::f64::NEG_INFINITY;
use std::fmt;
use term_rewriting::{Rule, TRS as UntypedTRS};

use super::trace::Trace;
use super::{Lexicon, SampleError, TypeError};

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
        let mut trace = Trace::new(&self.utrs, &datum.lhs, p_observe, max_steps, max_size);
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

    /// Sample a rule and add it to the rewrite system.
    pub fn add_rule<R: Rng>(&self, _rng: &mut R) -> Result<TRS, SampleError> {
        let mut trs = self.clone();
        let schema = TypeSchema::Monotype(trs.ctx.new_variable());
        let rule = trs.lex.0.write().expect("poisoned lexicon").sample_rule(
            &schema,
            &mut trs.ctx,
            true,
            4,
            0,
        )?;
        trs.lex
            .0
            .write()
            .expect("poisoned lexicon")
            .infer_rule(&rule, &mut trs.ctx)?;
        trs.utrs.rules.insert(0, rule);
        Ok(trs)
    }
    /// Delete a rule from the rewrite system if possible. Background knowledge cannot be deleted.
    pub fn delete_rule<R: Rng>(&self, rng: &mut R) -> Option<TRS> {
        let background_size = self.lex
            .0
            .read()
            .expect("poisoned lexicon")
            .background
            .len();
        let deletable = self.utrs.len() - background_size;
        if deletable == 0 {
            None
        } else {
            let mut trs = self.clone();
            let idx = rng.gen_range(0, deletable);
            trs.utrs.rules.remove(idx);
            Some(trs)
        }
    }
}
impl fmt::Display for TRS {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let sig = &self.lex.0.read().expect("poisoned lexicon").signature;
        write!(f, "{}", self.utrs.display(sig))
    }
}
