//! Hindley-Milner Typing for First-Order Term Rewriting Systems (no abstraction)
//!
//! Much thanks to:
//! - https://github.com/rob-smallshire/hindley-milner-python
//! - https://en.wikipedia.org/wiki/Hindley%E2%80%93Milner_type_system
//! - (TAPL; Pierce, 2002, ch. 22)

use polytype::{Context, TypeSchema};
use rand::Rng;
use std::f64::NEG_INFINITY;
use std::fmt;
use term_rewriting as trs;

use super::trace::Trace;
use super::{Lexicon, SampleError, TypeError};

/// Manages the semantics of a term rewriting system.
#[derive(Debug, PartialEq, Clone)]
pub struct RewriteSystem {
    // TODO: may also want to track background knowledge here.
    pub(crate) lex: Lexicon,
    pub(crate) trs: trs::TRS,
    pub(crate) ctx: Context,
}
impl RewriteSystem {
    pub fn new(lex: &Lexicon, trs: trs::TRS) -> Result<RewriteSystem, TypeError> {
        let lex = lex.clone();
        let mut ctx = Context::default();
        {
            let lex = lex.0.read().expect("poisoned lexicon");
            lex.infer_trs(&trs, &mut ctx)?;
        }
        Ok(RewriteSystem { lex, trs, ctx })
    }

    /// The size of the TRS (the sum over the size of the rules in the underlying [`TRS`])
    ///
    /// [`TRS`]: ../../term_rewriting/struct.TRS.html
    pub fn size(&self) -> usize {
        self.trs.size()
    }

    pub fn pseudo_log_prior(&self, temp: f64, prior_temp: f64) -> f64 {
        let raw_prior = -(self.size() as f64);
        raw_prior / ((temp + 1.0) * prior_temp)
    }

    pub fn log_likelihood(
        &self,
        data: &[trs::Rule],
        p_partial: f64,
        temp: f64,
        ll_temp: f64,
    ) -> f64 {
        data.iter()
            .map(|x| self.single_log_likelihood(x, p_partial, temp) / ll_temp)
            .sum()
    }

    fn single_log_likelihood(&self, datum: &trs::Rule, p_partial: f64, temp: f64) -> f64 {
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
        data: &[trs::Rule],
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
    pub fn add_rule<R: Rng>(&self, _rng: &mut R) -> Result<RewriteSystem, SampleError> {
        let mut rs = self.clone();
        let schema = TypeSchema::Monotype(rs.ctx.new_variable());
        let rule = {
            let mut lex = self.lex.0.write().expect("poisoned lexicon");
            lex.sample_rule(&schema, &mut rs.ctx, true, 4, 0)?
        };
        {
            let lex = self.lex.0.write().expect("poisoned lexicon");
            lex.infer_rule(&rule, &mut rs.ctx)?;
        }
        rs.trs.push(rule);
        Ok(rs)
    }
    /// Delete a rule from the rewrite system if possible.
    pub fn delete_rule<R: Rng>(&self, rng: &mut R) -> Result<RewriteSystem, SampleError> {
        let mut rs = self.clone();
        if !self.trs.is_empty() {
            let idx = rng.gen_range(0, self.trs.len());
            rs.trs.rules.remove(idx);
            Ok(rs)
        } else {
            Err(SampleError::OptionsExhausted)
        }
    }
}
impl fmt::Display for RewriteSystem {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let lex = self.lex.0.read().expect("poisoned lexicon");
        write!(f, "{}", self.trs.display(&lex.signature))
    }
}
