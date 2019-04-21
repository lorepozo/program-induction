//! Polymorphically typed (Hindley-Milner) First-Order Term Rewriting Systems (no abstraction)
//!
//! Much thanks to:
//! - https://github.com/rob-smallshire/hindley-milner-python
//! - https://en.wikipedia.org/wiki/Hindley%E2%80%93Milner_type_system
//! - (TAPL; Pierce, 2002, ch. 22)

use itertools::Itertools;
use polytype::Context as TypeContext;
use rand::seq::sample_iter;
use rand::Rng;
use std::f64::NEG_INFINITY;
use std::fmt;
use term_rewriting::trace::Trace;
use term_rewriting::{Rule, RuleContext, Strategy as RewriteStrategy, TRS as UntypedTRS};

use super::{Lexicon, ModelParams, SampleError, TypeError};

/// Manages the semantics of a term rewriting system.
#[derive(Debug, PartialEq, Clone)]
pub struct TRS {
    pub(crate) lex: Lexicon,
    // INVARIANT: UntypedTRS.rules ends with lex.background
    pub(crate) utrs: UntypedTRS,
    pub(crate) ctx: TypeContext,
}
impl TRS {
    /// Create a new `TRS` under the given [`Lexicon`]. Any background knowledge
    /// will be appended to the given ruleset.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate term_rewriting;
    /// # use programinduction::trs::{TRS, Lexicon};
    /// # use term_rewriting::{Signature, parse_rule};
    /// # use polytype::Context as TypeContext;
    /// # fn main() {
    /// let mut sig = Signature::default();
    ///
    /// let mut ops = vec![];
    /// sig.new_op(2, Some("PLUS".to_string()));
    /// ops.push(ptp![@arrow[tp!(int), tp!(int), tp!(int)]]);
    /// sig.new_op(1, Some("SUCC".to_string()));
    /// ops.push(ptp![@arrow[tp!(int), tp!(int)]]);
    /// sig.new_op(0, Some("ZERO".to_string()));
    /// ops.push(ptp![int]);
    ///
    /// let rules = vec![
    ///     parse_rule(&mut sig, "PLUS(x_ ZERO) = x_").expect("parsed rule"),
    ///     parse_rule(&mut sig, "PLUS(x_ SUCC(y_)) = SUCC(PLUS(x_ y_))").expect("parsed rule"),
    /// ];
    ///
    /// let vars = vec![
    ///     ptp![int],
    ///     ptp![int],
    ///     ptp![int],
    /// ];
    ///
    /// let lexicon = Lexicon::from_signature(sig, ops, vars, vec![], vec![], false, TypeContext::default());
    ///
    /// let ctx = lexicon.context();
    ///
    /// let trs = TRS::new(&lexicon, rules, &ctx).unwrap();
    ///
    /// assert_eq!(trs.size(), 12);
    /// # }
    /// ```
    /// [`Lexicon`]: struct.Lexicon.html
    pub fn new(
        lexicon: &Lexicon,
        mut rules: Vec<Rule>,
        ctx: &TypeContext,
    ) -> Result<TRS, TypeError> {
        let lexicon = lexicon.clone();
        let mut ctx = ctx.clone();
        let utrs = {
            let lex = lexicon.0.read().expect("poisoned lexicon");
            rules.append(&mut lex.background.clone());
            let utrs = UntypedTRS::new(rules);
            lex.infer_utrs(&utrs, &mut ctx)?;
            utrs
        };
        Ok(TRS {
            lex: lexicon,
            utrs,
            ctx,
        })
    }

    /// The size of the underlying [`term_rewriting::TRS`].
    ///
    /// [`term_rewriting::TRS`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.TRS.html#method.size
    pub fn size(&self) -> usize {
        self.utrs.size()
    }

    /// The length of the underlying [`term_rewriting::TRS`].
    ///
    /// [`term_rewriting::TRS`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.TRS.html#method.size
    pub fn len(&self) -> usize {
        self.utrs.len()
    }

    /// Is the underlying [`term_rewriting::TRS`] empty?.
    ///
    /// [`term_rewriting::TRS`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/struct.TRS.html#method.size
    pub fn is_empty(&self) -> bool {
        self.utrs.is_empty()
    }

    /// A pseudo log prior for a `TRS`: the negative [`size`] of the `TRS`.
    ///
    /// [`size`]: struct.TRS.html#method.size
    pub fn pseudo_log_prior(&self) -> f64 {
        -(self.size() as f64)
    }

    /// A log likelihood for a `TRS`: the probability of `data`'s RHSs appearing
    /// in [`term_rewriting::Trace`]s rooted at its LHSs.
    ///
    /// [`term_rewriting::Trace`]: https://docs.rs/term_rewriting/~0.3/term_rewriting/trace/struct.Trace.html
    pub fn log_likelihood(&self, data: &[Rule], params: ModelParams) -> f64 {
        data.iter()
            .map(|x| self.single_log_likelihood(x, params))
            .sum()
    }

    /// Compute the log likelihood for a single datum.
    fn single_log_likelihood(&self, datum: &Rule, params: ModelParams) -> f64 {
        let ll = if let Some(ref rhs) = datum.rhs() {
            let mut trace = Trace::new(
                &self.utrs,
                &datum.lhs,
                params.p_observe,
                params.max_size,
                RewriteStrategy::All,
            );
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

    /// Combine [`pseudo_log_prior`] and [`log_likelihood`], failing early if the
    /// prior is `0.0`.
    ///
    /// [`pseudo_log_prior`]: struct.TRS.html#method.pseudo_log_prior
    /// [`log_likelihood`]: struct.TRS.html#method.log_likelihood
    pub fn posterior(&self, data: &[Rule], params: ModelParams) -> f64 {
        let prior = self.pseudo_log_prior();
        if prior == NEG_INFINITY {
            NEG_INFINITY
        } else {
            prior + self.log_likelihood(data, params)
        }
    }

    /// Sample a rule and add it to the rewrite system.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate rand;
    /// # extern crate term_rewriting;
    /// # use programinduction::trs::{TRS, Lexicon};
    /// # use polytype::Context as TypeContext;
    /// # use rand::{thread_rng};
    /// # use term_rewriting::{Context, RuleContext, Signature, parse_rule};
    /// # fn main() {
    /// let mut sig = Signature::default();
    ///
    /// let mut ops = vec![];
    /// sig.new_op(2, Some(".".to_string()));
    /// ops.push(ptp![0, 1; @arrow[tp!(@arrow[tp!(0), tp!(1)]), tp!(0), tp!(1)]]);
    /// sig.new_op(2, Some("PLUS".to_string()));
    /// ops.push(ptp![@arrow[tp!(int), tp!(int), tp!(int)]]);
    /// sig.new_op(1, Some("SUCC".to_string()));
    /// ops.push(ptp![@arrow[tp!(int), tp!(int)]]);
    /// sig.new_op(0, Some("ZERO".to_string()));
    /// ops.push(ptp![int]);
    ///
    /// let rules = vec![
    ///     parse_rule(&mut sig, "PLUS(x_ ZERO) = x_").expect("parsed rule"),
    ///     parse_rule(&mut sig, "PLUS(x_ SUCC(y_)) = SUCC(PLUS(x_ y_))").expect("parsed rule"),
    /// ];
    ///
    /// let vars = vec![
    ///     ptp![int],
    ///     ptp![int],
    ///     ptp![int],
    /// ];
    ///
    /// for op in sig.operators() {
    ///     println!("{:?}/{}", op.name(), op.arity())
    /// }
    /// for r in &rules {
    ///     println!("{:?}", r.pretty());
    /// }
    /// let lexicon = Lexicon::from_signature(sig, ops, vars, vec![], vec![], false, TypeContext::default());
    ///
    /// let mut trs = TRS::new(&lexicon, rules, &lexicon.context()).unwrap();
    ///
    /// assert_eq!(trs.len(), 2);
    ///
    /// let contexts = vec![
    ///     RuleContext {
    ///         lhs: Context::Hole,
    ///         rhs: vec![Context::Hole],
    ///     }
    /// ];
    /// let mut rng = thread_rng();
    /// let atom_weights = (0.5, 0.25, 0.25);
    /// let max_size = 50;
    ///
    /// if let Ok(new_trs) = trs.add_rule(&contexts, atom_weights, max_size, &mut rng) {
    ///     assert_eq!(new_trs.len(), 3);
    /// } else {
    ///     assert_eq!(trs.len(), 2);
    /// }
    /// # }
    /// ```
    pub fn add_rule<R: Rng>(
        &self,
        contexts: &[RuleContext],
        atom_weights: (f64, f64, f64),
        max_size: usize,
        rng: &mut R,
    ) -> Result<TRS, SampleError> {
        let mut trs = self.clone();
        let context = sample_iter(rng, contexts, 1)?[0].clone();
        let rule = trs.lex.sample_rule_from_context(
            context,
            &mut trs.ctx,
            atom_weights,
            true,
            max_size,
        )?;
        trs.lex
            .0
            .write()
            .expect("poisoned lexicon")
            .infer_rule(&rule, &mut trs.ctx)?;
        trs.utrs.push(rule)?;
        Ok(trs)
    }
    /// Delete a rule from the rewrite system if possible. Background knowledge
    /// cannot be deleted.
    pub fn delete_rule<R: Rng>(&self, rng: &mut R) -> Result<TRS, SampleError> {
        let background = &self.lex.0.read().expect("poisoned lexicon").background;
        let clauses = self.utrs.clauses();
        let deletable: Vec<_> = clauses.iter().filter(|c| !background.contains(c)).collect();
        if deletable.is_empty() {
            Err(SampleError::OptionsExhausted)
        } else {
            let mut trs = self.clone();
            trs.utrs
                .remove_clauses(sample_iter(rng, deletable, 1)?[0])?;
            Ok(trs)
        }
    }
}
impl fmt::Display for TRS {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let true_len = self.utrs.len()
            - self
                .lex
                .0
                .read()
                .expect("poisoned lexicon")
                .background
                .len();
        let trs_str = self
            .utrs
            .rules
            .iter()
            .take(true_len)
            .map(|r| format!("{};", r.display()))
            .join("\n");

        write!(f, "{}", trs_str)
    }
}
