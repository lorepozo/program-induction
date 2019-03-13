//! Polymorphically typed (Hindley-Milner) First-Order Term Rewriting Systems (no abstraction)
//!
//! Much thanks to:
//! - https://github.com/rob-smallshire/hindley-milner-python
//! - https://en.wikipedia.org/wiki/Hindley%E2%80%93Milner_type_system
//! - (TAPL; Pierce, 2002, ch. 22)

use itertools::Itertools;
use polytype::Context as TypeContext;
use rand::seq::SliceRandom;
use rand::Rng;
use std::f64::NEG_INFINITY;
use std::fmt;
use term_rewriting::trace::Trace;
use term_rewriting::{Rule, RuleContext, Strategy as RewriteStrategy, Term, TRS as UntypedTRS};

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
        let context = contexts.choose(rng).ok_or(SampleError::OptionsExhausted)?;
        let rule = trs.lex.sample_rule_from_context(
            context.clone(),
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
                .remove_clauses(deletable.choose(rng).ok_or(SampleError::OptionsExhausted)?)?;
            Ok(trs)
        }
    }
    /// Move a Rule from one place in the TRS to another at random, excluding the background.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate rand;
    /// # extern crate term_rewriting;
    /// # use programinduction::trs::{TRS, Lexicon};
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
    /// println!("{:?}", sig.operators());
    /// for op in sig.operators() {
    ///     println!("{:?}/{}", op.name(&sig), op.arity(&sig))
    /// }
    /// for r in &rules {
    ///     println!("{:?}", r);
    /// }
    /// let lexicon = Lexicon::from_signature(sig.clone(), ops, vars, vec![], false);
    ///
    /// let mut trs = TRS::new(&lexicon, rules).unwrap();
    ///
    /// let pretty_before = trs.pretty_utrs(&sig);
    ///
    /// let mut rng = thread_rng();
    ///
    /// let new_trs = trs.randomly_move_rule(&mut rng).expect("failed when moving rule");
    ///
    /// assert_ne!(pretty_before, new_trs.pretty_utrs(&sig));
    /// assert_eq!(new_trs.pretty_utrs(&sig), "PLUS(x_, SUCC(y_)) = SUCC(PLUS(x_, y_));\nPLUS(x_, 0) = x_;");
    /// # }
    /// ```
    pub fn randomly_move_rule<R: Rng>(&self, rng: &mut R) -> Result<TRS, SampleError> {
        let mut trs = self.clone();
        let num_rules = self.len();
        let num_background = self
            .lex
            .0
            .read()
            .expect("poisoned lexicon")
            .background
            .len();
        if num_background >= num_rules - 1 {
            return Ok(trs);
        }
        let i: usize = rng.gen_range(num_background, num_rules);
        let mut j: usize = rng.gen_range(num_background, num_rules);
        while j == i {
            j = rng.gen_range(0, num_rules);
        }
        trs.utrs
            .move_rule(i, j)
            .expect("moving rule from random locations i to j");
        Ok(trs)
    }
    /// Selects a rule from the TRS at random, finds all differences in the LHS and RHS,
    /// and makes rules from those differences and inserts them back into copies of the TRS imediately after the background.
    pub fn local_difference_vec<R: Rng>(&self, rng: &mut R) -> Result<Vec<TRS>, SampleError> {
        let mut trs = self.clone();
        let num_rules = self.len();
        let num_background = self
            .lex
            .0
            .read()
            .expect("poisoned lexicon")
            .background
            .len();
        if num_rules == num_background {
            return Ok(vec![trs]);
        }
        let idx = rng.gen_range(num_background, num_rules);
        let result = TRS::local_difference_helper(&trs.utrs.rules[idx]);
        if result == None {
            return Ok(vec![trs]);
        }
        trs.utrs.remove_idx(idx).expect("removing original rule");
        let new_rules = result.unwrap();
        let mut trs_vec = vec![];
        for idx in 0..new_rules.len() {
            let mut temp_trs = trs.clone();
            temp_trs
                .utrs
                .insert_idx(num_background, new_rules[idx].clone())?;
            trs_vec.push(temp_trs);
        }
        Ok(trs_vec)
    }
    /// Selects a rule from the TRS at random, finds all differences in the LHS and RHS,
    /// and makes rules from those differences and inserts them back into the TRS imediately after the background.
    pub fn local_difference<R: Rng>(&self, rng: &mut R) -> Result<TRS, SampleError> {
        let mut trs = self.clone();
        let num_rules = self.len();
        let num_background = self
            .lex
            .0
            .read()
            .expect("poisoned lexicon")
            .background
            .len();
        if num_rules == num_background {
            return Ok(trs);
        }
        let idx = rng.gen_range(num_background, num_rules);
        let result = TRS::local_difference_helper(&trs.utrs.rules[idx]);
        if result == None {
            return Ok(trs);
        }
        trs.utrs.remove_idx(idx).expect("removing original rule");
        let new_rules = result.unwrap();
        trs.utrs.inserts_idx(num_background, new_rules)?;
        Ok(trs)
    }
    /// Given a rule that has similar terms in the lhs and rhs,
    /// returns a list of rules where each similarity is removed one at a time
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate polytype;
    /// # extern crate programinduction;
    /// # extern crate rand;
    /// # extern crate term_rewriting;
    /// # use programinduction::trs::{TRS, Lexicon};
    /// # use rand::{thread_rng};
    /// # use term_rewriting::{Context, RuleContext, Signature, parse_rule};
    /// # fn main() {
    /// let mut sig = Signature::default();
    ///
    /// let r = parse_rule(&mut sig, "F(A(B C(x_))) = F(A(D E(x_)))").expect("parse of F(A(B C(x_))) = F(A(D E(x_)))");
    ///
    /// let result = TRS::local_difference_helper(&r);
    ///
    /// if result == None {
    ///     assert!(false);
    /// } else {
    ///     let rules = result.unwrap();
    ///     assert_eq!(rules.len(), 4);
    ///     assert_eq!(rules[0].display(&sig), "F(A(B C(x_))) = F(A(D E(x_)))");
    ///     assert_eq!(rules[1].display(&sig), "A(B C(x_)) = A(D E(x_))");
    ///     assert_eq!(rules[2].display(&sig), "B = D");
    ///     assert_eq!(rules[3].display(&sig), "C(x_) = E(x_)");
    /// }
    /// # }
    /// ```
    pub fn local_difference_helper(rule: &Rule) -> Option<Vec<Rule>> {
        let r = rule.clone();
        let rhs = r.rhs();
        if rhs == None {
            return None;
        }
        let temp_differences = TRS::find_differences(r.lhs, rhs.unwrap());
        if temp_differences == None {
            return None;
        }
        let differences = temp_differences.unwrap();
        let mut rules: Vec<Rule> = vec![];
        for idx in 0..differences.len() {
            let temp_rule = Rule::new(differences[idx].0.clone(), vec![differences[idx].1.clone()]);
            if temp_rule != None {
                rules.push(temp_rule.unwrap());
            }
        }
        if rules == vec![] {
            return None;
        }
        Some(rules)
    }
    // helper for local difference, finds differences in the given lhs and rhs recursively
    pub fn find_differences(lhs: Term, rhs: Term) -> Option<Vec<(Term, Term)>> {
        if lhs == rhs {
            return None;
        }
        match lhs.clone() {
            Term::Variable(_x) => {
                return None;
            }
            Term::Application {
                op: lop,
                args: largs,
            } => {
                if largs.len() == 0 {
                    return Some(vec![(lhs, rhs)]);
                }
                match rhs.clone() {
                    Term::Variable(_x) => {
                        return Some(vec![(lhs, rhs)]);
                    }
                    Term::Application {
                        op: rop,
                        args: rargs,
                    } => {
                        if lop != rop {
                            return Some(vec![(lhs, rhs)]);
                        }
                        let mut differences: Vec<(Term, Term)> = vec![(lhs, rhs)];
                        for idx in 0..largs.len() {
                            let diff =
                                TRS::find_differences(largs[idx].clone(), rargs[idx].clone());
                            if diff != None {
                                let new_diffs = diff.unwrap();
                                for ids in 0..new_diffs.len() {
                                    differences.push(new_diffs[ids].clone());
                                }
                            }
                        }
                        if differences == vec![] {
                            return None;
                        }
                        return Some(differences);
                    }
                }
            }
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
