use std::ops::Deref;
use super::Language;
use super::super::{Frontier, Task};

/// Parameters for grammar induction.
///
/// Proposed grammars are scored as `likelihood - aic * #primitives - structure_penalty * #nodes`.
/// Additionally, `pseudocounts` affects the likelihood calculation, and `topk` and `arity` affect
/// what fragments can be proposed.
pub struct Params {
    /// Pseudocounts are added to the observed counts associated with each primitive and invented
    /// expression.
    pub pseudocounts: u64,
    /// Rather than using every expression in the frontier for proposing fragments, only use the
    /// `topk` best expressions in each frontier.
    pub topk: u64,
    /// Structure penalty penalizes the total number of nodes in each [`Expression`] of the
    /// grammar's primitives and invented expressions.
    ///
    /// [`Expression`]: enum.Expression.html
    pub structure_penalty: f64,
    /// AIC is a penalty in the number of parameters, i.e. the number of primitives and invented
    /// expressions.
    pub aic: f64,
    /// Arity is the largest applicative depth of an expression that may be manipulated to propose
    /// a fragment.
    pub arity: u32,
}
impl Default for Params {
    /// The default params prevent completely discarding of primives by having non-zero
    /// pseudocounts.
    ///
    /// ```
    /// # use programinduction::lambda::Params;
    /// Params {
    ///     pseudocounts: 1,
    ///     topk: 2,
    ///     structure_penalty: 0f64,
    ///     aic: 1f64,
    ///     arity: 2,
    /// }
    /// # ;
    /// ```
    fn default() -> Self {
        Params {
            pseudocounts: 1,
            topk: 2,
            structure_penalty: 0f64,
            aic: 1f64,
            arity: 2,
        }
    }
}

pub fn induce<O: Sync>(
    dsl: &Language,
    params: &Params,
    tasks: &[Task<Language, O>],
    frontiers: &[Frontier<Language>],
) -> Language {
    let grammar: FragmentGrammar = dsl.into();
    let _ = (params, tasks, frontiers);
    grammar.0
}

struct FragmentGrammar(Language);
impl Deref for FragmentGrammar {
    type Target = Language;
    fn deref(&self) -> &Language {
        &self.0
    }
}
impl<'a> From<&'a Language> for FragmentGrammar {
    fn from(dsl: &'a Language) -> Self {
        FragmentGrammar(dsl.clone())
    }
}
