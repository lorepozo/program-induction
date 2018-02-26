use std::ops::Deref;
use super::Language;
use super::super::{Frontier, Task};

pub struct Params {
    pub smoothing: f64,
    pub structure_penalty: f64,
    pub aic: f64,
    pub arity: u32,
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
