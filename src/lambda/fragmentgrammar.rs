use std::ops::Deref;
use super::Language;
use super::super::{Frontier, Task};

pub fn induce<O: Sync>(
    dsl: &Language,
    tasks: &[Task<Language, O>],
    frontiers: &[Frontier<Language>],
) -> Language {
    let grammar: FragmentGrammar = dsl.into();
    let _ = (tasks, frontiers);
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
