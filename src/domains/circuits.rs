//! The classic circuit domain as in the paper "Bootstrap Learning for Modular Concept Discovery"
//! (2013).
//!
//! The [`Representation`] for circuits, [`BASE_REPR`], is just the `nand` operation in lambda
//! calculus ([`lambda::Language`]).
//!
//! [`BASE_REPR`]: struct.BASE_DSL.html
//! [`Representation`]: ../../trait.Representation.html
//! [`lambda::Language`]: ../../lambda/struct.Language.html

use super::super::lambda::Language;

lazy_static! {
    /// Treat this as any other [`lambda::Language`].
    ///
    /// It only defines the binary `nand` operation:
    ///
    /// ```ignore
    /// "nand": arrow![tp!(bool), tp!(bool), tp!(bool)])
    /// ```
    ///
    /// [`lambda::Language`]: ../../lambda/struct.Language.html
    pub static ref BASE_REPR: Language = {
        Language::uniform(
            vec![
                (String::from("nand"), arrow![tp!(bool), tp!(bool), tp!(bool)]),
            ],
            vec![],
        )
    };
}

/// Evaluate an expression in this domain in accordance with the argument of
/// [`lambda::task_by_examples`].
///
/// [`lambda::task_by_examples`]: ../../lambda/fn.task_by_example.html
pub fn evaluator(primitive: &str, inp: &[bool]) -> bool {
    match primitive {
        "nand" => !(inp[0] & inp[1]),
        _ => unreachable!(),
    }
}
