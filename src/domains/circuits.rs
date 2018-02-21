//! The classic circuit domain as in the paper "Bootstrap Learning for Modular Concept Discovery"
//! (2013).
//!
//! The [`BASE_DSL`] is just the `nand` operation.
//!
//! [`BASE_DSL`]: struct.BASE_DSL.html

use super::super::DSL;

lazy_static! {
    /// Treat this struct as any other DSL.
    ///
    /// It only defines the binary `nand` operation:
    ///
    /// ```ignore
    /// "nand": arrow![tp!(bool), tp!(bool), tp!(bool)])
    /// ```
    pub static ref BASE_DSL: DSL = {
        DSL {
            primitives: vec![
                (String::from("nand"), arrow![tp!(bool), tp!(bool), tp!(bool)]),
            ],
            invented: vec![],
        }
    };
}

/// Evaluate an expression in this domain.
pub fn evaluator(primitive: &str, inp: &Vec<bool>) -> bool {
    match primitive {
        "nand" => !(inp[0] & inp[1]),
        _ => unreachable!(),
    }
}
