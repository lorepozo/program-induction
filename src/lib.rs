//! A library for program induction and learning grammars.
//!
//! Good places to look are [`ec`] and [`DSL`].
//!
//! [`ec`]: fn.ec.html
//! [`DSL`]: struct.DSL.html

#[macro_use]
extern crate polytype;

mod expression;
mod task;
pub mod ec;

pub use expression::*;
pub use task::*;
