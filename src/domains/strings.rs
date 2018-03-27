//! The string editing domain, for tackling Flashfill-style problems.
//!
//! # Examples
//!
//! ```
//! #[macro_use]
//! extern crate polytype;
//! extern crate programinduction;
//! use programinduction::{ECParams, EC};
//! use programinduction::lambda::LispEvaluator;
//! use programinduction::domains::strings;
//!
//! # fn main() {
//! let dsl = strings::dsl();
//! let lisp = LispEvaluator::new(strings::lisp_prims());
//! let task = lisp.make_task(
//!     arrow![tp!(str), tp!(str)],
//!     &[
//!         // Replace delimiter '>' with '/'
//!         ("\"OFJQc>BLVP>eMS\"", "\"OFJQc/BLVP/eMS\""),
//!     ],
//! );
//!
//! let ec_params = ECParams {
//!     frontier_limit: 10,
//!     search_limit_timeout: None,
//!     search_limit_description_length: Some(12.0),
//! };
//! let frontiers = dsl.explore(&ec_params, &[task]);
//! let solution = &frontiers[0].best_solution().unwrap().0;
//! assert_eq!(
//!     "(λ (join (char->str /) (split > $0)))",
//!     dsl.display(solution)
//! );
//! # }
//! ```

use lambda::Language;

/// The string editing [`lambda::Language`] defines the following operations:
///
/// ```ignore
/// "0":         tp!(int)
/// "+1":        arrow![tp!(int), tp!(int)]
/// "-1":        arrow![tp!(int), tp!(int)]
/// "len":       arrow![tp!(str), tp!(int)]
/// "empty_str": tp!(str)
/// "lower":     arrow![tp!(str), tp!(str)]
/// "upper":     arrow![tp!(str), tp!(str)]
/// "concat":    arrow![tp!(str), tp!(str), tp!(str)]
/// "slice":     arrow![tp!(int), tp!(int), tp!(str), tp!(str)]
/// "nth":       arrow![tp!(int), tp!(list(tp!(str))), tp!(str)]
/// "map":       arrow![arrow![tp!(0), tp!(1)], tp!(list(tp!(0))), tp!(list(tp!(1)))]
/// "strip":     arrow![tp!(str), tp!(str)]
/// "split":     arrow![tp!(char), tp!(str), tp!(list(tp!(str)))]
/// "join":      arrow![tp!(str), tp!(list(tp!(str))), tp!(str)]
/// "char->str": arrow![tp!(char), tp!(str)]
/// "space":     tp!(char)
/// ".":         tp!(char)
/// ",":         tp!(char)
/// "<":         tp!(char)
/// ">":         tp!(char)
/// "/":         tp!(char)
/// "@":         tp!(char)
/// "-":         tp!(char)
/// "|":         tp!(char)
/// ```
///
/// [`lambda::Language`]: ../../lambda/struct.Language.html
pub fn dsl() -> Language {
    Language::uniform(vec![
        ("0", tp!(int)),
        ("+1", arrow![tp!(int), tp!(int)]),
        ("-1", arrow![tp!(int), tp!(int)]),
        ("len", arrow![tp!(str), tp!(int)]),
        ("empty_str", tp!(str)),
        ("lower", arrow![tp!(str), tp!(str)]),
        ("upper", arrow![tp!(str), tp!(str)]),
        ("concat", arrow![tp!(str), tp!(str), tp!(str)]),
        ("slice", arrow![tp!(int), tp!(int), tp!(str), tp!(str)]),
        ("nth", arrow![tp!(int), tp!(list(tp!(str))), tp!(str)]),
        (
            "map",
            arrow![arrow![tp!(0), tp!(1)], tp!(list(tp!(0))), tp!(list(tp!(1)))],
        ),
        ("strip", arrow![tp!(str), tp!(str)]),
        ("split", arrow![tp!(char), tp!(str), tp!(list(tp!(str)))]),
        ("join", arrow![tp!(str), tp!(list(tp!(str))), tp!(str)]),
        ("char->str", arrow![tp!(char), tp!(str)]),
        ("space", tp!(char)),
        (".", tp!(char)),
        (",", tp!(char)),
        ("<", tp!(char)),
        (">", tp!(char)),
        ("/", tp!(char)),
        ("@", tp!(char)),
        ("-", tp!(char)),
        ("|", tp!(char)),
    ])
}
