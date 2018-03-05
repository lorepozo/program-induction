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
//! let dsl = strings::repr();
//! let lisp = LispEvaluator::new(strings::lisp_prims());
//! let task = lisp.make_task(
//!     arrow![tp!(str), tp!(str)],
//!     vec![
//!         // Replace delimiter '>' with '/'
//!         (Some("\"OFJQc>BLVP>eMS\""), "\"OFJQc/BLVP/eMS\""),
//!     ],
//! );
//!
//! let ec_params = ECParams {
//!     frontier_limit: 10,
//!     search_limit: 2500,
//! };
//! let frontiers = dsl.explore(&ec_params, &[task], None);
//! let solution = &frontiers[0].best_solution().unwrap().0;
//! assert_eq!(
//!     "(λ (join (char->str /) (split > $0)))",
//!     dsl.stringify(solution)
//! );
//! # }
//! ```

use super::super::lambda::Language;

/// The string editing [`Representation`] defines the following operations:
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
/// [`Representation`]: ../../trait.Representation.html
/// [`lambda::Language`]: ../../lambda/struct.Language.html
pub fn repr() -> Language {
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

/// Primitives for evaluation with [`lambda::LispEvaluator::new`].
///
/// [`lambda::LispEvaluator::new`]: ../../lambda/struct.LispEvaluator.html#method.new
pub fn lisp_prims() -> Vec<(&'static str, &'static str)> {
    vec![
        ("+1", "(λ (x) (+ x 1))"),
        ("-1", "(λ (x) (- x 1))"),
        ("len", "string-length"),
        ("empty_str", "\"\""),
        ("lower", "string-downcase"),
        ("upper", "string-upcase"),
        ("concat", "string-append"),
        ("slice", "(λ (x y s) (substring s x y))"),
        ("nth", "(λ (n s) (list-ref s n))"),
        ("strip", "string-trim"),
        ("split", "(λ (c s) (string-split s c))"),
        ("join", "(λ (s ss) (string-join ss s))"),
        ("char->str", "identity"),
        ("space", "\" \""),
        (".", "\".\""),
        (",", "\",\""),
        ("<", "\"<\""),
        (">", "\">\""),
        ("/", "\"/\""),
        ("@", "\"@\""),
        ("-", "\"-\""),
        ("|", "\"|\""),
    ]
}
