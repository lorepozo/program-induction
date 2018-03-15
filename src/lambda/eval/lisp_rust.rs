use std::collections::HashMap;
use std::f64;
use std::rc::Rc;
use itertools::Itertools;
use polytype::Type;

use Task;
use lambda::{Expression, Language};
use super::{interp, sexp};

#[derive(Debug)]
pub enum LispError {
    IdentNotFound(String),
    ParseError(::nom::ErrorKind),
    ExpectedBool(&'static str, interp::Value),
    ExpectedInteger(&'static str, interp::Value),
    ExpectedNumber(&'static str, interp::Value),
    ExpectedStr(&'static str, interp::Value),
    ExpectedList(&'static str, interp::Value),
    InvalidLambdaParameter(sexp::Sexp),
    InvalidCall,
    BadArity(&'static str),
    Runtime(&'static str),
}

/// Execute expressions in a simple lisp interpreter.
///
/// Cannot handle:
///
/// - define/let
/// - quote/quasiquote
/// - (escaped) double quotes in a string literal
/// - rational and complex numbers
/// - some obviously sophisticated things (e.g. continuations)
///
/// If you want a more sophisticated lisp interpreter, you can install racket and enable the racket
/// feature which will spawn threads for evaluating expressions with the racket REPL.
///
/// ```toml
/// [dependencies.programinduction]
/// version = "0.2"
/// features = ["racket"]
/// ```
///
/// # Examples
///
/// ```
/// #[macro_use]
/// extern crate polytype;
/// extern crate programinduction;
/// use programinduction::lambda::{Language, LispEvaluator};
///
/// # fn main() {
/// let dsl = Language::uniform(vec![
///     ("map", arrow![arrow![tp!(0), tp!(1)], tp!(list(tp!(0))), tp!(list(tp!(1)))]),
///     ("*2", arrow![tp!(int), tp!(int)]),
///     ("+", arrow![tp!(int), tp!(int), tp!(int)]),
///     ("1", tp!(int)),
/// ]);
/// let lisp = LispEvaluator::new(vec![
///     // only one primitive in our DSL doesn't match what's provided by the interpreter:
///     ("*2", "(λ (x) (* x 2))"),
/// ]);
///
/// let task = lisp.make_task(
///     arrow![tp!(list(tp!(int))), tp!(list(tp!(int)))],
///     &[
///         // these are evaluated along with the expression.
///         ("(list 1 2 3)", "(list 2 4 6)"),
///         ("(list 3 5)", "(list 6 10)"),
///     ],
/// );
///
/// // this expression fails the task
/// let expr = dsl.parse("(λ (map (λ (+ 1 $0)) $0))").expect("parse");
/// assert!((task.oracle)(&dsl, &expr).is_infinite());
/// // this expression succeeds
/// let expr = dsl.parse("(λ (map *2 $0))").expect("parse");
/// assert!((task.oracle)(&dsl, &expr).is_finite());
/// # }
/// ```
pub struct LispEvaluator {
    conversions: HashMap<String, String>,
}
impl LispEvaluator {
    /// Create a lisp evaluator.
    ///
    /// Primitives used in an expression are automatically treated as arbitrary symbols by the
    /// interpreter. So make sure any primitives you need that either are not in scheme or have
    /// different type than in scheme are specified.
    ///
    /// # Examples
    ///
    /// ```
    /// use programinduction::lambda;
    ///
    /// let lisp = lambda::LispEvaluator::new(vec![
    ///     ("+1", "(lambda (x) (+ 1 x))"),
    ///     ("*2", "(λ (x) (* 2 x))"),
    /// ]);
    /// ```
    pub fn new(prims: Vec<(&str, &str)>) -> Self {
        let conversions = prims
            .into_iter()
            .map(|(name, definition)| (String::from(name), String::from(definition)))
            .collect();
        LispEvaluator { conversions }
    }
    /// Check if an expressions matches expected output by evaluating it.
    ///
    /// If input is `None`, the expression is treated as a constant and compared to the output.
    /// Otherwise, the expression is treated as a unary procedure and is applied to the input
    /// before comparison to the output.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use]
    /// # extern crate polytype;
    /// # extern crate programinduction;
    /// # use programinduction::lambda::{Language, LispEvaluator};
    /// # fn main() {
    /// let dsl = Language::uniform(vec![
    ///     ("+", arrow![tp!(int), tp!(int), tp!(int)]),
    ///     ("1", tp!(int)),
    ///     ("2", tp!(int)),
    /// ]);
    /// let lisp = LispEvaluator::default();
    ///
    /// // 2 + 2 == 4
    /// let expr = dsl.parse("(+ 2 2)").expect("parse");
    /// assert!(lisp.check(&dsl, &expr, None, "4").unwrap());
    ///
    /// // 1 + 2 != 4
    /// let expr = dsl.parse("(+ 1 2)").expect("parse");
    /// assert!(!lisp.check(&dsl, &expr, None, "4").unwrap());
    /// # }
    /// ```
    pub fn check(
        &self,
        dsl: &Language,
        expr: &Expression,
        input: Option<&str>,
        output: &str,
    ) -> Result<bool, LispError> {
        let cmd = dsl.lispify(expr, &self.conversions);
        let op = if let Some(inp) = input {
            format!("(equal? ({} {}) {})", cmd, inp, output)
        } else {
            format!("(equal? {} {})", cmd, output)
        };
        let e = sexp::parse(&op)?.eval()?;
        match *e {
            interp::Value::Bool(b) => Ok(b),
            _ => Err(LispError::ExpectedBool(
                "META",
                Rc::try_unwrap(e).unwrap_or_else(|x| (*x).clone()),
            )),
        }
    }
    /// Like [`check`], but checks against multiple input/output pairs.
    ///
    /// Expressions is treated as unary procedures and are applied to each input before comparison
    /// to the corresponding output.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use]
    /// # extern crate polytype;
    /// # extern crate programinduction;
    /// # use programinduction::lambda::{Language, LispEvaluator};
    /// # fn main() {
    /// let dsl = Language::uniform(vec![
    ///     ("map", arrow![arrow![tp!(0), tp!(1)], tp!(list(tp!(0))), tp!(list(tp!(1)))]),
    ///     ("list", arrow![tp!(int), tp!(int), tp!(list(tp!(int)))]),
    ///     ("*2", arrow![tp!(int), tp!(int)]),
    ///     ("+", arrow![tp!(int), tp!(int), tp!(int)]),
    ///     ("1", tp!(int)),
    ///     ("2", tp!(int)),
    /// ]);
    /// let lisp = LispEvaluator::new(vec![
    ///     // only one primitive in our DSL doesn't match what's provided by the interpreter:
    ///     ("*2", "(λ (x) (* x 2))"),
    /// ]);
    ///
    /// let expr = dsl.parse("(λ (map (λ (+ (*2 1) $0)) $0))").expect("parse");
    /// assert!(
    ///     lisp.check_many(&dsl, &expr, &[("(list 1 2)", "(list 3 4)")])
    ///         .expect("evaluation should not fail")
    /// );
    /// # }
    /// ```
    ///
    /// [`check`]: #method.check
    pub fn check_many(
        &self,
        dsl: &Language,
        expr: &Expression,
        examples: &[(&str, &str)],
    ) -> Result<bool, LispError> {
        let cmd = dsl.lispify(expr, &self.conversions);
        let op = format!(
            "(and {})",
            examples
                .iter()
                .map(|&(i, o)| format!("(equal? ({} {}) {})", cmd, i, o))
                .join(" ")
        );
        let e = sexp::parse(&op)?.eval()?;
        match *e {
            interp::Value::Bool(b) => Ok(b),
            _ => Err(LispError::ExpectedBool(
                "META",
                Rc::try_unwrap(e).unwrap_or_else(|x| (*x).clone()),
            )),
        }
    }
    /// Create a task based on evaluating a lisp expressions against test input/output pairs.
    ///
    /// The resulting task is "all-or-nothing": the oracle returns either 0 if all examples are
    /// correctly hit or `f64::NEG_INFINITY` otherwise.
    pub fn make_task<'a>(
        &'a self,
        tp: Type,
        examples: &[(&'a str, &'a str)],
    ) -> Task<'a, Language, Expression, Vec<(String, String)>> {
        let examples: Vec<_> = examples.to_vec();
        let observation: Vec<_> = examples
            .iter()
            .map(|&(inp, out)| (String::from(inp), String::from(out)))
            .collect();
        let oracle = Box::new(move |dsl: &Language, expr: &Expression| -> f64 {
            if self.check_many(dsl, expr, &examples).unwrap_or(false) {
                0f64
            } else {
                f64::NEG_INFINITY
            }
        });
        Task {
            oracle,
            observation,
            tp,
        }
    }
    /// Like [`make_task`], but doesn't treat expressions as unary procedures: they are evaluated
    /// and directly compared against the output.
    ///
    /// [`make_task`]: #method.make_task
    pub fn make_task_output_only<'a>(
        &'a self,
        tp: Type,
        output: &'a str,
    ) -> Task<'a, Language, Expression, String> {
        let observation = String::from(output);
        let oracle = Box::new(move |dsl: &Language, expr: &Expression| -> f64 {
            if self.check(dsl, expr, None, output).unwrap_or(false) {
                0f64
            } else {
                f64::NEG_INFINITY
            }
        });
        Task {
            oracle,
            observation,
            tp,
        }
    }
}
impl Default for LispEvaluator {
    fn default() -> Self {
        let conversions = HashMap::new();
        LispEvaluator { conversions }
    }
}

#[cfg(test)]
mod tests {
    use std::f64::EPSILON;
    use std::rc::Rc;

    use super::sexp::{self, Sexp};
    use super::interp::Value;

    fn list_of_pair(v: Value) -> Vec<Rc<Value>> {
        let mut lst = Vec::new();
        let mut v = Rc::new(v);
        while let Value::Pair(ref car, ref cdr) = *v.clone() {
            lst.push(Rc::clone(car));
            v = Rc::clone(cdr);
        }
        lst.push(v);
        lst
    }

    macro_rules! check_eval {
        ($title: ident, $inp: expr, $pat: pat) => {
            check_eval!($title, $inp, $pat => ());
        };
        ($title: ident, $inp: expr, $pat: pat => $arm: expr) => {
            #[test]
            fn $title() {
                let rv = sexp::parse($inp)
                    .expect(&format!("sexp {}", stringify!($title)))
                    .eval()
                    .expect(&format!("value {}", stringify!($title)));
                match Rc::try_unwrap(rv).unwrap_or_else(|x| (*x).clone()) {
                    $pat => $arm,
                    ref e => panic!(
                        "assertion failed: `{:?}` does not match `{}`",
                        e,
                        stringify!($pat)
                    ),
                }
            }
        };
    }

    check_eval!(literal_bool_1, "#t", Value::Bool(true));
    check_eval!(literal_bool_2, "#f", Value::Bool(false));
    check_eval!(literal_int, "42", Value::Integer(42));
    check_eval!(literal_real, "42.42", Value::Real(f) => assert_eq!(f, 42.42));
    check_eval!(literal_char_1, "#\\,", Value::Char(','));
    check_eval!(literal_char_2, "#\\ ", Value::Char(' '));
    check_eval!(literal_str, "\"some string\"", Value::Str(s) => assert_eq!(s, "some string"));
    check_eval!(literal_null_1, "null", Value::Null);
    check_eval!(literal_null_2, "'()", Value::Null);

    check_eval!(eval_if_1, "(if #t 3 4)", Value::Integer(3));
    check_eval!(eval_if_2, "(if #f 3 4)", Value::Integer(4));

    check_eval!(eval_lambda_1, "(lambda (x) 42)", Value::Lambda(params, body) => {
        assert_eq!(params, vec![String::from("x")]);
        assert_eq!(*body, Sexp::Integer(42));
    });
    check_eval!(eval_lambda_2, "(λ (x) 42)", Value::Lambda(params, body) => {
        assert_eq!(params, vec![String::from("x")]);
        assert_eq!(*body, Sexp::Integer(42));
    });
    check_eval!(eval_lambda_3, "((lambda (x) 42) #t)", Value::Integer(42));
    check_eval!(eval_lambda_4, "((lambda (x) x) 42)", Value::Integer(42));
    check_eval!(
        eval_lambda_5,
        "((lambda (x y) x) 42 52)",
        Value::Integer(42)
    );
    check_eval!(
        eval_lambda_6,
        "((lambda (x y) y) 42 52)",
        Value::Integer(52)
    );

    check_eval!(eval_add_1, "(+ 1)", Value::Integer(1));
    check_eval!(eval_add_2, "(+ 1 2)", Value::Integer(3));
    check_eval!(eval_add_3, "(+ 1 2 3)", Value::Integer(6));
    check_eval!(eval_add_4, "(+ 1 2.2 3)", Value::Real(f) => assert_eq!(f, 6.2));

    check_eval!(eval_sub_1, "(- 1)", Value::Integer(-1));
    check_eval!(eval_sub_2, "(- 3 1)", Value::Integer(2));
    check_eval!(eval_sub_3, "(- 10 4 3 2 1)", Value::Integer(0));

    check_eval!(eval_and_1, "(and)", Value::Bool(true));
    check_eval!(eval_and_2, "(and #t)", Value::Bool(true));
    check_eval!(eval_and_3, "(and #f)", Value::Bool(false));
    check_eval!(eval_and_4, "(and #t #t #t)", Value::Bool(true));
    check_eval!(eval_and_5, "(and #t #t #f)", Value::Bool(false));

    check_eval!(eval_or_1, "(or)", Value::Bool(false));
    check_eval!(eval_or_2, "(or #t)", Value::Bool(true));
    check_eval!(eval_or_3, "(or #f)", Value::Bool(false));
    check_eval!(eval_or_4, "(or #t #t #t)", Value::Bool(true));
    check_eval!(eval_or_5, "(or #t #t #f)", Value::Bool(true));
    check_eval!(eval_or_6, "(or #f #f #t)", Value::Bool(true));
    check_eval!(eval_or_7, "(or #f #f #f)", Value::Bool(false));

    check_eval!(eval_not_1, "(not #f)", Value::Bool(true));
    check_eval!(eval_not_2, "(not #t)", Value::Bool(false));
    check_eval!(eval_not_3, "(not 0)", Value::Bool(false));
    check_eval!(eval_not_4, "(not 1)", Value::Bool(false));
    check_eval!(eval_not_5, "(not null)", Value::Bool(false));
    check_eval!(eval_not_6, "(not (cons 1 2))", Value::Bool(false));
    check_eval!(eval_not_7, "(not #\\,)", Value::Bool(false));
    check_eval!(eval_not_8, "(not \"foo\")", Value::Bool(false));

    check_eval!(eval_mul_1, "(* 5)", Value::Integer(5));
    check_eval!(eval_mul_2, "(* 1 2)", Value::Integer(2));
    check_eval!(eval_mul_3, "(* -4 10)", Value::Integer(-40));
    check_eval!(eval_mul_4, "(* -4.2 10)", Value::Real(f) => assert_eq!(f, -42.0));
    check_eval!(eval_mul_5, "(* 4.2 -10.5)", Value::Real(f) => assert_eq!(f, -44.1));

    check_eval!(eval_div_1, "(/ 10)", Value::Real(f) => assert_eq!(f, 0.1));
    check_eval!(eval_div_2, "(/ 4 2)", Value::Integer(2));
    check_eval!(eval_div_3, "(/ 5 2)", Value::Integer(2));
    check_eval!(eval_div_4, "(/ -4 10)", Value::Integer(0));
    check_eval!(eval_div_5, "(/ -14 10)", Value::Integer(-1));
    check_eval!(eval_div_6, "(/ -4.2 10)", Value::Real(f) => assert!((f + 0.42).abs() < EPSILON));
    check_eval!(eval_div_7, "(/ 4.2 -10)", Value::Real(f) => assert!((f + 0.42).abs() < EPSILON));
    check_eval!(eval_div_8, "(/ 200 10 10 2)", Value::Integer(1));

    check_eval!(eval_quotient_1, "(quotient 10 3)", Value::Integer(3));
    check_eval!(eval_quotient_2, "(quotient -10.0 3)", Value::Real(f) => assert_eq!(f, -3.0));
    check_eval!(eval_rem_1, "(remainder 10 3)", Value::Integer(1));
    check_eval!(eval_rem_2, "(remainder -10.0 3)", Value::Real(f) => assert_eq!(f, -1.0));
    check_eval!(eval_rem_3, "(remainder 10.0 -3)", Value::Real(f) => assert_eq!(f, 1.0));
    check_eval!(eval_rem_4, "(remainder -10 -3)", Value::Integer(-1));
    check_eval!(eval_mod_1, "(modulo 10 3)", Value::Integer(1));
    check_eval!(eval_mod_2, "(modulo -10.0 3)", Value::Real(f) => assert_eq!(f, 2.0));
    check_eval!(eval_mod_3, "(modulo 10.0 -3)", Value::Real(f) => assert_eq!(f, -2.0));
    check_eval!(eval_mod_4, "(modulo -10 -3)", Value::Integer(-1));

    check_eval!(eval_cons_1, "(cons 1 2)", Value::Pair(car, cdr) => {
        match *car {
            Value::Integer(1) => (),
            ref e => panic!("assertion failed: `{:?}` does not match Value::Integer(1)", e),
        }
        match *cdr {
            Value::Integer(2) => (),
            ref e => panic!("assertion failed: `{:?}` does not match Value::Integer(1)", e),
        }
    });
    check_eval!(eval_cons_2, "(cons 1 null)", Value::Pair(car, cdr) => {
        match *car {
            Value::Integer(1) => (),
            ref e => panic!("assertion failed: `{:?}` does not match Value::Integer(1)", e),
        }
        match *cdr {
            Value::Null => (),
            ref e => panic!("assertion failed: `{:?}` does not match Value::Null", e),
        }
    });
    check_eval!(eval_cons_3, "(cons 1 (cons 2 3))", Value::Pair(car, cdr) => {
        match *car {
            Value::Integer(1) => (),
            ref e => panic!("assertion failed: `{:?}` does not match Value::Integer(1)", e),
        }
        match *cdr {
            Value::Pair(ref cadr, ref cddr) => {
                match **cadr {
                    Value::Integer(2) => (),
                    ref e => panic!("assertion failed: `{:?}` does not match Value::Integer(2)", e),
                }
                match **cddr {
                    Value::Integer(3) => (),
                    ref e => panic!("assertion failed: `{:?}` does not match Value::Integer(3)", e),
                }
            }
            ref e => panic!("assertion failed: `{:?}` does not match Value::Pair(_, _)", e),
        }
    });

    check_eval!(eval_car, "(car (cons 1 2))", Value::Integer(1));
    check_eval!(eval_cdr, "(cdr (cons 1 2))", Value::Integer(2));

    check_eval!(eval_eq_1, "(eq? (cons 1 2) (cons 1 2))", Value::Bool(true));
    check_eval!(eval_eq_2, "(eq? (cons 1 2) (cons 1 3))", Value::Bool(false));
    check_eval!(eval_eq_3, "(eq? (cons 1 3) (cons 1 2))", Value::Bool(false));
    check_eval!(
        eval_eq_4,
        "(eq? (cons 0.5 (cons 2 null)) (list 0.5 2))",
        Value::Bool(true)
    );
    check_eval!(
        eval_eq_5,
        "(eq? (cons 1.1 (cons 2 null)) (list 1.1 2 null))",
        Value::Bool(false)
    );
    check_eval!(
        eval_eq_6,
        "(eq? (cons \"foo\" #t) (cons \"foo\" #t))",
        Value::Bool(true)
    );
    check_eval!(
        eval_eq_7,
        "(eq? (cons \"foo\" #t) (cons \"bar\" #t))",
        Value::Bool(false)
    );
    check_eval!(
        eval_eq_8,
        "(eq? (list #t 2 0.5 #\\, \"foo\") (list #t 2 0.5 #\\, \"foo\"))",
        Value::Bool(true)
    );
    check_eval!(
        eval_eq_9,
        "(eq? (list #t 2 0.5 #\\, \"foo\") (list #t 2 0.5 #\\. \"foo\"))",
        Value::Bool(false)
    );

    check_eval!(eval_eq_10, "(= 1 1)", Value::Bool(true));
    check_eval!(eval_eq_11, "(= 1 2)", Value::Bool(false));
    check_eval!(eval_eq_12, "(= 2 1)", Value::Bool(false));
    check_eval!(eval_eq_13, "(= 1.0 1.0)", Value::Bool(true));
    check_eval!(eval_eq_14, "(= 1.0 2.0)", Value::Bool(false));
    check_eval!(eval_eq_15, "(= 2.0 1.0)", Value::Bool(false));
    check_eval!(eval_gt_1, "(> 1 1)", Value::Bool(false));
    check_eval!(eval_gt_2, "(> 1 2)", Value::Bool(false));
    check_eval!(eval_gt_3, "(> 2 1)", Value::Bool(true));
    check_eval!(eval_gt_4, "(> 1.0 1.0)", Value::Bool(false));
    check_eval!(eval_gt_5, "(> 1.0 2.0)", Value::Bool(false));
    check_eval!(eval_gt_6, "(> 2.0 1.0)", Value::Bool(true));
    check_eval!(eval_gte_1, "(>= 1 1)", Value::Bool(true));
    check_eval!(eval_gte_2, "(>= 1 2)", Value::Bool(false));
    check_eval!(eval_gte_3, "(>= 2 1)", Value::Bool(true));
    check_eval!(eval_gte_4, "(>= 1.0 1.0)", Value::Bool(true));
    check_eval!(eval_gte_5, "(>= 1.0 2.0)", Value::Bool(false));
    check_eval!(eval_gte_6, "(>= 2.0 1.0)", Value::Bool(true));
    check_eval!(eval_lt_1, "(< 1 1)", Value::Bool(false));
    check_eval!(eval_lt_2, "(< 1 2)", Value::Bool(true));
    check_eval!(eval_lt_3, "(< 2 1)", Value::Bool(false));
    check_eval!(eval_lt_4, "(< 1.0 1.0)", Value::Bool(false));
    check_eval!(eval_lt_5, "(< 1.0 2.0)", Value::Bool(true));
    check_eval!(eval_lt_6, "(< 2.0 1.0)", Value::Bool(false));
    check_eval!(eval_lte_1, "(<= 1 1)", Value::Bool(true));
    check_eval!(eval_lte_2, "(<= 1 2)", Value::Bool(true));
    check_eval!(eval_lte_3, "(<= 2 1)", Value::Bool(false));
    check_eval!(eval_lte_4, "(<= 1.0 1.0)", Value::Bool(true));
    check_eval!(eval_lte_5, "(<= 1.0 2.0)", Value::Bool(true));
    check_eval!(eval_lte_6, "(<= 2.0 1.0)", Value::Bool(false));
    check_eval!(eval_neq_1, "(!= 1 1)", Value::Bool(false));
    check_eval!(eval_neq_2, "(!= 1 2)", Value::Bool(true));
    check_eval!(eval_neq_3, "(!= 2 1)", Value::Bool(true));
    check_eval!(eval_neq_4, "(!= 1.0 1.0)", Value::Bool(false));
    check_eval!(eval_neq_5, "(!= 1.0 2.0)", Value::Bool(true));
    check_eval!(eval_neq_6, "(!= 2.0 1.0)", Value::Bool(true));

    check_eval!(eval_is_pair_1, "(pair? (cons 1 2))", Value::Bool(true));
    check_eval!(eval_is_pair_2, "(pair? 1)", Value::Bool(false));
    check_eval!(eval_is_null_1, "(null? (cons 1 2))", Value::Bool(false));
    check_eval!(eval_is_null_2, "(null? null)", Value::Bool(true));
    check_eval!(eval_is_null_3, "(null? (list))", Value::Bool(true));
    check_eval!(eval_identity, "(identity null)", Value::Null);
    // list is tested indirectly via other tests
    check_eval!(
        eval_list_ref_1,
        "(list-ref (list 1 2 3) 2)",
        Value::Integer(3)
    );
    check_eval!(
        eval_list_ref_2,
        "(list-ref (cons 1 2) 0)",
        Value::Integer(1)
    );
    check_eval!(
        eval_list_ref_3,
        "(list-ref (list null \"foo\" #t) 1)",
        Value::Str(s) => assert_eq!(s, "foo")
    );

    check_eval!(eval_append_1, "(append)", Value::Null);
    check_eval!(eval_append_2, "(append null)", Value::Null);
    check_eval!(
        eval_append_3,
        "(eq? (append (list 1 2) (list 3 4)) (list 1 2 3 4))",
        Value::Bool(true)
    );
    check_eval!(
        eval_append_4,
        "(eq? (append (list 1 2) (list 3 4)) (list 1 2 3 4))",
        Value::Bool(true)
    );
    check_eval!(
        eval_append_5,
        "(eq? (append (list 1 2) (list 3 4) (list 5 6) (list 7 8)) (list 1 2 3 4 5 6 7 8))",
        Value::Bool(true)
    );
    check_eval!(
        eval_appendmap,
        "(eq? (append-map (λ (e) (list e e)) (list 1 2 3)) (list 1 1 2 2 3 3))",
        Value::Bool(true)
    );
    check_eval!(
        eval_map_1,
        "(eq? (map (lambda (x) (+ 1 x)) (list 2 3 4)) (list 3 4 5))",
        Value::Bool(true)
    );
    check_eval!(
        eval_map_2,
        "(eq? (map (lambda (x) (+ 1 x)) null) null)",
        Value::Bool(true)
    );
    check_eval!(
        eval_map_3,
        "(eq? (map (lambda (x) (* (+ 1 x) -2)) (list 2 3 4)) (list -6 -8 -10))",
        Value::Bool(true)
    );
    check_eval!(
        eval_filter_1,
        "(eq? (filter (lambda (x) (> x 2)) (list 2 3 4)) (list 3 4))",
        Value::Bool(true)
    );
    check_eval!(
        eval_filter_2,
        "(eq? (filter (lambda (x) (= (modulo x 2) 0)) (list 1 2 3 4)) (list 2 4))",
        Value::Bool(true)
    );
    check_eval!(
        eval_foldl_1,
        "(eq? (foldl cons '() (list 1 2 3 4)) (list 4 3 2 1))",
        Value::Bool(true)
    );
    check_eval!(
        eval_foldl_2,
        "(eq? (foldl + 0 (list 1 2 3 4)) 10)",
        Value::Bool(true)
    );
    check_eval!(
        eval_count_1,
        "(eq? (count (lambda (x) (> x 2)) (list 2 3 4)) 2)",
        Value::Bool(true)
    );
    check_eval!(
        eval_count_2,
        "(eq? (count (lambda (x) (= (modulo x 2) 0)) (list 1 2 3 4)) 2)",
        Value::Bool(true)
    );
    check_eval!(
        eval_member_1,
        "(eq? (member 2 (list 1 2 3 4)) (list 2 3 4))",
        Value::Bool(true)
    );
    check_eval!(
        eval_member_2,
        "(not (member 9 (list 1 2 3 4)))",
        Value::Bool(true)
    );

    check_eval!(
        eval_string_length_1,
        "(string-length \"foo\")",
        Value::Integer(3)
    );
    check_eval!(
        eval_string_length_2,
        "(string-length \"\")",
        Value::Integer(0)
    );
    check_eval!(
        eval_string_length_3,
        "(string-length \"π\")",
        Value::Integer(1) // 2 bytes, 1 unicode grapheme cluster
    );

    check_eval!(
        eval_string_downcase,
        "(string-downcase \"lIsP\")",
        Value::Str(s) => assert_eq!(s, "lisp")
    );
    check_eval!(
        eval_string_upcase,
        "(string-upcase \"lIsP\")",
        Value::Str(s) => assert_eq!(s, "LISP")
    );
    check_eval!(
        eval_string_append_1,
        "(string-append)",
        Value::Str(s) => assert_eq!(s, "")
    );
    check_eval!(
        eval_string_append_2,
        "(string-append \"foo\")",
        Value::Str(s) => assert_eq!(s, "foo")
    );
    check_eval!(
        eval_string_append_3,
        "(string-append \"foo\" \"bar\")",
        Value::Str(s) => assert_eq!(s, "foobar")
    );
    check_eval!(
        eval_string_append_4,
        "(string-append \"foo\" \"bar\" \"baz\")",
        Value::Str(s) => assert_eq!(s, "foobarbaz")
    );
    check_eval!(
        eval_substring_1,
        "(substring \"foo\" 0)",
        Value::Str(s) => assert_eq!(s, "foo")
    );
    check_eval!(
        eval_substring_2,
        "(substring \"foo\" 1)",
        Value::Str(s) => assert_eq!(s, "oo")
    );
    check_eval!(
        eval_substring_3,
        "(substring \"bar\" 1 2)",
        Value::Str(s) => assert_eq!(s, "a")
    );
    check_eval!(
        eval_substring_4,
        "(substring \"bar\" 1 3)",
        Value::Str(s) => assert_eq!(s, "ar")
    );
    check_eval!(
        eval_string_split_1,
        "(string-split \"foo\")",
        p @ Value::Pair(_, _) => {
            let xs = list_of_pair(p);
            assert_eq!(xs.len(), 2);
            match *xs[0] {
                Value::Str(ref s) => assert_eq!(s, "foo"),
                ref e => panic!("assertion failed: `{:?}` does not match Value::Str(..)", e),
            }
            match *xs[1] {
                Value::Null => (),
                ref e => panic!("assertion failed: `{:?}` does not match Value::Null", e),
            }
        }
    );
    check_eval!(
        eval_string_split_2,
        "(string-split \"foo bar  baz	buzz\")",
        p @ Value::Pair(_, _) => {
            let xs = list_of_pair(p);
            assert_eq!(xs.len(), 5);
            match *xs[0] {
                Value::Str(ref s) => assert_eq!(s, "foo"),
                ref e => panic!("assertion failed: `{:?}` does not match Value::Str(..)", e),
            }
            match *xs[1] {
                Value::Str(ref s) => assert_eq!(s, "bar"),
                ref e => panic!("assertion failed: `{:?}` does not match Value::Str(..)", e),
            }
            match *xs[2] {
                Value::Str(ref s) => assert_eq!(s, "baz"),
                ref e => panic!("assertion failed: `{:?}` does not match Value::Str(..)", e),
            }
            match *xs[3] {
                Value::Str(ref s) => assert_eq!(s, "buzz"),
                ref e => panic!("assertion failed: `{:?}` does not match Value::Str(..)", e),
            }
            match *xs[4] {
                Value::Null => (),
                ref e => panic!("assertion failed: `{:?}` does not match Value::Null", e),
            }
        }
    );
    check_eval!(
        eval_string_split_3,
        "(string-split \"foo_bar__baz_buzz\" \"_\")",
        p @ Value::Pair(_, _) => {
            let xs = list_of_pair(p);
            assert_eq!(xs.len(), 6);
            match *xs[0] {
                Value::Str(ref s) => assert_eq!(s, "foo"),
                ref e => panic!("assertion failed: `{:?}` does not match Value::Str(..)", e),
            }
            match *xs[1] {
                Value::Str(ref s) => assert_eq!(s, "bar"),
                ref e => panic!("assertion failed: `{:?}` does not match Value::Str(..)", e),
            }
            match *xs[2] {
                Value::Str(ref s) => assert_eq!(s, ""),
                ref e => panic!("assertion failed: `{:?}` does not match Value::Str(..)", e),
            }
            match *xs[3] {
                Value::Str(ref s) => assert_eq!(s, "baz"),
                ref e => panic!("assertion failed: `{:?}` does not match Value::Str(..)", e),
            }
            match *xs[4] {
                Value::Str(ref s) => assert_eq!(s, "buzz"),
                ref e => panic!("assertion failed: `{:?}` does not match Value::Str(..)", e),
            }
            match *xs[5] {
                Value::Null => (),
                ref e => panic!("assertion failed: `{:?}` does not match Value::Null", e),
            }
        }
    );

    check_eval!(
        eval_string_join_1,
        "(string-join null)",
        Value::Str(s) => assert_eq!(s, "")
    );
    check_eval!(
        eval_string_join_2,
        "(string-join (list \"foo\"))",
        Value::Str(s) => assert_eq!(s, "foo")
    );
    check_eval!(
        eval_string_join_3,
        "(string-join (list \"foo\" \"bar\"))",
        Value::Str(s) => assert_eq!(s, "foobar")
    );
    check_eval!(
        eval_string_join_4,
        "(string-join null \"_\")",
        Value::Str(s) => assert_eq!(s, "")
    );
    check_eval!(
        eval_string_join_5,
        "(string-join (list \"foo\") \"_\")",
        Value::Str(s) => assert_eq!(s, "foo")
    );
    check_eval!(
        eval_string_join_6,
        "(string-join (list \"foo\" \"bar\") \"_\")",
        Value::Str(s) => assert_eq!(s, "foo_bar")
    );
}
