use std::collections::HashMap;
use std::f64;
use std::io::{self, BufRead, BufReader, Write};
use std::process::{ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc::channel;
use polytype::Type;
use workerpool::{Pool, Worker};
use super::super::Task;
use super::{Expression, Language};

/// Execute expressions in the Racket runtime for evaluation.
///
/// Racket must be installed and available in the `PATH`.
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
///     // only one primitive in our DSL doesn't match what's provided by racket:
///     ("*2", "(λ (x) (* x 2))"),
/// ]);
///
/// let task = lisp.make_task(
///     arrow![tp!(list(tp!(int))), tp!(list(tp!(int)))],
///     vec![
///         // create a task using whichever lisp syntax.
///         // these are evaluated along with the expression.
///         (Some("(list 1 2 3)"), "(list 2 4 6)"),
///         (Some("'(3 5)"), "'(6 10)"),
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
    pool: Pool<Racket>,
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
    /// let lisp = lambda::LispEvaluator::new(
    ///     vec![
    ///         ("+1", "(lambda (x) (+ 1 x))"),
    ///         ("*2", "(λ (x) (* 2 x))"),
    ///     ]
    /// );
    /// ```
    pub fn new(prims: Vec<(&str, &str)>) -> Self {
        let conversions = prims
            .into_iter()
            .map(|(name, definition)| (String::from(name), String::from(definition)))
            .collect();
        let pool = Pool::<Racket>::default();
        LispEvaluator { conversions, pool }
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
    /// #[macro_use]
    /// extern crate polytype;
    /// extern crate programinduction;
    /// use programinduction::lambda::{Language, LispEvaluator};
    ///
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
    ///     // only one primitive in our DSL doesn't match what's provided by racket:
    ///     ("*2", "(λ (x) (* x 2))"),
    /// ]);
    ///
    /// let expr = dsl.parse("(λ (map (λ (+ (*2 1) $0)) $0))").expect("parse");
    /// assert!(
    ///     lisp.check(&dsl, &expr, Some("(list 1 2)"), "(list 3 4)")
    ///         .expect("evaluation should not fail")
    /// );
    /// # }
    /// ```
    pub fn check(
        &self,
        dsl: &Language,
        expr: &Expression,
        input: Option<&str>,
        output: &str,
    ) -> Result<bool, io::Error> {
        let cmd = dsl.lispify(expr, &self.conversions);
        let op = if let Some(inp) = input {
            format!("(equal? ({} {}) {})", cmd, inp, output)
        } else {
            format!("(equal? {} {})", cmd, output)
        };
        let (tx, rx) = channel();
        self.pool.execute_to(tx, op.clone());
        let response = rx.recv().expect("receive")?;
        match &*response {
            "#t\n" => Ok(true),
            "#f\n" => Ok(false),
            _ => Err(io::Error::new(io::ErrorKind::Other, response)),
        }
    }
    /// Create a task based on evaluating a lisp expressions against test input/output pairs.
    ///
    /// The resulting task is "all-or-nothing": the oracle returns either 0 if all examples are
    /// correctly hit or `f64::NEG_INFINITY` otherwise.
    pub fn make_task<'a>(
        &'a self,
        tp: Type,
        examples: Vec<(Option<&str>, &str)>,
    ) -> Task<'a, Language, Vec<(Option<String>, String)>> {
        let examples: Vec<_> = examples
            .into_iter()
            .map(|(inp, out)| (inp.map(String::from), String::from(out)))
            .collect();
        let examples_oracle = examples.clone();
        let oracle = Box::new(move |dsl: &Language, expr: &Expression| -> f64 {
            let success = examples_oracle.iter().all(|&(ref inp, ref out)| {
                self.check(dsl, expr, inp.as_ref().map(|x| &**x), out)
                    .unwrap_or(false)
            });
            if success {
                0f64
            } else {
                f64::NEG_INFINITY
            }
        });
        Task {
            oracle,
            observation: examples,
            tp,
        }
    }
}

/// Maintains the interactive connection to a racket runtime.
struct Racket {
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}
impl Default for Racket {
    fn default() -> Self {
        let child = Command::new("racket")
            .arg("-e")
            .arg(
                "(let lp ()
                    (with-handlers ([exn:fail? (λ (exn) (displayln \"ERROR\"))])
                        (displayln (eval (read))))
                    (flush-output)
                    (lp))",
            )
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("could not spawn racket process");
        let stdin = child.stdin.expect("connect to racket stdin");
        let stdout = child.stdout.expect("connect to racket stdout");
        let stdout = BufReader::new(stdout);
        Racket { stdin, stdout }
    }
}
impl Worker for Racket {
    type Input = String;
    type Output = Result<String, io::Error>;

    fn execute(&mut self, op: Self::Input) -> Self::Output {
        self.stdin.write_all(op.as_bytes())?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;
        let mut s = String::new();
        self.stdout.read_line(&mut s)?;
        Ok(s)
    }
}
