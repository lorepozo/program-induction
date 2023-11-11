//! The string editing domain, for tackling Flashfill-style problems.
//!
//! # Examples
//!
//! ```ignore
//! use programinduction::domains::strings;
//! use programinduction::{ECParams, EC};
//!
//! let dsl = strings::dsl();
//! let tasks = strings::make_tasks(250, 4);
//! let ec_params = ECParams {
//!     frontier_limit: 10,
//!     search_limit_timeout: None,
//!     search_limit_description_length: Some(15.0),
//! };
//!
//! let frontiers = dsl.explore(&ec_params, &tasks);
//! let hits = frontiers.iter().filter_map(|f| f.best_solution()).count();
//! assert!(50 < hits && hits < 80, "hits = {}", hits);
//! ```

use itertools::Itertools;
use once_cell::sync::Lazy;
use polytype::{ptp, tp};
use std::collections::HashMap;
use std::f64;
use std::fmt;

use crate::lambda::{Evaluator as EvaluatorT, Expression, Language, LiftedFunction};
use crate::Task;

/// The string editing [`lambda::Language`] defines the following operations:
///
/// ```compile_fails
/// "0":         ptp!(int)
/// "+1":        ptp!(@arrow[tp!(int), tp!(int)])
/// "-1":        ptp!(@arrow[tp!(int), tp!(int)])
/// "len":       ptp!(@arrow[tp!(str), tp!(int)])
/// "empty_str": ptp!(str)
/// "lower":     ptp!(@arrow[tp!(str), tp!(str)])
/// "upper":     ptp!(@arrow[tp!(str), tp!(str)])
/// "concat":    ptp!(@arrow[tp!(str), tp!(str), tp!(str)])
/// "slice":     ptp!(@arrow[tp!(int), tp!(int), tp!(str), tp!(str)])
/// "nth":       ptp!(@arrow[tp!(int), tp!(list(tp!(str))), tp!(str)])
/// "map":       ptp!(0, 1; @arrow[
///                  tp!(@arrow[tp!(0), tp!(1)]),
///                  tp!(list(tp!(0))),
///                  tp!(list(tp!(1))),
///              ])
/// "strip":     ptp!(@arrow[tp!(str), tp!(str)])
/// "split":     ptp!(@arrow[tp!(char), tp!(str), tp!(list(tp!(str)))])
/// "join":      ptp!(@arrow[tp!(str), tp!(list(tp!(str))), tp!(str)])
/// "char->str": ptp!(@arrow[tp!(char), tp!(str)])
/// "space":     ptp!(char)
/// ".":         ptp!(char)
/// ",":         ptp!(char)
/// "<":         ptp!(char)
/// ">":         ptp!(char)
/// "/":         ptp!(char)
/// "@":         ptp!(char)
/// "-":         ptp!(char)
/// "|":         ptp!(char)
/// ```
///
/// [`lambda::Language`]: ../../lambda/struct.Language.html
pub fn dsl() -> Language {
    let mut dsl = Language::uniform(vec![
        ("0", ptp!(int)),
        ("+1", ptp!(@arrow[tp!(int), tp!(int)])),
        ("-1", ptp!(@arrow[tp!(int), tp!(int)])),
        ("len", ptp!(@arrow[tp!(str), tp!(int)])),
        ("empty_str", ptp!(str)),
        ("lower", ptp!(@arrow[tp!(str), tp!(str)])),
        ("upper", ptp!(@arrow[tp!(str), tp!(str)])),
        ("concat", ptp!(@arrow[tp!(str), tp!(str), tp!(str)])),
        (
            "slice",
            ptp!(@arrow[tp!(int), tp!(int), tp!(str), tp!(str)]),
        ),
        ("nth", ptp!(@arrow[tp!(int), tp!(list(tp!(str))), tp!(str)])),
        (
            "map",
            ptp!(0, 1; @arrow[tp!(@arrow[tp!(0), tp!(1)]), tp!(list(tp!(0))), tp!(list(tp!(1)))]),
        ),
        ("strip", ptp!(@arrow[tp!(str), tp!(str)])),
        (
            "split",
            ptp!(@arrow[tp!(char), tp!(str), tp!(list(tp!(str)))]),
        ),
        (
            "join",
            ptp!(@arrow[tp!(str), tp!(list(tp!(str))), tp!(str)]),
        ),
        ("char->str", ptp!(@arrow[tp!(char), tp!(str)])),
        ("space", ptp!(char)),
        (".", ptp!(char)),
        (",", ptp!(char)),
        ("<", ptp!(char)),
        (">", ptp!(char)),
        ("/", ptp!(char)),
        ("@", ptp!(char)),
        ("-", ptp!(char)),
        ("|", ptp!(char)),
    ]);
    // disallow (+1 (-1 _)) and (-1 (+1 _))
    dsl.add_symmetry_violation(1, 0, 2);
    dsl.add_symmetry_violation(2, 0, 1);
    // disallow len of empty_str, (lower _), (upper _), and (char->str _)
    dsl.add_symmetry_violation(3, 0, 4);
    dsl.add_symmetry_violation(3, 0, 5);
    dsl.add_symmetry_violation(3, 0, 6);
    dsl.add_symmetry_violation(3, 0, 14);
    // disallow (concat (concat ..) _) in favor of (concat _ (concat ..))
    dsl.add_symmetry_violation(7, 0, 7);
    // disallow concat of empty_str
    dsl.add_symmetry_violation(7, 0, 4);
    dsl.add_symmetry_violation(7, 1, 4);
    dsl
}

use self::Space::*;
/// All values in the strings domain can be represented in this `Space`.
#[derive(Clone)]
pub enum Space {
    Num(i32),
    Char(char),
    Str(String),
    List(Vec<Space>),
    Func(LiftedFunction<Space, Evaluator>),
}
impl fmt::Debug for Space {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            Num(x) => write!(f, "Num({:?})", x),
            Char(x) => write!(f, "Char({:?})", x),
            Str(ref x) => write!(f, "Str({:?})", x),
            List(ref x) => write!(f, "List({:?})", x),
            Func(_) => write!(f, "<function>"),
        }
    }
}
impl PartialEq for Space {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (&Num(x), &Num(y)) => x == y,
            (&Char(x), &Char(y)) => x == y,
            (Str(x), Str(y)) => x == y,
            (List(xs), List(ys)) => xs == ys,
            _ => false,
        }
    }
}
/// An [`Evaluator`] for the strings domain.
///
/// # Examples
///
/// ```
/// use polytype::{ptp, tp};
/// use programinduction::domains::strings;
/// use programinduction::{lambda, ECParams, EC};
///
/// # fn main() {
/// let dsl = strings::dsl();
/// let examples = vec![
///     // Replace delimiter '>' with '/'
///     (
///         vec![strings::Space::Str("OFJQc>BLVP>eMS".to_string())],
///         strings::Space::Str("OFJQc/BLVP/eMS".to_string()),
///     ),
/// ];
/// let task = lambda::task_by_evaluation(
///     strings::Evaluator,
///     ptp!(@arrow[tp!(str), tp!(str)]),
///     &examples,
/// );
///
/// let ec_params = ECParams {
///     frontier_limit: 10,
///     search_limit_timeout: None,
///     search_limit_description_length: Some(12.0),
/// };
/// let frontiers = dsl.explore(&ec_params, &[task]);
/// let solution = &frontiers[0].best_solution().unwrap().0;
/// assert_eq!(
///     "(λ (join (char->str /) (split > $0)))",
///     dsl.display(solution)
/// );
/// # }
/// ```
///
/// [`Evaluator`]: ../../lambda/trait.Evaluator.html
#[derive(Copy, Clone)]
pub struct Evaluator;
impl EvaluatorT for Evaluator {
    type Space = Space;
    type Error = ();
    fn evaluate(&self, name: &str, inps: &[Self::Space]) -> Result<Self::Space, Self::Error> {
        match OPERATIONS[name] {
            Op::Zero => Ok(Num(0)),
            Op::Incr => match inps[0] {
                Num(x) => Ok(Num(x + 1)),
                _ => unreachable!(),
            },
            Op::Decr => match inps[0] {
                Num(x) => Ok(Num(x - 1)),
                _ => unreachable!(),
            },
            Op::Len => match inps[0] {
                Str(ref s) => Ok(Num(s.len() as i32)),
                _ => unreachable!(),
            },
            Op::Empty => Ok(Str(String::new())),
            Op::Lower => match inps[0] {
                Str(ref s) => Ok(Str(s.to_lowercase())),
                _ => unreachable!(),
            },
            Op::Upper => match inps[0] {
                Str(ref s) => Ok(Str(s.to_uppercase())),
                _ => unreachable!(),
            },
            Op::Concat => match (&inps[0], &inps[1]) {
                (Str(x), Str(y)) => {
                    let mut s = x.to_string();
                    s.push_str(y);
                    Ok(Str(s))
                }
                _ => unreachable!(),
            },
            Op::Slice => match (&inps[0], &inps[1], &inps[2]) {
                (&Num(x), &Num(y), Str(s)) => {
                    if x as usize > s.len() || y < x {
                        Err(())
                    } else {
                        Ok(Str(s
                            .chars()
                            .skip(x as usize)
                            .take((y - x) as usize)
                            .collect()))
                    }
                }
                _ => unreachable!(),
            },
            Op::Nth => match (&inps[0], &inps[1]) {
                (&Num(x), List(ss)) => ss.get(x as usize).cloned().ok_or(()),
                _ => unreachable!(),
            },
            Op::Map => match (&inps[0], &inps[1]) {
                (Func(f), List(xs)) => Ok(List(
                    xs.iter()
                        .map(|x| f.eval(&[x.clone()]).map_err(|_| ()))
                        .collect::<Result<_, _>>()?,
                )),
                _ => unreachable!(),
            },
            Op::Strip => match inps[0] {
                Str(ref s) => Ok(Str(s.trim().to_owned())),
                _ => unreachable!(),
            },
            Op::Split => match (&inps[0], &inps[1]) {
                (&Char(c), Str(s)) => Ok(List(s.split(c).map(|s| Str(s.to_owned())).collect())),
                _ => unreachable!(),
            },
            Op::Join => match (&inps[0], &inps[1]) {
                (Str(delim), List(ss)) => Ok(Str(ss
                    .iter()
                    .map(|s| match *s {
                        Str(ref s) => s,
                        _ => unreachable!(),
                    })
                    .join(delim))),
                _ => unreachable!(),
            },
            Op::CharToStr => match inps[0] {
                Char(c) => Ok(Str(c.to_string())),
                _ => unreachable!(),
            },
            Op::CharSpace => Ok(Char(' ')),
            Op::CharDot => Ok(Char('.')),
            Op::CharComma => Ok(Char(',')),
            Op::CharLess => Ok(Char('<')),
            Op::CharGreater => Ok(Char('>')),
            Op::CharSlash => Ok(Char('/')),
            Op::CharAt => Ok(Char('@')),
            Op::CharDash => Ok(Char('-')),
            Op::CharPipe => Ok(Char('|')),
        }
    }
    fn lift(&self, f: LiftedFunction<Self::Space, Self>) -> Result<Self::Space, ()> {
        Ok(Func(f))
    }
}

/// Randomly generate string editing [`Task`]s.
///
/// The task observations input/output pairs, where sequentially-applied inputs are gathered into a
/// list.
///
/// [`Task`]: ../../struct.Task.html
#[allow(clippy::type_complexity)]
pub fn make_tasks(
    count: usize,
    n_examples: usize,
) -> Vec<Task<'static, Language, Expression, Vec<(Vec<Space>, Space)>>> {
    (0..=count / 1467) // make_examples yields 1467 tasks
        .flat_map(|_| make_examples(n_examples))
        .take(count)
        .map(|(_name, tp, examples)| {
            let evaluator = ::std::sync::Arc::new(Evaluator);
            let oracle_examples = examples.clone();
            let oracle = Box::new(move |dsl: &Language, expr: &Expression| -> f64 {
                let success = oracle_examples.iter().all(|(inps, out)| {
                    if let Ok(o) = dsl.eval_arc(expr, &evaluator, inps) {
                        o == *out
                    } else {
                        false
                    }
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
        })
        .collect()
}

/// Using an enum with a hashmap will be much faster than string comparisons.
enum Op {
    Zero,
    Incr,
    Decr,
    Len,
    Empty,
    Lower,
    Upper,
    Concat,
    Slice,
    Nth,
    Map,
    Strip,
    Split,
    Join,
    CharToStr,
    CharSpace,
    CharDot,
    CharComma,
    CharLess,
    CharGreater,
    CharSlash,
    CharAt,
    CharDash,
    CharPipe,
}

static OPERATIONS: Lazy<HashMap<&'static str, Op>> = Lazy::new(|| {
    HashMap::from([
        ("0", Op::Zero),
        ("+1", Op::Incr),
        ("-1", Op::Decr),
        ("len", Op::Len),
        ("empty_str", Op::Empty),
        ("lower", Op::Lower),
        ("upper", Op::Upper),
        ("concat", Op::Concat),
        ("slice", Op::Slice),
        ("nth", Op::Nth),
        ("map", Op::Map),
        ("strip", Op::Strip),
        ("split", Op::Split),
        ("join", Op::Join),
        ("char->str", Op::CharToStr),
        ("space", Op::CharSpace),
        (".", Op::CharDot),
        (",", Op::CharComma),
        ("<", Op::CharLess),
        (">", Op::CharGreater),
        ("/", Op::CharSlash),
        ("@", Op::CharAt),
        ("-", Op::CharDash),
        ("|", Op::CharPipe),
    ])
});

use self::gen::make_examples;
mod gen {
    use itertools::Itertools;
    use polytype::{ptp, tp, TypeSchema};
    use rand::distributions::{Distribution, Uniform};
    use rand::{self, Rng};

    use super::Space::{self, *};

    static DELIMITERS: [char; 9] = ['.', ',', ' ', '<', '>', '/', '@', '-', '|'];

    fn character<R: Rng>(rng: &mut R) -> char {
        let c: u8 = Uniform::from(0..26u8).sample(rng);
        let c = c + if rng.gen() { 65 } else { 97 };
        c as char
    }

    fn word<R: Rng>(rng: &mut R) -> String {
        let size = Uniform::from(3..6).sample(rng);
        (0..size).map(|_| character(rng)).collect()
    }
    fn words<R: Rng>(delim: char, rng: &mut R) -> String {
        let size = Uniform::from(2..5).sample(rng);
        (0..size).map(|_| word(rng)).join(&delim.to_string())
    }

    fn white_word<R: Rng>(rng: &mut R) -> String {
        let size = Uniform::from(4..7).sample(rng);
        let mut s: String = (0..size).map(|_| character(rng)).collect();
        let n_spaces = Uniform::from(0..3).sample(rng);
        for _ in 0..n_spaces {
            let j = Uniform::from(1..s.len()).sample(rng);
            s.insert(j, ' ');
        }
        let between = Uniform::from(0..3usize);
        let mut starting = 0;
        let mut ending = 0;
        while starting == 0 && ending == 0 {
            starting = between.sample(rng);
            ending = between.sample(rng);
        }
        s.insert_str(0, &" ".repeat(starting));
        let len = s.len();
        s.insert_str(len, &" ".repeat(ending));
        s
    }
    fn white_words<R: Rng>(delim: char, rng: &mut R) -> String {
        let size = Uniform::from(2..5).sample(rng);
        (0..size).map(|_| white_word(rng)).join(&delim.to_string())
    }

    #[allow(clippy::cognitive_complexity)]
    #[allow(clippy::redundant_closure_call)]
    #[allow(clippy::type_complexity)]
    pub fn make_examples(
        n_examples: usize,
    ) -> Vec<(&'static str, TypeSchema, Vec<(Vec<Space>, Space)>)> {
        let rng = &mut rand::thread_rng();
        let mut tasks = Vec::new();

        macro_rules! t {
            ($name:expr, $tp:expr, $body:block) => {
                let examples = (0..n_examples)
                    .map(|_| {
                        let (i, o) = $body;
                        (vec![i], o)
                    })
                    .collect();
                tasks.push(($name, $tp, examples));
            };
        }
        t!(
            "map strip",
            ptp!(@arrow[tp!(list(tp!(str))), tp!(list(tp!(str)))]),
            {
                let n_words = Uniform::from(1..5).sample(rng);
                let xs: Vec<_> = (0..n_words).map(|_| white_word(rng)).collect();
                let ys = xs.iter().map(|s| Str(s.trim().to_owned())).collect();
                let xs = xs.into_iter().map(Str).collect();
                (List(xs), List(ys))
            }
        );
        t!("strip", ptp!(@arrow[tp!(str), tp!(str)]), {
            let x = white_word(rng);
            let y = x.trim().to_owned();
            (Str(x), Str(y))
        });
        for d in &DELIMITERS {
            let d: char = *d;
            t!(
                "map strip after splitting on d",
                ptp!(@arrow[tp!(str), tp!(list(tp!(str)))]),
                {
                    let x = words(d, rng);
                    let ys = x.split(d).map(|s| Str(s.trim().to_owned())).collect();
                    (Str(x), List(ys))
                }
            );
            t!(
                "map strip and then join with d",
                ptp!(@arrow[tp!(list(tp!(str))), tp!(str)]),
                {
                    let n_words = Uniform::from(1..5).sample(rng);
                    let xs: Vec<_> = (0..n_words).map(|_| word(rng)).collect();
                    let y = xs.iter().map(|s| s.trim().to_owned()).join(&d.to_string());
                    let xs = xs.into_iter().map(Str).collect();
                    (List(xs), Str(y))
                }
            );
            t!("delete delimiter d", ptp!(@arrow[tp!(str), tp!(str)]), {
                let x = words(d, rng);
                let y = x.replace(d, "");
                (Str(x), Str(y))
            });
            t!(
                "extract prefix up to d, exclusive",
                ptp!(@arrow[tp!(str), tp!(str)]),
                {
                    let y = word(rng);
                    let x = format!("{}{}{}", y, d, word(rng));
                    (Str(x), Str(y))
                }
            );
            t!(
                "extract prefix up to d, inclusive",
                ptp!(@arrow[tp!(str), tp!(str)]),
                {
                    let mut y = word(rng);
                    y.push(d);
                    let x = format!("{}{}{}", y, d, word(rng));
                    (Str(x), Str(y))
                }
            );
            t!(
                "extract suffix up to d, exclusive",
                ptp!(@arrow[tp!(str), tp!(str)]),
                {
                    let y = word(rng);
                    let x = format!("{}{}{}", word(rng), y, d);
                    (Str(x), Str(y))
                }
            );
            t!(
                "extract suffix up to d, inclusive",
                ptp!(@arrow[tp!(str), tp!(str)]),
                {
                    let y = format!("{}{}", word(rng), d);
                    let x = format!("{}{}{}", word(rng), y, d);
                    (Str(x), Str(y))
                }
            );
            let d1 = d;
            for d2 in &DELIMITERS {
                let d2: char = *d2;
                t!(
                    "extract delimited by d1, d2",
                    ptp!(@arrow[tp!(str), tp!(str)]),
                    {
                        let y = word(rng);
                        let x = format!("{}{}{}{}{}", word(rng), d1, y, d2, word(rng));
                        (Str(x), Str(y))
                    }
                );
                t!(
                    "extract delimited by d1 (incl), d2",
                    ptp!(@arrow[tp!(str), tp!(str)]),
                    {
                        let y = format!("{}{}{}", d1, word(rng), d2);
                        let x = format!("{}{}{}", word(rng), y, word(rng));
                        (Str(x), Str(y))
                    }
                );
                t!(
                    "extract delimited by d1 (incl), d2 (incl)",
                    ptp!(@arrow[tp!(str), tp!(str)]),
                    {
                        let y = format!("{}{}", d1, word(rng));
                        let x = format!("{}{}{}{}", word(rng), y, d2, word(rng));
                        (Str(x), Str(y))
                    }
                );
                if d1 != ' ' {
                    t!(
                        "strip delimited by d1 from inp delimited by d2",
                        ptp!(@arrow[tp!(str), tp!(str)]),
                        {
                            let x = white_words(d1, rng);
                            let y = x
                                .split(d1)
                                .map(|s| s.trim().to_owned())
                                .join(&d2.to_string());
                            (Str(x), Str(y))
                        }
                    );
                    if d2 != ' ' {
                        t!(
                            "strip from inp delimited by d1, d2",
                            ptp!(@arrow[tp!(str), tp!(str)]),
                            {
                                let y = white_word(rng);
                                let x = format!("{}{}{}{}{}", word(rng), d1, y, d2, word(rng));
                                (Str(x), Str(y))
                            }
                        );
                    }
                }
                if d1 != d2 {
                    t!(
                        "replace delimiter d1 with d2",
                        ptp!(@arrow[tp!(str), tp!(str)]),
                        {
                            let x = words(d1, rng);
                            let y = x.replace(d1, &d2.to_string());
                            (Str(x), Str(y))
                        }
                    );
                }
            }
        }

        macro_rules! single_word_edit {
            ($name:expr, $f:expr) => {
                t!(
                    concat!($name, " strip"),
                    ptp!(@arrow[tp!(str), tp!(str)]),
                    {
                        let x = white_word(rng);
                        let y = ($f)(&x);
                        (Str(x), Str(y))
                    }
                );
                t!(
                    concat!("map ", $name),
                    ptp!(@arrow[tp!(list(tp!(str))), tp!(list(tp!(str)))]),
                    {
                        let n_words = Uniform::from(1..5).sample(rng);
                        let xs: Vec<_> = (0..n_words).map(|_| word(rng)).collect();
                        let ys = xs.iter().map(|s| Str(($f)(s))).collect();
                        let xs = xs.into_iter().map(Str).collect();
                        (List(xs), List(ys))
                    }
                );
                for d in &DELIMITERS {
                    let d: char = *d;
                    t!(
                        concat!("map ", $name, " after splitting"),
                        ptp!(@arrow[tp!(str), tp!(list(tp!(str)))]),
                        {
                            let x = words(d, rng);
                            let ys = x.split(d).map(|s| Str(($f)(s))).collect();
                            (Str(x), List(ys))
                        }
                    );
                    t!(
                        concat!("map ", $name, " then join"),
                        ptp!(@arrow[tp!(list(tp!(str))), tp!(str)]),
                        {
                            let n_words = Uniform::from(1..5).sample(rng);
                            let xs: Vec<_> = (0..n_words).map(|_| word(rng)).collect();
                            let y = xs.iter().map(|s| ($f)(s)).join(&d.to_string());
                            let xs = xs.into_iter().map(Str).collect();
                            (List(xs), Str(y))
                        }
                    );
                    if $name != "lowercase" && $name != "uppercase" {
                        t!(
                            concat!($name, " of delimited inp"),
                            ptp!(@arrow[tp!(str), tp!(str)]),
                            {
                                let x = words(d, rng);
                                let y = x.split(d).map(|s| ($f)(s)).join("");
                                (Str(x), Str(y))
                            }
                        );
                    }
                    let d1 = d;
                    for d2 in &DELIMITERS {
                        let d2: char = *d2;
                        t!(
                            concat!($name, " of doubly delimited inp"),
                            ptp!(@arrow[tp!(str), tp!(str)]),
                            {
                                let y = word(rng);
                                let x = format!("{}{}{}{}{}", word(rng), d1, y, d2, word(rng));
                                (Str(x), Str(($f)(&y)))
                            }
                        );
                        if d1 != d2 && $name != "lowercase" && $name != "uppercase" {
                            t!(
                                concat!("delimited ", $name, " of delimited inp"),
                                ptp!(@arrow[tp!(str), tp!(str)]),
                                {
                                    let x = words(d, rng);
                                    let y = x.split(d).map(|s| ($f)(s)).join(&d2.to_string());
                                    (Str(x), Str(y))
                                }
                            );
                        }
                    }
                }
                let importance = if $name != "lowercase" && $name != "uppercase" {
                    2
                } else {
                    1
                };
                for _ in 0..importance {
                    t!($name, ptp!(@arrow[tp!(str), tp!(str)]), {
                        let x = word(rng);
                        let y = ($f)(&x);
                        (Str(x), Str(y))
                    });
                }
            };
        }

        single_word_edit!("lowercase", |s: &str| -> String { s.to_lowercase() });
        single_word_edit!("uppercase", |s: &str| -> String { s.to_uppercase() });
        single_word_edit!("capitalize", |s: &str| -> String {
            let mut s = s.to_owned();
            if let Some(c) = s.get_mut(..1) {
                c.make_ascii_uppercase()
            }
            s
        });
        single_word_edit!("double", |s: &str| -> String { format!("{}{}", s, s) });
        single_word_edit!("first character", |s: &str| -> String {
            s.chars().next().unwrap().to_string()
        });
        single_word_edit!("drop first character", |s: &str| -> String {
            s.chars().skip(1).collect()
        });

        macro_rules! word_edit_pair {
            ($name1:expr, $f1:expr, $name2:expr, $f2:expr) => {
                t!(
                    concat!($name1, " . ", $name2),
                    ptp!(@arrow[tp!(str), tp!(str)]),
                    {
                        let x = word(rng);
                        let y = ($f2)(&(($f1)(&x)));
                        (Str(x), Str(y))
                    }
                )
            };
        }

        word_edit_pair!(
            "lowercase",
            |s: &str| -> String { s.to_lowercase() },
            "first character",
            |s: &str| -> String { s.chars().next().unwrap().to_string() }
        );
        word_edit_pair!(
            "lowercase",
            |s: &str| -> String { s.to_lowercase() },
            "drop first character",
            |s: &str| -> String { s.chars().skip(1).collect() }
        );
        word_edit_pair!(
            "uppercase",
            |s: &str| -> String { s.to_uppercase() },
            "first character",
            |s: &str| -> String { s.chars().next().unwrap().to_string() }
        );
        word_edit_pair!(
            "uppercase",
            |s: &str| -> String { s.to_uppercase() },
            "drop first character",
            |s: &str| -> String { s.chars().skip(1).collect() }
        );
        word_edit_pair!(
            "double",
            |s: &str| -> String { format!("{}{}", s, s) },
            "first character",
            |s: &str| -> String { s.chars().next().unwrap().to_string() }
        );
        word_edit_pair!(
            "double",
            |s: &str| -> String { format!("{}{}", s, s) },
            "drop first character",
            |s: &str| -> String { s.chars().skip(1).collect() }
        );
        word_edit_pair!(
            "double",
            |s: &str| -> String { format!("{}{}", s, s) },
            "capitalize",
            |s: &str| -> String {
                let mut s = s.to_owned();
                if let Some(c) = s.get_mut(..1) {
                    c.make_ascii_uppercase()
                }
                s
            }
        );
        word_edit_pair!(
            "first character",
            |s: &str| -> String { s.chars().next().unwrap().to_string() },
            "drop first character",
            |s: &str| -> String { s.chars().skip(1).collect() }
        );
        word_edit_pair!(
            "drop first character",
            |s: &str| -> String { s.chars().skip(1).collect() },
            "drop first character",
            |s: &str| -> String { s.chars().skip(1).collect() }
        );
        word_edit_pair!(
            "drop first character",
            |s: &str| -> String { s.chars().skip(1).collect() },
            "double",
            |s: &str| -> String { format!("{}{}", s, s) }
        );
        word_edit_pair!(
            "capitalize",
            |s: &str| -> String {
                let mut s = s.to_owned();
                if let Some(c) = s.get_mut(..1) {
                    c.make_ascii_uppercase()
                }
                s
            },
            "double",
            |s: &str| -> String { format!("{}{}", s, s) }
        );

        tasks
    }
}
