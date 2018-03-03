//! The classic circuit domain as in the paper "Bootstrap Learning for Modular Concept Discovery"
//! (2013).
//!
//! The [`Representation`] for circuits, [`repr()`], is just the `nand` operation in lambda
//! calculus ([`lambda::Language`]).
//!
//! [`repr()`]: fn.repr.html
//! [`Representation`]: ../../trait.Representation.html
//! [`lambda::Language`]: ../../lambda/struct.Language.html

use std::f64;
use polytype::Type;
use rand::{self, Rng};
use rand::distributions::{IndependentSample, Weighted, WeightedChoice};
use super::super::lambda::{Expression, Language};
use super::super::Task;

/// Treat this as any other [`lambda::Language`].
///
/// It only defines the binary `nand` operation:
///
/// ```ignore
/// "nand": arrow![tp!(bool), tp!(bool), tp!(bool)])
/// ```
///
/// [`lambda::Language`]: ../../lambda/struct.Language.html
pub fn repr() -> Language {
    Language::uniform(
        vec![
            (
                String::from("nand"),
                arrow![tp!(bool), tp!(bool), tp!(bool)],
            ),
        ],
        vec![],
    )
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

fn extended_evaluator(primitive: &str, inp: &[bool]) -> bool {
    match primitive {
        "nand" => !(inp[0] & inp[1]),
        "not" => !inp[0],
        "and" => inp[0] & inp[1],
        "or" => inp[0] | inp[1],
        "mux2" => [inp[0], inp[1]][inp[2] as usize],
        "mux4" => [inp[0], inp[1], inp[2], inp[3]][((inp[5] as usize) << 1) + inp[4] as usize],
        _ => unreachable!(),
    }
}

pub fn make_tasks(count: u32) -> Vec<Task<'static, Language, Vec<(Vec<bool>, bool)>>> {
    let dsl = Language {
        primitives: vec![
            (String::from("not"), vec![tp!(bool); 2].into(), 1f64),
            (String::from("and"), vec![tp!(bool); 3].into(), 2f64),
            (String::from("or"), vec![tp!(bool); 3].into(), 2f64),
            (String::from("mux2"), vec![tp!(bool); 4].into(), 4f64),
            (String::from("mux4"), vec![tp!(bool); 7].into(), 0f64),
        ],
        invented: vec![],
        variable_logprob: 0f64,
    };
    let mut n_input_weights = vec![
        Weighted { weight: 1, item: 1 },
        Weighted { weight: 2, item: 2 },
        Weighted { weight: 3, item: 3 },
        Weighted { weight: 4, item: 4 },
        Weighted { weight: 4, item: 5 },
        Weighted { weight: 4, item: 6 },
    ];
    let n_input_distribution = WeightedChoice::new(&mut n_input_weights);

    let mut rng = rand::thread_rng();
    (0..count)
        .map(move |_| {
            // TODO: completely rewrite circuit generation
            let n_inputs = n_input_distribution.ind_sample(&mut rng);
            let tp = Type::from(vec![tp!(bool); n_inputs + 1]);
            eprintln!("enum");
            let choice = dsl.enumerate(tp.clone())
                .find(|_| rng.gen_weighted_bool(count / 6));
            let expr = choice.unwrap().0;
            eprintln!("found {}", dsl.stringify(&expr));
            let examples: Vec<_> = truth_table(n_inputs)
                .into_iter()
                .map(|ins| {
                    let out = dsl.eval(&expr, &extended_evaluator, &ins).unwrap();
                    (ins, out)
                })
                .collect();
            let oracle_examples = examples.clone();
            eprintln!("added task {}", dsl.stringify(&expr));
            let oracle = Box::new(move |dsl: &Language, expr: &Expression| -> f64 {
                let success = oracle_examples.iter().all(|&(ref inps, ref out)| {
                    if let Some(ref o) = dsl.eval(expr, &extended_evaluator, inps) {
                        o == out
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

pub fn truth_table(dim: usize) -> Vec<Vec<bool>> {
    match dim {
        0 => vec![],
        1 => vec![vec![true], vec![false]],
        _ => {
            let mut x1 = truth_table(dim - 1);
            let mut x2 = x1.clone();
            for mut row in x1.iter_mut() {
                row.push(true)
            }
            for mut row in x2.iter_mut() {
                row.push(false)
            }
            x1.append(&mut x2);
            x1
        }
    }
}
