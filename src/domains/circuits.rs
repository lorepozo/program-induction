//! The Boolean circuit domain.
//!
//! As in the paper "Bootstrap Learning for Modular Concept Discovery" (2013).
//!
//! # Examples
//!
//! ```
//! use programinduction::{ECParams, EC};
//! use programinduction::domains::circuits;
//!
//! let dsl = circuits::dsl();
//! let tasks = circuits::make_tasks(250);
//! let ec_params = ECParams {
//!     frontier_limit: 100,
//!     search_limit_timeout: None,
//!     search_limit_description_length: Some(9.0),
//! };
//!
//! let frontiers = dsl.explore(&ec_params, &tasks);
//! let hits = frontiers.iter().filter_map(|f| f.best_solution()).count();
//! assert!(40 < hits && hits < 80, "hits = {}", hits);
//! ```

use std::f64;
use std::iter;
use itertools::Itertools;
use polytype::Type;
use rand;
use rand::distributions::{IndependentSample, Weighted, WeightedChoice};

use Task;
use lambda::{Expression, Language};

/// The circuit representation, a [`lambda::Language`], only defines the binary `nand` operation.
///
/// ```ignore
/// "nand": arrow![tp!(bool), tp!(bool), tp!(bool)]
/// ```
///
/// [`lambda::Language`]: ../../lambda/struct.Language.html
pub fn dsl() -> Language {
    Language::uniform(vec![("nand", arrow![tp!(bool), tp!(bool), tp!(bool)])])
}

/// Evaluate an expression in this domain in accordance with the argument of
/// [`lambda::task_by_simple_evaluation`].
///
/// # Examples
///
/// ```
/// # #[macro_use] extern crate polytype;
/// # extern crate programinduction;
/// use programinduction::{lambda, ECParams, EC};
/// use programinduction::domains::circuits;
///
/// # fn main() {
/// let dsl = circuits::dsl();
///
/// let examples = vec![ // NOT
///     (vec![false], true),
///     (vec![true], false),
/// ];
/// let task = lambda::task_by_simple_evaluation(
///     &circuits::simple_evaluator,
///     arrow![tp!(bool), tp!(bool)],
///     &examples,
/// );
/// let ec_params = ECParams {
///     frontier_limit: 1,
///     search_limit_timeout: None,
///     search_limit_description_length: Some(5.0),
/// };
///
/// let frontiers = dsl.explore(&ec_params, &[task]);
/// let &(ref expr, _logprior, _loglikelihood) = frontiers[0].best_solution().unwrap();
/// assert_eq!(dsl.display(expr), "(λ (nand $0 $0))");
/// # }
/// ```
///
/// [`lambda::task_by_simple_evaluation`]: ../../lambda/fn.task_by_simple_evaluation.html
pub fn simple_evaluator(primitive: &str, inp: &[bool]) -> bool {
    match primitive {
        "nand" => !(inp[0] & inp[1]),
        _ => unreachable!(),
    }
}

fn truth_table(dim: usize) -> Box<Iterator<Item = Vec<bool>>> {
    Box::new(
        iter::repeat(vec![false, true])
            .take(dim)
            .multi_cartesian_product(),
    )
}

/// Randomly sample a number of circuits into [`Task`]s.
///
/// The task observations are outputs of the truth table in sequence, for example
///
/// ```text
///    --- INPUTS ---      OUTPUTS
/// false, false, false => false
/// false, false, true  => false
/// false, true,  false => false
/// false, true,  true  => false
/// true,  false, false => false
/// true,  false, true  => false
/// true,  true,  false => false
/// true,  true,  true  => true
/// ```
///
/// [`Task`]: ../../struct.Task.html
pub fn make_tasks(count: u32) -> Vec<Task<'static, Language, Expression, Vec<bool>>> {
    let mut n_input_weights = vec![
        Weighted { weight: 1, item: 1 },
        Weighted { weight: 2, item: 2 },
        Weighted { weight: 3, item: 3 },
        Weighted { weight: 4, item: 4 },
        Weighted { weight: 4, item: 5 },
        Weighted { weight: 4, item: 6 },
    ];
    let n_input_distribution = WeightedChoice::new(&mut n_input_weights);
    let mut n_gate_weights = vec![
        Weighted { weight: 1, item: 1 },
        Weighted { weight: 2, item: 2 },
        Weighted { weight: 2, item: 3 },
    ];
    let n_gate_distribution = WeightedChoice::new(&mut n_gate_weights);

    let mut gate_weights = vec![
        Weighted {
            weight: 1,
            item: Gate::Not,
        },
        Weighted {
            weight: 2,
            item: Gate::And,
        },
        Weighted {
            weight: 2,
            item: Gate::Or,
        },
        Weighted {
            weight: 4,
            item: Gate::Mux2,
        },
        Weighted {
            weight: 0,
            item: Gate::Mux4,
        },
    ];
    let gate_distribution = WeightedChoice::new(&mut gate_weights);

    let mut rng = rand::thread_rng();
    (0..count)
        .map(move |_| {
            let mut n_inputs = n_input_distribution.ind_sample(&mut rng);
            let mut n_gates = n_gate_distribution.ind_sample(&mut rng);
            while n_inputs / n_gates >= 3 {
                n_inputs = n_input_distribution.ind_sample(&mut rng);
                n_gates = n_gate_distribution.ind_sample(&mut rng);
            }
            let tp = Type::from(vec![tp!(bool); n_inputs + 1]);
            let circuit = Circuit::new(&mut rng, &gate_distribution, n_inputs as u32, n_gates);
            let outputs: Vec<_> = truth_table(n_inputs)
                .map(|ins| circuit.eval(&ins))
                .collect();
            let oracle_outputs = outputs.clone();
            let oracle = Box::new(move |dsl: &Language, expr: &Expression| -> f64 {
                let success = truth_table(n_inputs)
                    .zip(&oracle_outputs)
                    .all(|(inps, out)| {
                        if let Some(o) = dsl.eval(expr, &simple_evaluator, &inps) {
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
                observation: outputs,
                tp,
            }
        })
        .collect()
}

use self::gates::{Circuit, Gate};
mod gates {
    use rand::Rng;
    use rand::distributions::{IndependentSample, WeightedChoice};

    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub enum Gate {
        Not,
        And,
        Or,
        Mux2,
        Mux4,
    }
    impl Gate {
        fn n_inputs(&self) -> u32 {
            match *self {
                Gate::Not => 1,
                Gate::And | Gate::Or => 2,
                Gate::Mux2 => 3,
                Gate::Mux4 => 6,
            }
        }
        fn eval(&self, inp: &[bool]) -> bool {
            match *self {
                Gate::Not => !inp[0],
                Gate::And => inp[0] & inp[1],
                Gate::Or => inp[0] | inp[1],
                Gate::Mux2 => [inp[0], inp[1]][inp[2] as usize],
                Gate::Mux4 => {
                    [inp[0], inp[1], inp[2], inp[3]][((inp[5] as usize) << 1) + inp[4] as usize]
                }
            }
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    pub struct Circuit {
        n_inputs: u32,
        operations: Vec<(Gate, Vec<u32>)>,
    }
    impl Circuit {
        pub fn new<T: Rng>(
            rng: &mut T,
            gate_distribution: &WeightedChoice<Gate>,
            n_inputs: u32,
            n_gates: usize,
        ) -> Self {
            let mut circuit = Circuit {
                n_inputs,
                operations: Vec::new(),
            };
            loop {
                while circuit.operations.len() < n_gates {
                    let gate = gate_distribution.ind_sample(rng);
                    match gate {
                        Gate::Mux2 | Gate::Mux4 if n_inputs < gate.n_inputs() => continue,
                        _ => (),
                    }
                    if gate.n_inputs() > n_inputs + (circuit.operations.len() as u32) {
                        continue;
                    }
                    let mut valid_inputs: Vec<u32> =
                        (0..n_inputs + (circuit.operations.len() as u32)).collect();
                    rng.shuffle(valid_inputs.as_mut_slice());
                    let args = valid_inputs[..(gate.n_inputs() as usize)].to_vec();
                    circuit.operations.push((gate, args));
                }
                if circuit.is_connected() {
                    break;
                }
                circuit.operations = Vec::new();
            }
            circuit
        }
        /// A circuit is connected if every output except for the last one is an input for some
        /// other gate.
        fn is_connected(&self) -> bool {
            let mut is_used = vec![false; self.n_inputs as usize + self.operations.len()];
            for &(_, ref args) in &self.operations {
                for i in args {
                    is_used[*i as usize] = true;
                }
            }
            is_used.pop();
            is_used.into_iter().all(|x| x)
        }
        pub fn eval(&self, inp: &[bool]) -> bool {
            let mut outputs = vec![];
            for &(ref gate, ref args) in &self.operations {
                let gate_inp: Vec<bool> = args.iter()
                    .map(|a| *inp.iter().chain(&outputs).nth(*a as usize).unwrap())
                    .collect();
                outputs.push(gate.eval(&gate_inp));
            }
            outputs.pop().unwrap()
        }
    }
}
