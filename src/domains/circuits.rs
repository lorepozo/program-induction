//! The Boolean circuit domain.
//!
//! As in the paper "Bootstrap Learning for Modular Concept Discovery" (2013).
//!
//! # Examples
//!
//! ```
//! use programinduction::domains::circuits;
//! use programinduction::{ECParams, EC};
//! use rand::{rngs::SmallRng, SeedableRng};
//!
//! let dsl = circuits::dsl();
//! let rng = &mut SmallRng::from_seed([1u8; 32]);
//! let tasks = circuits::make_tasks(rng, 250);
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

use itertools::Itertools;
use polytype::{ptp, tp, Type, TypeScheme};
use rand::{
    distributions::{Distribution, WeightedIndex},
    Rng,
};
use std::iter;
use std::sync::Arc;

use crate::lambda::{Evaluator as EvaluatorT, Expression, Language};
use crate::Task;

/// The circuit representation, a [`lambda::Language`], only defines the binary `nand` operation.
///
/// ```compile_fails
/// "nand": ptp!(@arrow[tp!(bool), tp!(bool), tp!(bool)])
/// ```
///
/// [`lambda::Language`]: ../../lambda/struct.Language.html
pub fn dsl() -> Language {
    Language::uniform(vec![(
        "nand",
        ptp!(@arrow[tp!(bool), tp!(bool), tp!(bool)]),
    )])
}

/// All values in the circuits domain can be represented in this `Space`.
pub type Space = bool;

/// An [`Evaluator`] for the circuits domain.
///
/// # Examples
///
/// ```
/// use polytype::{ptp, tp};
/// use programinduction::domains::circuits;
/// use programinduction::{lambda, ECParams, EC};
///
/// let dsl = circuits::dsl();
///
/// let examples = vec![ // NOT
///     (vec![false], true),
///     (vec![true], false),
/// ];
/// let task = lambda::task_by_evaluation(
///     circuits::Evaluator,
///     ptp!(@arrow[tp!(bool), tp!(bool)]),
///     &examples,
/// );
/// let ec_params = ECParams {
///     frontier_limit: 1,
///     search_limit_timeout: None,
///     search_limit_description_length: Some(5.0),
/// };
///
/// let frontiers = dsl.explore(&ec_params, &[task]);
/// let (expr, _logprior, _loglikelihood) = frontiers[0].best_solution().unwrap();
/// assert_eq!(dsl.display(expr), "(λ (nand $0 $0))");
/// ```
///
/// [`Evaluator`]: ../../lambda/trait.Evaluator.html
#[derive(Copy, Clone)]
pub struct Evaluator;
impl EvaluatorT for Evaluator {
    type Space = Space;
    type Error = ();
    fn evaluate(&self, primitive: &str, inp: &[Self::Space]) -> Result<Self::Space, Self::Error> {
        match primitive {
            "nand" => Ok(!(inp[0] & inp[1])),
            _ => unreachable!(),
        }
    }
}

/// Randomly sample a number of circuits into [`Task`]s.
///
/// For a circuit, the number of inputs is sampled from 1 to 6 with weights 1, 2, 3, 4, 4, and 4
/// respectively. The number of gates is sampled from 1 to 3 with weights 1, 2, and 2 respectively.
/// The gates themselves are sampled from NOT, AND, OR, and MUX2 with weights 1, 2, 2, and 4,
/// respectively. All circuits are connected: every input is used and every gate's output is either
/// wired to another gate or to the circuits final output.
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
pub fn make_tasks<R: Rng>(
    rng: &mut R,
    count: u32,
) -> Vec<impl Task<[bool], Representation = Language, Expression = Expression>> {
    make_tasks_advanced(
        rng,
        count,
        [1, 2, 3, 4, 4, 4, 0, 0],
        [1, 2, 2, 0, 0, 0, 0, 0],
        1,
        2,
        2,
        4,
        0,
    )
}

/// Like [`make_tasks`], but with a configurable circuit distribution.
///
/// This works by randomly constructing Boolean circuits, consisting of some number of inputs and
/// some number of gates transforming those inputs into final output. The resulting truth table
/// serves the task's oracle, providing a log-likelihood of `0.0` for success
/// and `f64::NEG_INFINITY` for failure.
///
/// The `n_input_weights` and `n_gate_weights` arguments specify the relative distributions for the
/// number of inputs/gates respectively from 1 to 8. The `gate_` arguments are relative
/// weights for sampling the respective logic gate.
/// Sample circuits which are invalid (i.e. not connected) are rejected.
///
/// [`make_tasks`]: fn.make_tasks.html
#[allow(clippy::too_many_arguments)]
pub fn make_tasks_advanced<R: Rng>(
    rng: &mut R,
    count: u32,
    n_input_weights: [u32; 8],
    n_gate_weights: [u32; 8],
    gate_not: u32,
    gate_and: u32,
    gate_or: u32,
    gate_mux2: u32,
    gate_mux4: u32,
) -> Vec<impl Task<[bool], Representation = Language, Expression = Expression>> {
    let n_input_distribution =
        WeightedIndex::new(n_input_weights).expect("invalid weights for number of circuit inputs");
    let n_gate_distribution =
        WeightedIndex::new(n_gate_weights).expect("invalid weights for number of circuit gates");
    let gate_weights = WeightedIndex::new([gate_not, gate_and, gate_or, gate_mux2, gate_mux4])
        .expect("invalid weights for circuit gates");

    (0..count)
        .map(move |_| {
            let mut n_inputs = 1 + n_input_distribution.sample(rng);
            let mut n_gates = 1 + n_gate_distribution.sample(rng);
            while n_inputs / n_gates >= 3 {
                n_inputs = 1 + n_input_distribution.sample(rng);
                n_gates = 1 + n_gate_distribution.sample(rng);
            }
            let circuit = gates::Circuit::new(rng, &gate_weights, n_inputs as u32, n_gates);
            let outputs: Vec<_> = iter::repeat(vec![false, true])
                .take(n_inputs)
                .multi_cartesian_product()
                .map(|ins| circuit.eval(&ins))
                .collect();
            CircuitTask::new(n_inputs, outputs)
        })
        .collect()
}

struct CircuitTask {
    n_inputs: usize,
    expected_outputs: Vec<bool>,
    tp: TypeScheme,
}
impl CircuitTask {
    fn new(n_inputs: usize, expected_outputs: Vec<bool>) -> Self {
        let tp = TypeScheme::Monotype(Type::from(vec![tp!(bool); n_inputs + 1]));
        CircuitTask {
            n_inputs,
            expected_outputs,
            tp,
        }
    }
}
impl Task<[bool]> for CircuitTask {
    type Representation = Language;
    type Expression = Expression;

    fn oracle(&self, dsl: &Self::Representation, expr: &Self::Expression) -> f64 {
        let evaluator = Arc::new(Evaluator);
        let success = iter::repeat(vec![false, true])
            .take(self.n_inputs)
            .multi_cartesian_product()
            .zip(&self.expected_outputs)
            .all(|(inps, out)| {
                if let Ok(o) = dsl.eval_arc(expr, &evaluator, &inps) {
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
    }
    fn tp(&self) -> &TypeScheme {
        &self.tp
    }
    fn observation(&self) -> &[bool] {
        &self.expected_outputs
    }
}

mod gates {
    use rand::{
        distributions::{Distribution, WeightedIndex},
        seq::index::sample,
        Rng,
    };

    const GATE_CHOICES: [Gate; 5] = [Gate::Not, Gate::And, Gate::Or, Gate::Mux2, Gate::Mux4];

    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub enum Gate {
        Not,
        And,
        Or,
        Mux2,
        Mux4,
    }
    impl Gate {
        fn n_inputs(self) -> u32 {
            match self {
                Gate::Not => 1,
                Gate::And | Gate::Or => 2,
                Gate::Mux2 => 3,
                Gate::Mux4 => 6,
            }
        }
        fn eval(self, inp: &[bool]) -> bool {
            match self {
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
            gate_distribution: &WeightedIndex<u32>,
            n_inputs: u32,
            n_gates: usize,
        ) -> Self {
            loop {
                let mut operations = Vec::with_capacity(n_gates);
                while operations.len() < n_gates {
                    let gate = GATE_CHOICES[gate_distribution.sample(rng)];
                    let n_lanes = n_inputs + (operations.len() as u32);
                    if gate.n_inputs() > n_lanes {
                        continue;
                    }
                    let args = sample(rng, n_lanes as usize, gate.n_inputs() as usize)
                        .into_iter()
                        .map(|x| x as u32)
                        .collect();
                    operations.push((gate, args));
                }
                let circuit = Circuit {
                    n_inputs,
                    operations,
                };
                if circuit.is_connected() {
                    break circuit;
                }
            }
        }
        /// A circuit is connected if every output except for the last one is an input for some
        /// other gate.
        fn is_connected(&self) -> bool {
            let n_lanes = self.n_inputs as usize + self.operations.len();
            let mut is_used = vec![false; n_lanes];
            for (_, args) in &self.operations {
                for i in args {
                    is_used[*i as usize] = true;
                }
            }
            is_used.pop();
            is_used.into_iter().all(|x| x)
        }
        pub fn eval(&self, inp: &[bool]) -> bool {
            let mut lanes = inp.to_vec();
            for (gate, args) in &self.operations {
                let gate_inp: Vec<bool> = args.iter().map(|a| lanes[*a as usize]).collect();
                lanes.push(gate.eval(&gate_inp));
            }
            lanes.pop().unwrap()
        }
    }
}
