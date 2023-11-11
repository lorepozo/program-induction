use polytype::{ptp, tp};
use programinduction::pcfg::{self, Grammar, Rule};
use programinduction::{GPParams, GPSelection, Task, GP};
use rand::{rngs::SmallRng, SeedableRng};

#[test]
fn gp_sum_arith() {
    fn evaluator(name: &str, inps: &[i32]) -> Result<i32, ()> {
        match name {
            "0" => Ok(0),
            "1" => Ok(1),
            "plus" => Ok(inps[0] + inps[1]),
            _ => unreachable!(),
        }
    }
    let g = Grammar::new(
        tp!(EXPR),
        vec![
            Rule::new("0", tp!(EXPR), 1.0),
            Rule::new("1", tp!(EXPR), 1.0),
            Rule::new("plus", tp!(@arrow[tp!(EXPR), tp!(EXPR), tp!(EXPR)]), 1.0),
        ],
    );
    let target = 6;
    let task = Task {
        oracle: Box::new(|g: &Grammar, expr| {
            if let Ok(n) = g.eval(expr, &evaluator) {
                (n - target).abs() as f64 // numbers close to target
            } else {
                std::f64::INFINITY
            }
        }),
        tp: ptp!(EXPR),
        observation: (),
    };

    let gpparams = GPParams {
        selection: GPSelection::Deterministic,
        population_size: 10,
        tournament_size: 5,
        mutation_prob: 0.6,
        n_delta: 1,
    };
    let params = pcfg::GeneticParams::default();
    let generations = 1000;
    let rng = &mut SmallRng::from_seed([1u8; 16]);

    let mut pop = g.init(&params, rng, &gpparams, &task);
    for _ in 0..generations {
        g.evolve(&params, rng, &gpparams, &task, &mut pop)
    }

    // perfect winner is found!
    let &(ref winner, score) = &pop[0];
    assert_eq!(6, g.eval(winner, &evaluator).unwrap());
    assert_eq!(0.0, score);
}
