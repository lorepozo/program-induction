#[macro_use]
extern crate polytype;
extern crate programinduction;

use programinduction::{ECParams, EC};
use programinduction::lambda;
use programinduction::pcfg::{self, Grammar, Rule};
use programinduction::domains::{circuits, strings};

fn arith_evaluator(name: &str, inps: &[i32]) -> i32 {
    match name {
        "0" => 0,
        "1" => 1,
        "plus" => inps[0] + inps[1],
        _ => unreachable!(),
    }
}

#[test]
fn ec_circuits() {
    let dsl = circuits::dsl();
    let tasks = circuits::make_tasks(100);
    let ec_params = ECParams {
        frontier_limit: 10,
        search_limit: 1000,
    };
    let params = lambda::CompressionParams::default();

    let (dsl, _frontiers) = dsl.ec(&ec_params, &params, &tasks);
    assert!(!dsl.invented.is_empty());
}

#[test]
fn explore_arith_pcfg() {
    let g = Grammar::new(
        tp!(EXPR),
        vec![
            Rule::new("0", tp!(EXPR), 1.0),
            Rule::new("1", tp!(EXPR), 1.0),
            Rule::new("plus", arrow![tp!(EXPR), tp!(EXPR), tp!(EXPR)], 1.0),
        ],
    );
    let ec_params = ECParams {
        frontier_limit: 1,
        search_limit: 50,
    };
    // task: the number 4
    let task = pcfg::task_by_simple_evaluation(&arith_evaluator, &4, tp!(EXPR));

    let frontiers = g.explore(&ec_params, &[task]);
    assert!(frontiers[0].best_solution().is_some());
}

#[test]
fn explore_strings() {
    let dsl = strings::dsl();
    let lisp = lambda::LispEvaluator::new(strings::lisp_prims());
    let task = lisp.make_task(
        arrow![tp!(str), tp!(str)],
        &[("\"OFJQc>BLVP>eMS\"", "\"OFJQc/BLVP/eMS\"")],
    );

    let ec_params = ECParams {
        frontier_limit: 10,
        search_limit: 2500,
    };
    let frontiers = dsl.explore(&ec_params, &[task]);
    let solution = &frontiers[0].best_solution().unwrap().0;
    assert_eq!(
        "(λ (join (char->str /) (split > $0)))",
        dsl.display(solution)
    );
}
