use std::time::Duration;

use polytype::{ptp, tp};
use programinduction::domains::{circuits, strings};
use programinduction::lambda;
use programinduction::pcfg::{self, Grammar, Rule};
use programinduction::{ECParams, EC};

fn arith_evaluate(name: &str, inps: &[i32]) -> Result<i32, ()> {
    match name {
        "0" => Ok(0),
        "1" => Ok(1),
        "plus" => Ok(inps[0] + inps[1]),
        _ => unreachable!(),
    }
}

#[test]
#[ignore]
fn ec_circuits_dl() {
    let dsl = circuits::dsl();
    let tasks = circuits::make_tasks(100);
    let ec_params = ECParams {
        frontier_limit: 10,
        search_limit_timeout: None,
        search_limit_description_length: Some(9.0),
    };
    let params = lambda::CompressionParams::default();

    let (dsl, _frontiers) = dsl.ec(&ec_params, &params, &tasks);
    assert!(!dsl.invented.is_empty());
}

#[test]
fn explore_circuits_timeout() {
    let dsl = circuits::dsl();
    let tasks = circuits::make_tasks(100);
    let ec_params = ECParams {
        frontier_limit: 10,
        search_limit_timeout: Some(Duration::new(1, 0)),
        search_limit_description_length: None,
    };

    let frontiers = dsl.explore(&ec_params, &tasks);
    assert!(frontiers.iter().any(|f| !f.is_empty()));
}

#[test]
fn explore_arith_pcfg() {
    let g = Grammar::new(
        tp!(EXPR),
        vec![
            Rule::new("0", tp!(EXPR), 1.0),
            Rule::new("1", tp!(EXPR), 1.0),
            Rule::new("plus", tp!(@arrow[tp!(EXPR), tp!(EXPR), tp!(EXPR)]), 1.0),
        ],
    );
    let ec_params = ECParams {
        frontier_limit: 1,
        search_limit_timeout: None,
        search_limit_description_length: Some(8.0),
    };
    // task: the number 4
    let task = pcfg::task_by_evaluation(&arith_evaluate, &4, tp!(EXPR));

    let frontiers = g.explore(&ec_params, &[task]);
    assert!(frontiers[0].best_solution().is_some());
}

#[test]
fn explore_strings() {
    let dsl = strings::dsl();
    let examples = vec![
        // Replace delimiter '>' with '/'
        (
            vec![strings::Space::Str("OFJQc>BLVP>eMS".to_string())],
            strings::Space::Str("OFJQc/BLVP/eMS".to_string()),
        ),
    ];
    let task = lambda::task_by_evaluation(
        strings::Evaluator,
        ptp!(@arrow[tp!(str), tp!(str)]),
        &examples,
    );

    let ec_params = ECParams {
        frontier_limit: 1,
        search_limit_timeout: None,
        search_limit_description_length: Some(13.0),
    };

    let frontiers = dsl.explore(&ec_params, &[task]);
    let solution = &frontiers[0].best_solution().expect("could not solve").0;
    assert_eq!(
        "(λ (join (char->str /) (split > $0)))",
        dsl.display(solution)
    );
}

#[test]
#[ignore]
fn ec_strings() {
    let dsl = strings::dsl();
    let examples = vec![
        // Replace delimiter '>' with '/'
        vec![(
            vec![strings::Space::Str("OFJQc>BLVP>eMS".to_string())],
            strings::Space::Str("OFJQc/BLVP/eMS".to_string()),
        )],
        // Replace delimiter '<' with '/'
        vec![(
            vec![strings::Space::Str("OFJQc<BLVP<eMS".to_string())],
            strings::Space::Str("OFJQc/BLVP/eMS".to_string()),
        )],
        // Replace delimiter ' ' with '/'
        vec![(
            vec![strings::Space::Str("OFJQc BLVP eMS".to_string())],
            strings::Space::Str("OFJQc/BLVP/eMS".to_string()),
        )],
        // Replace delimiter '.' with '/'
        vec![(
            vec![strings::Space::Str("OFJQc.BLVP.eMS".to_string())],
            strings::Space::Str("OFJQc/BLVP/eMS".to_string()),
        )],
    ];
    let tasks = examples
        .iter()
        .map(|ex| {
            lambda::task_by_evaluation(strings::Evaluator, ptp!(@arrow[tp!(str), tp!(str)]), ex)
        })
        .collect::<Vec<_>>();

    let ec_params = ECParams {
        frontier_limit: 10,
        search_limit_timeout: None,
        search_limit_description_length: Some(13.0),
    };
    let params = lambda::CompressionParams::default();

    let (dsl, _) = dsl.ec(&ec_params, &params, &tasks);
    let (dsl, _) = dsl.ec(&ec_params, &params, &tasks);
    assert!(!dsl.invented.is_empty());
}
