# programinduction

[![Build Status](https://travis-ci.org/lucasem/programinduction.svg?branch=master)](https://travis-ci.org/lucasem/programinduction)
[![crates.io](https://img.shields.io/crates/v/programinduction.svg)](https://crates.io/crates/programinduction)
[![docs.rs](https://docs.rs/programinduction/badge.svg)](https://docs.rs/programinduction)

A library for program induction and learning representations.

Implements Bayesian program learning and genetic programming.

## Installation

Install [rust](https://rust-lang.org). In a new or existing project, add the
following to your `Cargo.toml`:

```toml
[dependencies]
programinduction = "0.1"
# many examples also depend on polytype for the tp! and arrow! macros:
polytype = "1.2"
```

## Usage

Specify a Probabilistic Context-Free Grammar (PCFG; see `pcfg::Grammar`) and
induce a sentence that matches an example:

```rust
#[macro_use]
extern crate polytype;
extern crate programinduction;

use programinduction::{ECParams, EC};
use programinduction::pcfg::{task_by_simple_evaluation, Grammar, Rule};

fn simple_evaluator(name: &str, inps: &[i32]) -> i32 {
    match name {
        "0" => 0,
        "1" => 1,
        "plus" => inps[0] + inps[1],
        _ => unreachable!(),
    }
}

fn main() {
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
    let task = task_by_simple_evaluation(&simple_evaluator, &4, tp!(EXPR));

    let frontiers = g.explore(&ec_params, &[task]);
    let sol = &frontiers[0].best_solution().unwrap().0;
    println!("{}", g.display(sol));
}
```

The Exploration-Compression (EC) algorithm iteratively learns a better
representation by finding common structure in induced programs. We can run
the EC algorithm with a polymorphically-typed lambda calculus representation
`lambda::Language` in a Boolean circuit domain:

```rust
#[macro_use]
extern crate polytype;
extern crate programinduction;

use programinduction::{domains, lambda, ECParams, EC};

fn main() {
    // circuit DSL
    let dsl = lambda::Language::uniform(vec![
        // NAND takes two bools and returns a bool
        ("nand", arrow![tp!(bool), tp!(bool), tp!(bool)]),
    ]);
    // parameters
    let lambda_params = lambda::CompressionParams::default();
    let ec_params = ECParams {
        frontier_limit: 1,
        search_limit: 50,
    };
    // randomly sample 250 circuit tasks
    let tasks = domains::circuits::make_tasks(250);

    // one iteration of EC:
    let (new_dsl, _solutions) = dsl.ec(&ec_params, &lambda_params, &tasks);
    // print the new concepts it invented, based on common structure:
    for &(ref expr, _, _) in &new_dsl.invented {
        println!("invented {}", new_dsl.display(expr))
        // one of the inventions was "(λ (nand $0 $0))",
        // which is the common and useful NOT operation!
    }
}
```

You may have noted the above use of `domains::circuits`. Some domains are
already implemented for you. Currently, this only consists of _circuits_ and
_strings_. The _strings_ domain uses a rich set of primitives and thus
depends on `lambda::LispEvaluator`. If you find this evaluator to be slow,
you may install [racket](https://racket-lang.org) and enable the `racket`
feature in your `Cargo.toml`:

```toml
[dependencies.programinduction]
version = "0.1"
features = ["racket"]
```

See the [documentation](https://docs.rs/programinduction) for more details.

## Tips

For more on the Exploration-Compression algorithm, see _Dechter, Eyal et al.
"Bootstrap Learning via Modular Concept Discovery." IJCAI (2013)._
[link](http://edechter.github.io/publications/DBLP_conf_ijcai_DechterMAT13.pdf)

For more on genetic programming, see _Poli, Riccardo et al. “A Field Guide
to Genetic Programming.” (2008)._ [link](http://www.gp-field-guide.org.uk)
