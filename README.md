# program-induction

[![Build Status](https://travis-ci.org/lucasem/program-induction.svg?branch=master)](https://travis-ci.org/lucasem/program-induction)
[![crates.io](https://img.shields.io/crates/v/programinduction.svg)](https://crates.io/crates/programinduction)
[![docs.rs](https://docs.rs/programinduction/badge.svg)](https://docs.rs/programinduction)

A library for program induction and learning representations.

Implements Bayesian program learning and genetic programming.

## Installation

Install [rust](https://rust-lang.org). In a new or existing project, add the
following to your `Cargo.toml`:

```toml
[dependencies]
programinduction = "0.5"
# many examples also depend on polytype for the tp! and ptp! macros:
polytype = "4.2"
```

The documentation requires a custom HTML header to include KaTeX for math
support. This isn't supported by `cargo doc`, so to build the documentation
you may use:

```sh
cargo rustdoc -- --html-in-header rustdoc-include-katex-header.html
```

## Usage

Specify a probabilistic context-free grammar (PCFG; see `pcfg::Grammar`) and
induce a sentence that matches an example:

```rust
#[macro_use]
extern crate polytype;
extern crate programinduction;

use programinduction::{ECParams, EC};
use programinduction::pcfg::{task_by_evaluation, Grammar, Rule};

fn evaluate(name: &str, inps: &[i32]) -> Result<i32, ()> {
    match name {
        "0" => Ok(0),
        "1" => Ok(1),
        "plus" => Ok(inps[0] + inps[1]),
        _ => unreachable!(),
    }
}

fn main() {
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
    let task = task_by_evaluation(&evaluate, &4, tp!(EXPR));

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
        ("nand", ptp!(@arrow[tp!(bool), tp!(bool), tp!(bool)])),
    ]);
    // parameters
    let lambda_params = lambda::CompressionParams::default();
    let ec_params = ECParams {
        frontier_limit: 1,
        search_limit_timeout: Some(std::time::Duration::new(1, 0)),
        search_limit_description_length: None,
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
_strings_.

## TODO

(you could be the one who does one of these!)

- [x] First-class function evaluation within Rust (and remove lisp
      interpreters).
- [x] Add task generation function in `domains::strings`
- [x] Fallible evaluation (e.g. see how `domains::strings` handles `slice`).
- [x] Lazy evaluation.
- [ ] Consolidate lazy/non-lazy evaluation (for ergonomics).
- [ ] Ability to include recursive primitives in `lambda` representation.
- [ ] Faster lambda calculus evaluation (less cloning; bubble up whether
      beta reduction happened rather than ultimate equality comparison).
- [ ] PCFG compression is currently only estimating parameters, not actually
      learning pieces of programs. An [adaptor
      grammar](http://cocosci.berkeley.edu/tom/papers/adaptornips.pdf)
      approach seems like a good direction to go, perhaps minus the Bayesian
      non-parametrics.
- [ ] `impl GP for pcfg::Grammar` is not yet complete.
- [ ] Add more learning traits (like `EC` or `GP`)
- [ ] Add more representations
- [ ] Add more domains

