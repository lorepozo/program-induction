//! A library for program induction and learning representations.
//!
//! # Bayesian program learning with the EC algorithm
//!
//! Bayesian program learning (BPL) is a generative model that samples programs, where programs are
//! generative models that sample observations (or examples). BPL is an appealing model for its
//! ability to learn rich concepts from few examples. It was popularized by Lake et al. in the
//! 2015 _Science_ paper [Human-level concept learning through probabilistic program induction].
//!
//! The Exploration-Compression (EC) algorithm is an inference scheme for BPL that learns new
//! representations by _exploring_ solutions to tasks and then _compressing_ those solutions by
//! recognizing common pieces and re-introducing them as primitives. This is an iterative algorithm
//! that can be applied repeatedly and has been found to converge in practice. It was introduced by
//! Dechter et al. in the 2013 _IJCAI_ paper [Bootstrap learning via modular concept discovery].
//! The EC algorithm can be viewed as an expectation-maximization (EM) algorithm for approximate
//! maximum a posteriori (MAP) estimation of programs and their representation. Roughly, the
//! exploration step of EC corresponds to the expectation step of EM and the compression step of EC
//! corresponds to the maximization step of EM.
//!
//! A domain-specific language (DSL) \\(\mathcal{D}\\) expresses the programming primitives and
//! encodes domain-specific knowledge about the space of programs. In conjunction with a weight
//! vector \\(\theta\\), the pair \\((\\mathcal{D}, \\theta)\\) determines a probability
//! distribution over programs. From a program \\(p\\), we can sample a task \\(x\\). Framed as a
//! hierarchical Bayesian model, we have that \\(p \\sim (\\mathcal{D}, \\theta)\\) and \\(x \\sim
//! p\\).  Given a set of tasks \\(X\\) together with a likelihood model \\(\\mathbb{P}[x|p]\\) for
//! scoring a task \\(x\\in X\\) given a program \\(p\\), our goal is to jointly infer a DSL
//! \\(\\mathcal{D}\\) and latent programs for each task that are likely (i.e. "solutions").  For
//! joint probability \\(J\\) of \\((\\mathcal{D},\\theta)\\) and \\(X\\), we want the
//! \\(\\mathcal{D}\^\*\\) and \\(\\theta\^\*\\) solving:
//! \\[
//! \\begin{aligned}
//! 	J(\\mathcal{D},\\theta) &\\triangleq
//!         \\mathbb{P}[\\mathcal{D},\\theta]
//!         \\prod\_{x\\in X} \\sum\_p \\mathbb{P}[x|p]\\mathbb{P}[p|\\mathcal{D},\\theta] \\\\
//!     \\mathcal{D}\^\* &= \\underset{\\mathcal{D}}{\\text{arg}\\,\\text{max}}
//!         \\int J(\\mathcal{D},\\theta)\\;\\mathrm{d}\\theta \\\\
//!     \\theta\^\* &= \\underset{\\theta}{\\text{arg}\\,\\text{max}}
//!         J(\\mathcal{D}\^\*,\\theta)
//! \\end{aligned}
//! \\]
//! This is intractible because calculating \\(J\\) involves taking a sum over every possible
//! program. We therefore define the _frontier_ of a task \\(x\\), written \\(\\mathcal{F}\_x\\),
//! to be a finite set of programs where \\(\mathbb{P}[x|p] > 0\\) for all \\(p \\in
//! \\mathcal{F}\_x\\) and establish an intuitive lower bound:
//! \\[
//!  J\\geq \\mathscr{L}\\triangleq \\mathbb{P}[\mathcal{D},\\theta]
//!     \\prod\_{x\\in X} \\sum\_{p\\in \\mathcal{F}\_x}
//!         \\mathbb{P}[x|p]\\mathbb{P}[p|\\mathcal{D},\\theta]
//! \\]
//! We find programs and induce a DSL by alternating maximization with respect to the frontiers
//! \\(\\mathcal{F}\_x\\) and the DSL \\(\\mathcal{D}\\). Frontiers are assigned by enumerating
//! programs up to some depth according to \\((\\mathcal{D}, \\theta)\\). The DSL \\(\mathcal{D}\\)
//! is modified by "compression", where common program fragments become invented primitives.
//! Estimation of \\(\\theta\\) is a detail that a representation must define (for example, this is
//! done for probabilistic context-free grammars, or PCFGs, using the inside-outside algorithm).
//!
//! In this library, we represent a [`Task`] as an [`observation`] \\(x\\) together with a
//! log-likelihood model (or [`oracle`]) \\(\\log\\mathbb{P}[x|p]\\). We additionally associate
//! tasks with a type, [`tp`], to provide a straightforward constraint for finding programs which
//! may be solutions. Programs may be expressed under different representations, so we provide a
//! representation-agnostic trait [`EC`]. We additionally provide two such representations: a
//! polymorphically-typed lambda calculus in the [`lambda`] module, and a probabilistic context
//! free grammar in the [`pcfg`] module.
//!
//! For example, here is the EC algorithm applied to learning Boolean circuits (a domain which we
//! provide useful tools for, though expanding to other domains is trivial) for five iterations:
//!
//! ```
//! extern crate programinduction;
//! use programinduction::{lambda, ECParams, EC};
//! use programinduction::domains::circuits;
//!
//! fn main() {
//!     let mut dsl = circuits::dsl();
//!     let tasks = circuits::make_tasks(250);
//!     let ec_params = ECParams {
//!         frontier_limit: 10,
//!         search_limit_timeout: None,
//!         search_limit_description_length: Some(9.0),
//!     };
//!     let params = lambda::CompressionParams::default();
//!
//!     let mut frontiers = Vec::new();
//!     for _ in 0..5 {
//!         let (new_dsl, new_frontiers) = dsl.ec(&ec_params, &params, &tasks);
//!         dsl = new_dsl;
//!         frontiers = new_frontiers;
//!     }
//!     let n_invented = dsl.invented.len();
//!     let n_hit = frontiers.iter().filter(|f| !f.is_empty()).count();
//!     println!("hit {} of {} using {} invented primitives", n_hit, tasks.len(), n_invented);
//! }
//! ```
//!
//! # Genetic Programming
//!
//! Genetic programming (GP) with this library is in progress. Our primary resource is the 2008
//! book by Poli et al.: [_A Field Guide to Genetic Programming_].
//!
//! [Human-level concept learning through probabilistic program induction]: http://web.mit.edu/cocosci/Papers/Science-2015-Lake-1332-8.pdf
//! [Bootstrap learning via modular concept discovery]: https://hips.seas.harvard.edu/files/dechter-bootstrap-ijcai-2013.pdf
//! [_A Field Guide to Genetic Programming_]: http://www.gp-field-guide.org.uk
//! [`Task`]: struct.Task.html
//! [`observation`]: struct.Task.html#structfield.observation
//! [`oracle`]: struct.Task.html#structfield.oracle
//! [`tp`]: struct.Task.html#structfield.tp
//! [`EC`]: trait.EC.html
//! [`lambda`]: lambda/index.html
//! [`pcfg`]: pcfg/index.html

extern crate crossbeam_channel;
extern crate fral;
extern crate itertools;
#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate maplit;
#[macro_use]
extern crate nom;
extern crate num_cpus;
#[macro_use]
extern crate polytype;
extern crate rand;
extern crate rayon;

pub mod domains;
mod ec;
mod gp;
pub mod lambda;
pub mod pcfg;
pub use ec::*;
pub use gp::*;

use std::f64;
use polytype::TypeSchema;

/// A task which is solved by an expression under some representation.
///
/// A task can be made from an evaluator and examples with [`lambda::task_by_evaluation`] or
/// [`pcfg::task_by_evaluation`].
///
/// [`lambda::task_by_evaluation`]: lambda/fn.task_by_simple_evaluation.html
/// [`pcfg::task_by_evaluation`]: pcfg/fn.task_by_simple_evaluation.html
pub struct Task<'a, R: Send + Sync + Sized, X: Clone + Send + Sync, O: Sync> {
    /// Evaluate an expression by getting its log-likelihood.
    pub oracle: Box<Fn(&R, &X) -> f64 + Send + Sync + 'a>,
    /// Some program induction methods can take advantage of observations. This may often
    /// practically be the unit type `()`.
    pub observation: O,
    /// An expression that is considered valid for the `oracle` is one of this type.
    pub tp: TypeSchema,
}
