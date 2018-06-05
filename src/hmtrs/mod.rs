//! (representation)
//! A [Hindley-Milner][1] [Term Rewriting System][0] (HMTRS).
//!
//! [0]: https://wikipedia.org/wiki/Hindley–Milner_type_system
//!      "Wikipedia - Hindley-Milner Type System"
//! [1]: https://en.wikipedia.org/wiki/Rewriting#Term_rewriting_systems
//!      "Wikipedia - Term Rewriting Systems"
mod hmtrs;
mod trace;
mod utils;

pub use self::hmtrs::HMTRS;
use self::trace::Trace;
use super::Task;
use polytype::TypeSchema;
use std::f64::NEG_INFINITY;
use term_rewriting::Rule;

pub fn make_task_from_data(
    data: &[Rule],
    tp: TypeSchema,
    p_partial: f64,
    temperature: f64,
    prior_temperature: f64,
    ll_temperature: f64,
) -> Task<HMTRS, (), ()> {
    Task {
        oracle: Box::new(move |h: &HMTRS, _x| {
            // TODO: only getting information from temperature-adjusted evaluation
            posterior(
                h,
                data,
                p_partial,
                temperature,
                prior_temperature,
                ll_temperature,
            )
        }),
        // TODO: compute type schema from the data
        tp,
        observation: (),
    }
}

pub fn posterior(
    h: &HMTRS,
    data: &[Rule],
    p_partial: f64,
    temperature: f64,
    prior_temperature: f64,
    ll_temperature: f64,
) -> f64 {
    let prior = pseudo_log_prior(h, temperature, prior_temperature);
    if prior == NEG_INFINITY {
        NEG_INFINITY
    } else {
        prior + log_likelihood(h, data, p_partial, temperature, ll_temperature)
    }
}

pub fn pseudo_log_prior(h: &HMTRS, temp: f64, prior_temp: f64) -> f64 {
    let raw_prior = -(h.size() as f64);
    raw_prior / ((temp + 1.0) * prior_temp)
}

pub fn log_likelihood(h: &HMTRS, data: &[Rule], p_partial: f64, temp: f64, ll_temp: f64) -> f64 {
    data.iter()
        .map(|x| single_log_likelihood(h, x, p_partial, temp) / ll_temp)
        .sum()
}

pub fn single_log_likelihood(h: &HMTRS, datum: &Rule, p_partial: f64, temp: f64) -> f64 {
    let p_observe = 0.0;
    let max_steps = 50;
    let max_size = 500;
    let mut trace = Trace::new(&h.trs, &datum.lhs, p_observe, max_steps, max_size);
    trace.run();

    let ll = if let Some(ref rhs) = datum.rhs() {
        trace.rewrites_to(rhs)
    } else {
        NEG_INFINITY
    };

    if ll == NEG_INFINITY {
        (p_partial + temp).ln()
    } else {
        (1.0 - p_partial + temp).ln() + ll
    }
}
