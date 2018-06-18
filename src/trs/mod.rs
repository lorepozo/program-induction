//! (representation) Polymorphically-typed term rewriting system.
mod hmtrs;
mod trace;
mod utils;

pub use self::hmtrs::{SampleError, TypeError, HMTRS};
use polytype::TypeSchema;
use term_rewriting::Rule;
use Task;

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
            h.posterior(
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
