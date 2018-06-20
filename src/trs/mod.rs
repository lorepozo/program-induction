//! (representation) Polymorphically-typed term rewriting system.

mod trace;
mod trs;
pub use self::trs::{SampleError, TypeError, TRS, TRSParams, TRSSpace};
use Task;

use polytype::TypeSchema;
use term_rewriting::Rule;

pub fn make_task_from_data(
    data: &[Rule],
    tp: TypeSchema,
    p_partial: f64,
    temperature: f64,
    prior_temperature: f64,
    ll_temperature: f64,
) -> Task<TRS, (), ()> {
    Task {
        oracle: Box::new(move |h: &TRS, _x| {
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
