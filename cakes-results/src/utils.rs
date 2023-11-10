//! Utility functions for the crate.

use core::cmp::Ordering;

use distances::Number;
use log::{debug, warn};
use num_format::ToFormattedString;

/// Format a `f32` as a string with 6 digits of precision and separators.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
pub fn format_f32(x: f32) -> String {
    let trunc = x.trunc() as u32;
    let fract = (x.fract() * 10f32.powi(3)).round() as u32;

    let trunc = trunc.to_formatted_string(&num_format::Locale::en);

    #[allow(clippy::unwrap_used)]
    let fract = fract.to_formatted_string(
        &num_format::CustomFormat::builder()
            .grouping(num_format::Grouping::Standard)
            .separator("_")
            .build()
            .unwrap(),
    );

    format!("{trunc}.{fract}")
}

/// Compute the recall of a knn-search algorithm.
///
/// # Arguments
///
/// * `hits`: the hits of the algorithm.
/// * `linear_hits`: the hits of linear search.
///
/// # Returns
///
/// * The recall of the algorithm.
#[must_use]
pub fn compute_recall<U: Number>(
    mut hits: Vec<(usize, U)>,
    mut linear_hits: Vec<(usize, U)>,
) -> f32 {
    if linear_hits.is_empty() {
        warn!("Linear search was too slow. Skipping recall computation.");
        1.0
    } else {
        let (num_hits, num_linear_hits) = (hits.len(), linear_hits.len());
        debug!("Num Hits: {num_hits}, Num Linear Hits: {num_linear_hits}");

        hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater));
        let mut hits = hits.into_iter().map(|(_, d)| d).peekable();

        linear_hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater));
        let mut linear_hits = linear_hits.into_iter().map(|(_, d)| d).peekable();

        let mut num_common = 0_usize;
        while let (Some(&hit), Some(&linear_hit)) = (hits.peek(), linear_hits.peek()) {
            if (hit - linear_hit).abs() < U::epsilon() {
                num_common += 1;
                hits.next();
                linear_hits.next();
            } else if hit < linear_hit {
                hits.next();
            } else {
                linear_hits.next();
            }
        }
        let recall = num_common.as_f32() / num_linear_hits.as_f32();
        debug!("Recall: {}, num_common: {}", format_f32(recall), num_common);

        recall
    }
}
