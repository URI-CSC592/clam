//! Genomic metrics for Silva-18S sequences.

use super::{nucleotides::Nucleotide, sequence::Sequence};

/// Returns the Levenshtein distance between the two given sequences.
///
/// # Arguments
///
/// * `x` - The first sequence.
/// * `y` - The second sequence.
pub fn levenshtein(x: &Sequence, y: &Sequence) -> u32 {
    if x.is_empty() {
        // handle special case of 0 length
        y.len() as u32
    } else if y.is_empty() {
        // handle special case of 0 length
        x.len() as u32
    } else if x.len() < y.len() {
        // require tat a is no shorter than b
        _levenshtein(y, x)
    } else {
        _levenshtein(x, y)
    }
}

/// Returns the Levenshtein distance between the two given sequences.
fn _levenshtein(x: &Sequence, y: &Sequence) -> u32 {
    let gap_penalty = Nucleotide::gap_penalty();
    // initialize the DP table for y
    let mut cur = (0..=y.len())
        .map(|i| (i as u32) * gap_penalty)
        .collect::<Vec<_>>();

    // calculate edit distance
    for (i, &c_x) in x.neucloetides().iter().enumerate() {
        // get first column for this row
        let mut pre = cur[0];
        cur[0] = (i as u32 + 1) * gap_penalty;

        // calculate rest of row
        for (j, &c_y) in y.neucloetides().iter().enumerate() {
            let tmp = cur[j + 1];

            let del = tmp + gap_penalty;
            let ins = cur[j] + gap_penalty;
            let sub = pre + c_x.penalty(c_y);
            cur[j + 1] = ins.min(del).min(sub);

            pre = tmp;
        }
    }

    cur[y.len()]
}

/// Returns the Hamming distance between the two given sequences.
///
/// # Arguments
///
/// * `x` - The first sequence.
/// * `y` - The second sequence.
#[allow(unused_variables)]
pub fn hamming(x: &Sequence, y: &Sequence) -> u32 {
    todo!()
}

/// Returns the Needleman-Wunsch distance between the two given sequences.
///
/// # Arguments
///
/// * `x` - The first sequence.
/// * `y` - The second sequence.
#[allow(unused_variables)]
pub fn needleman_wunsch(x: &Sequence, y: &Sequence) -> u32 {
    todo!()
}
