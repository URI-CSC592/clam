//! Create the Silva-18S dataset for use in CLAM.

use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use abd_clam::{Dataset, VecDataset};
use log::info;
use rand::prelude::*;

/// Read the Silva-18S dataset from the given path.
///
/// The `unaligned_path` should point to the `silva-SSU-Ref-unaligned.txt` file
/// which contains the unaligned sequences, one per line.
///
/// The `metric` function should compute the distance between two sequences.
///
/// The dataset is (randomly) split into a query set of `1_000` sequences and a
/// training set of the remaining sequences.
///
/// # Arguments
///
/// * `unaligned_path`: The path to the unaligned sequences file.
/// * `headers_path`: The path to the headers file.
/// * `metric`: The metric to use for computing distances.
/// * `is_expensive`: Whether the metric is expensive to compute.
///
/// # Returns
///
/// 4 datasets: the training set, the query set, the training set headers, and
/// the query set headers.
///
/// # Errors
///
/// * If the file at `unaligned_path` does not exist, cannot be read, is not
/// valid UTF-8, or is otherwise malformed.
#[allow(clippy::ptr_arg)]
pub fn silva_to_dataset(
    unaligned_path: &Path,
    headers_path: &Path,
    metric: fn(&String, &String) -> u32,
    is_expensive: bool,
) -> Result<[VecDataset<String, u32>; 4], String> {
    // Get the stem of the file name.
    let stem = unaligned_path
        .file_stem()
        .ok_or_else(|| format!("Could not get file stem for {unaligned_path:?}"))?;
    let stem = stem
        .to_str()
        .ok_or_else(|| format!("Could not convert file stem to string for {unaligned_path:?}"))?;

    // Open the unaligned sequences file and read the lines.
    let file = File::open(unaligned_path)
        .map_err(|e| format!("Could not open file {unaligned_path:?}: {e}"))?;
    let reader = BufReader::new(file);
    let sequences = reader
        .lines()
        .map(|line| line.map_err(|e| format!("Could not read line: {e}")))
        .collect::<Result<Vec<_>, _>>()?;
    info!(
        "Read {} sequences from {unaligned_path:?}.",
        sequences.len()
    );

    // Read the headers file.
    let file = File::open(headers_path)
        .map_err(|e| format!("Could not open file {headers_path:?}: {e}"))?;
    let reader = BufReader::new(file);
    let headers = reader
        .lines()
        .map(|line| line.map_err(|e| format!("Could not read line: {e}")))
        .collect::<Result<Vec<_>, _>>()?;
    info!("Read {} headers from {headers_path:?}.", headers.len());

    // join the lines and headers into a single vector of (line, header) pairs.
    let mut sequences = sequences.into_iter().zip(headers).collect::<Vec<_>>();
    sequences.shuffle(&mut thread_rng());
    info!("Shuffled sequences and headers.");

    // Split the lines into the training and query sets.
    let queries = sequences.split_off(1000);
    let (queries, query_headers): (Vec<_>, Vec<_>) = queries.into_iter().unzip();
    let queries = VecDataset::new(format!("{stem}-queries"), queries, metric, is_expensive);
    let query_headers = VecDataset::new(
        format!("{stem}-query-headers"),
        query_headers,
        metric,
        is_expensive,
    );
    info!(
        "Using {} sequences for queries.",
        query_headers.cardinality()
    );

    let (train, train_headers): (Vec<_>, Vec<_>) = sequences.into_iter().unzip();
    let train = VecDataset::new(format!("{stem}-train"), train, metric, is_expensive);
    let train_headers = VecDataset::new(
        format!("{stem}-train-headers"),
        train_headers,
        metric,
        is_expensive,
    );
    info!(
        "Using {} sequences for training.",
        train_headers.cardinality()
    );

    assert_eq!(train.cardinality(), train_headers.cardinality());
    assert_eq!(queries.cardinality(), query_headers.cardinality());

    Ok([train, queries, train_headers, query_headers])
}
