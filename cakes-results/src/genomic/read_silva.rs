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

    // Open the file and read the lines.
    let file = File::open(unaligned_path)
        .map_err(|e| format!("Could not open file {unaligned_path:?}: {e}"))?;
    let reader = BufReader::new(file);
    let lines = reader
        .lines()
        .map(|line| line.map_err(|e| format!("Could not read line: {e}")))
        .collect::<Result<Vec<_>, _>>()?;

    // shuffle the lines and keep track of the original indices.
    let mut lines = lines.into_iter().enumerate().collect::<Vec<_>>();
    lines.shuffle(&mut rand::thread_rng());

    // Collect the first 1000 lines for the query set. The remaining lines are
    // for the training set.
    let (train_indices, train_sequences): (Vec<_>, Vec<_>) =
        lines.split_off(1000).into_iter().unzip();
    let train = VecDataset::new(
        format!("{stem}-train"),
        train_sequences,
        metric,
        is_expensive,
    );

    // Collect the lines for the query set.
    let (query_indices, queries): (Vec<_>, Vec<_>) = lines.into_iter().unzip();
    let queries = VecDataset::new(format!("{stem}-queries"), queries, metric, is_expensive);

    // Read the headers file.
    let file = File::open(headers_path)
        .map_err(|e| format!("Could not open file {headers_path:?}: {e}"))?;
    let reader = BufReader::new(file);
    let headers = reader
        .lines()
        .map(|line| line.map_err(|e| format!("Could not read line: {e}")))
        .collect::<Result<Vec<_>, _>>()?;
    info!("Read {} headers from {headers_path:?}.", headers.len());

    // Split the headers into the training and query sets.
    let (query_headers, train_headers) = headers
        .into_iter()
        .enumerate()
        .partition::<Vec<_>, _>(|(i, _)| query_indices.contains(i));

    let train_headers = train_headers
        .into_iter()
        .filter(|(i, _)| train_indices.contains(i))
        .map(|(_, h)| h)
        .collect::<Vec<_>>();
    let train_headers = VecDataset::new(
        format!("{stem}-train-headers"),
        train_headers,
        metric,
        is_expensive,
    );

    let query_headers = query_headers
        .into_iter()
        .map(|(_, h)| h)
        .collect::<Vec<_>>();
    let query_headers = VecDataset::new(
        format!("{stem}-query-headers"),
        query_headers,
        metric,
        is_expensive,
    );

    assert_eq!(train.cardinality(), train_headers.cardinality());
    assert_eq!(queries.cardinality(), query_headers.cardinality());

    Ok([train, queries, train_headers, query_headers])
}
