//! Benchmarks for knn-search when the size of the data set is scaled.

use std::{path::Path, time::Instant};

use abd_clam::{knn, Cakes, PartitionCriteria, VecDataset};
use clap::Parser;
use distances::Number;
use mt_logger::*;
use num_format::ToFormattedString;
use serde::{Deserialize, Serialize};
use symagen::augmentation;

mod ann_datasets;
mod utils;

use crate::{ann_datasets::AnnDatasets, utils::format_f32};

fn main() -> Result<(), String> {
    mt_new!(None, Level::Info, OutputStream::StdOut);

    let args = Args::parse();

    if !args.dataset.starts_with("random") {
        // Check that the data set exists.
        let data_paths = [
            args.input_dir.join(format!("{}-train.npy", args.dataset)),
            args.input_dir.join(format!("{}-test.npy", args.dataset)),
        ];
        for path in &data_paths {
            if !path.exists() {
                return Err(format!("File {path:?} does not exist."));
            }
        }
    }

    // Check that the output directory exists.
    if !args.output_dir.exists() {
        return Err(format!(
            "Output directory {:?} does not exist.",
            args.output_dir
        ));
    }

    make_reports(
        &args.input_dir,
        &args.output_dir,
        &args.dataset,
        args.seed,
        args.max_scale,
        args.error_rate,
        &args.ks,
        args.max_memory,
    )?;

    mt_flush!().map_err(|e| e.to_string())?;

    Ok(())
}

/// Command line arguments for the replicating the ANN-Benchmarks results for Cakes.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the directory with the data sets. The directory should contain
    /// the hdf5 files downloaded from the ann-benchmarks repository.
    #[arg(long)]
    input_dir: std::path::PathBuf,
    /// Output directory for the report.
    #[arg(long)]
    output_dir: std::path::PathBuf,
    /// Name of the data set to process. `data_dir` should contain two files
    /// named `{name}-train.npy` and `{name}-test.npy`. The train file
    /// contains the data to be indexed for search, and the test file contains
    /// the queries to be searched for.
    #[arg(long)]
    dataset: String,
    /// Seed for the random number generator.
    #[arg(long)]
    seed: Option<u64>,
    /// Maximum scaling factor. The data set will be scaled by factors of
    /// `2 ^ i` for `i` in `0..=max_scale`.
    #[arg(long, default_value = "16")]
    max_scale: u32,
    /// Error rate used for scaling.
    #[arg(long, default_value = "0.01")]
    error_rate: f32,
    /// Number of nearest neighbors to search for.
    #[arg(long, value_parser, num_args = 1.., value_delimiter = ' ', default_value = "10 100")]
    ks: Vec<usize>,
    /// Maximum memory usage (in gigabytes) for the scaled data sets.
    #[arg(long, default_value = "256")]
    max_memory: usize,
}

/// Report the results of a scaling benchmark.
///
/// # Arguments
///
/// * `input_dir`: path to the directory with the data sets.
/// * `output_dir`: path to the directory where the report should be saved.
/// * `dataset`: name of the data set to process.
/// * `seed`: seed for the random number generator.
/// * `max_scale`: maximum scaling factor.
/// * `error_rate`: error rate used for scaling.
/// * `ks`: number of nearest neighbors to search for.
/// * `max_memory`: maximum memory usage (in gigabytes) for the scaled data sets.
///
/// # Panics
///
/// * If the knn-algorithms panic.
///
/// # Errors
///
/// * If the data set does not exist.
/// * If the metric of the data set is not supported.
/// * If the output directory does not exist.
/// * If the output directory is not writable.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cognitive_complexity
)]
pub fn make_reports(
    input_dir: &Path,
    output_dir: &Path,
    dataset: &str,
    seed: Option<u64>,
    max_scale: u32,
    error_rate: f32,
    ks: &[usize],
    max_memory: usize,
) -> Result<(), String> {
    let dataset = AnnDatasets::from_str(dataset)?;
    let metric = dataset.metric()?;
    let [train_data, queries] = dataset.read(input_dir)?;

    mt_log!(Level::Info, "Dataset: {}", dataset.name());

    let base_cardinality = train_data.len();
    mt_log!(
        Level::Info,
        "Base cardinality: {}",
        base_cardinality.to_formatted_string(&num_format::Locale::en)
    );

    let dimensionality = train_data[0].len();
    mt_log!(
        Level::Info,
        "Dimensionality: {}",
        dimensionality.to_formatted_string(&num_format::Locale::en)
    );

    let queries = queries.iter().take(1000).collect::<Vec<_>>();
    let num_queries = queries.len();
    mt_log!(
        Level::Info,
        "Number of queries: {}",
        num_queries.to_formatted_string(&num_format::Locale::en)
    );

    let csv_name = format!("{}_{}.csv", dataset.name(), (error_rate * 100.).as_u64());

    let report = Report {
        dataset: dataset.name(),
        metric: dataset.metric_name(),
        base_cardinality,
        dimensionality,
        num_queries,
        error_rate,
        ks: ks.to_vec(),
        csv_name: &csv_name,
    };
    report.save(output_dir)?;

    let csv_path = output_dir.join(&csv_name);
    if csv_path.exists() {
        std::fs::remove_file(&csv_path).map_err(|e| e.to_string())?;
    }

    let mut csv_writer = csv::Writer::from_path(csv_path).map_err(|e| e.to_string())?;
    csv_writer
        .write_record([
            "scale",
            "build_time",
            "algorithm",
            "k",
            "throughput",
            "mean_recall",
        ])
        .map_err(|e| e.to_string())?;

    for multiplier in (0..=max_scale).map(|s| 2_usize.pow(s)) {
        mt_log!(Level::Info, "");
        mt_log!(Level::Info, "Scaling data by a factor of {}.", multiplier);
        mt_log!(Level::Info, "Error rate: {}", error_rate);

        let data = if multiplier == 1 {
            train_data.clone()
        } else {
            // If memory cost would be too high, continue to next scale.
            let memory_cost = memory_cost(base_cardinality * multiplier, dimensionality);
            if memory_cost > max_memory * 1024 * 1024 * 1024 {
                mt_log!(
                    Level::Warning,
                    "Memory cost would be over 256G. Skipping scale {multiplier}."
                );
                continue;
            }

            augmentation::augment_data(&train_data, multiplier - 1, error_rate)
        };

        let cardinality = data.len();
        mt_log!(
            Level::Info,
            "Scaled cardinality: {}",
            cardinality.to_formatted_string(&num_format::Locale::en)
        );

        let data_name = format!("{}-{}", dataset.name(), multiplier + 1);
        let data = VecDataset::new(data_name, data, metric, false);
        let criteria = PartitionCriteria::default();

        let start = Instant::now();
        let cakes = Cakes::new(data, seed, &criteria);
        let cakes_time = start.elapsed().as_secs_f32();
        mt_log!(
            Level::Info,
            "Cakes tree-building time: {:.3e} s",
            cakes_time
        );

        let mut prev_linear_throughput = 1_000.0;
        let linear_throughput_threshold = 50.0;

        for &k in ks {
            mt_log!(Level::Info, "k: {}", k);

            #[allow(unused_variables)]
            let (linear_hits, linear_throughput) =
                if prev_linear_throughput >= linear_throughput_threshold {
                    // Measure throughput of linear search.
                    let (linear_hits, linear_throughput) =
                        measure_algorithm(&cakes, &queries, ks[0], knn::Algorithm::Linear);
                    mt_log!(
                        Level::Info,
                        "Linear throughput: {} QPS",
                        format_f32(linear_throughput)
                    );
                    prev_linear_throughput = linear_throughput;
                    (linear_hits, linear_throughput)
                } else {
                    mt_log!(
                        Level::Warning,
                        "Linear throughput is too low. Skipping linear search."
                    );
                    (Vec::new(), prev_linear_throughput)
                };

            line_to_csv(
                &mut csv_writer,
                multiplier,
                cakes_time,
                knn::Algorithm::Linear,
                k,
                linear_throughput,
                1.0,
            )?;

            for &algorithm in knn::Algorithm::variants() {
                mt_log!(Level::Info, "Algorithm: {}, k: {}", algorithm.name(), k);

                let (hits, throughput) = measure_algorithm(&cakes, &queries, k, algorithm);
                mt_log!(Level::Info, "Throughput: {} QPS", format_f32(throughput));

                let misses = hits
                    .iter()
                    .map(Vec::len)
                    .filter(|&h| h < k)
                    .collect::<Vec<_>>();
                if !misses.is_empty() {
                    let &min_hits = misses
                        .iter()
                        .min()
                        .unwrap_or_else(|| unreachable!("`misses` is not empty."));
                    mt_log!(
                        Level::Warning,
                        "{} queries returned as low as {} neighbors.",
                        misses.len(),
                        min_hits
                    );
                }

                let mean_recall = hits
                    .into_iter()
                    .zip(linear_hits.iter().cloned())
                    .map(|(h, l)| utils::compute_recall(h, l))
                    .sum::<f32>()
                    / linear_hits.len().as_f32();
                mt_log!(Level::Info, "Mean recall: {}", format_f32(mean_recall));

                line_to_csv(
                    &mut csv_writer,
                    multiplier,
                    cakes_time,
                    algorithm,
                    k,
                    throughput,
                    mean_recall,
                )?;
            }
        }
    }

    Ok(())
}

/// Measure the throughput of a knn-search algorithm.
///
/// # Arguments
///
/// * `cakes`: the cakes index.
/// * `queries`: the queries.
/// * `k`: the number of nearest neighbors to search for.
///
/// # Returns
///
/// * A vector of the hits for each query.
/// * The throughput of the algorithm.
fn measure_algorithm<'a>(
    cakes: &'a Cakes<Vec<f32>, f32, VecDataset<Vec<f32>, f32>>,
    queries: &'a [&Vec<f32>],
    k: usize,
    algorithm: knn::Algorithm,
) -> (Vec<Vec<(usize, f32)>>, f32) {
    let num_queries = queries.len();
    let start = Instant::now();
    let hits = cakes.batch_knn_search(queries, k, algorithm);
    let elapsed = start.elapsed().as_secs_f32();
    let throughput = num_queries.as_f32() / elapsed;

    (hits, throughput)
}

/// Write a line of scaling-benchmarks to the csv file.
///
/// # Arguments
///
/// * `csv_writer`: the csv writer.
/// * `multiplier`: the dataset cardinality scaling factor.
/// * `cakes_time`: the time to build the cakes index.
/// * `algorithm`: the knn-search algorithm.
/// * `k`: the number of nearest neighbors to search for.
/// * `throughput`: the throughput of the algorithm.
/// * `mean_recall`: the mean recall of the algorithm.
///
/// # Errors
///
/// * If the csv writer fails to write to the file.
/// * If the csv writer fails to flush to the file.
fn line_to_csv(
    csv_writer: &mut csv::Writer<std::fs::File>,
    multiplier: usize,
    cakes_time: f32,
    algorithm: knn::Algorithm,
    k: usize,
    throughput: f32,
    mean_recall: f32,
) -> Result<(), String> {
    csv_writer
        .write_record(&[
            multiplier.to_string(),
            cakes_time.to_string(),
            algorithm.name().to_string(),
            k.to_string(),
            throughput.to_string(),
            mean_recall.to_string(),
        ])
        .map_err(|e| e.to_string())?;

    csv_writer.flush().map_err(|e| e.to_string())?;

    Ok(())
}

const fn memory_cost(cardinality: usize, dimensionality: usize) -> usize {
    cardinality * dimensionality * std::mem::size_of::<f32>()
}

/// A report of the results of a scaling benchmark.
#[derive(Debug, Serialize, Deserialize)]
struct Report<'a> {
    /// Name of the data set.
    dataset: &'a str,
    /// Name of the distance function.
    metric: &'a str,
    /// Cardinality of the real data set.
    base_cardinality: usize,
    /// Dimensionality of the data set.
    dimensionality: usize,
    /// Number of queries.
    num_queries: usize,
    /// Error rate used for scaling.
    error_rate: f32,
    /// Values of k used for knn-search.
    ks: Vec<usize>,
    /// Csv name
    csv_name: &'a str,
}

impl Report<'_> {
    /// Save the report to a file in the given directory.
    fn save(&self, dir: &Path) -> Result<(), String> {
        let e = (self.error_rate * 100.).as_u64();
        let path = dir.join(format!("{}_{}.json", self.dataset, e));
        let report = serde_json::to_string_pretty(&self).map_err(|e| e.to_string())?;
        std::fs::write(path, report).map_err(|e| e.to_string())?;
        Ok(())
    }
}
