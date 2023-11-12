#![deny(clippy::correctness)]
#![warn(
    missing_docs,
    clippy::all,
    clippy::suspicious,
    clippy::style,
    clippy::complexity,
    clippy::perf,
    clippy::pedantic,
    clippy::nursery,
    clippy::missing_docs_in_private_items,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_lossless
)]
#![allow(unused_imports)]

//! Cakes benchmarks on genomic datasets.

mod read_silva;

use core::cmp::Ordering;
use std::{
    path::{Path, PathBuf},
    time::Instant,
};

use abd_clam::{Cakes, Dataset, Instance, VecDataset};
use clap::Parser;
use distances::Number;
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};

fn main() -> Result<(), String> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    let args = Args::parse();

    // Parse the metric.
    let metric = Metric::from_str(&args.metric)?;
    let metric_name = metric.name();
    info!("Using metric: {metric_name}");

    let is_expensive = metric.is_expensive();
    let metric = metric.metric();

    // Check that the data set exists.
    let stem = "silva-SSU-Ref";
    let data_paths = [
        args.input_dir.join(format!("{stem}-unaligned.txt")),
        args.input_dir.join(format!("{stem}-alphabet.txt")),
        args.input_dir.join(format!("{stem}-headers.txt")),
    ];
    for path in &data_paths {
        if !path.exists() {
            return Err(format!("File {path:?} does not exist."));
        }
    }
    let [unaligned_path, _, headers_path] = data_paths;
    info!("Using data from {unaligned_path:?}.");

    // Check that the output directory exists.
    let output_dir = &args.output_dir;
    if !output_dir.exists() {
        return Err(format!("Output directory {output_dir:?} does not exist."));
    }

    let [train, queries, train_headers, query_headers] =
        read_silva::silva_to_dataset(&unaligned_path, &headers_path, metric, is_expensive)?;
    info!(
        "Read {} data set. Cardinality: {}",
        train.name(),
        train.cardinality()
    );
    info!(
        "Read {} query set. Cardinality: {}",
        queries.name(),
        queries.cardinality()
    );
    info!("Read {} training set headers.", train_headers.cardinality());
    info!("Read {} query set headers.", query_headers.cardinality());

    let queries = queries.data().iter().collect::<Vec<_>>();
    let query_headers = query_headers.data().iter().collect::<Vec<_>>();

    // Check if the cakes data structure already exists.
    let cakes_dir = args.input_dir.join(format!("{stem}-cakes"));
    let (cakes, built, build_time) = if cakes_dir.exists() {
        info!("Loading search tree from {cakes_dir:?} ...");
        let start = Instant::now();
        let cakes = Cakes::load(&cakes_dir, metric, is_expensive)?;
        let load_time = start.elapsed().as_secs_f32();
        info!("Loaded search tree in {load_time:.2e} seconds.");

        let tuned_algorithm = cakes.tuned_knn_algorithm();
        let tuned_algorithm = tuned_algorithm.name();
        info!("Tuned algorithm is {tuned_algorithm}");

        (cakes, false, load_time)
    } else {
        let seed = args.seed;
        let criteria = abd_clam::PartitionCriteria::default();

        info!("Creating search tree ...");
        let start = Instant::now();
        let mut cakes = Cakes::new(train, seed, &criteria);
        let build_time = start.elapsed().as_secs_f32();
        info!("Created search tree in {build_time:.2e} seconds.");

        let tuning_depth = args.tuning_depth;
        let tuning_k = args.tuning_k;
        info!("Tuning knn-search with k {tuning_k} and depth {tuning_depth} ...");

        let start = Instant::now();
        cakes.auto_tune_knn(tuning_depth, tuning_k);
        let tuning_time = start.elapsed().as_secs_f32();
        info!("Tuned knn-search in {tuning_time:.2e} seconds.");

        let tuned_algorithm = cakes.tuned_knn_algorithm();
        let tuned_algorithm = tuned_algorithm.name();
        info!("Tuned algorithm is {tuned_algorithm}");

        // Save the Cakes data structure.
        std::fs::create_dir(&cakes_dir).map_err(|e| e.to_string())?;
        cakes.save(&cakes_dir)?;

        (cakes, true, build_time + tuning_time)
    };

    measure_throughput(
        &cakes,
        built,
        build_time,
        &args,
        &queries,
        &query_headers,
        &train_headers,
        stem,
        metric_name,
        output_dir,
    )
}

/// Measure the throughput of the tuned algorithm.
///
/// # Arguments
///
/// * `cakes`: The cakes data structure.
/// * `built`: Whether the cakes data structure was built or loaded from disk.
/// * `build_time`: The time taken to build the cakes data structure or load it
/// from disk.
/// * `args`: The command line arguments.
/// * `queries`: The queries.
/// * `query_headers`: The headers of the queries.
/// * `train_headers`: The headers of the training set.
/// * `stem`: The stem of the data set name.
/// * `metric_name`: The name of the metric.
/// * `output_dir`: The output directory.
///
/// # Errors
///
/// * If the `output_dir` does not exist or cannot be written to.
#[allow(clippy::too_many_arguments)]
fn measure_throughput(
    cakes: &Cakes<String, u32, VecDataset<String, u32>>,
    built: bool,
    build_time: f32,
    args: &Args,
    queries: &[&String],
    query_headers: &[&String],
    train_headers: &VecDataset<String, u32>,
    stem: &str,
    metric_name: &str,
    output_dir: &Path,
) -> Result<(), String> {
    let tuned_algorithm = cakes.tuned_knn_algorithm();
    let tuned_algorithm = tuned_algorithm.name();

    let train = cakes.shards()[0];

    // Perform knn-search for each value of k on all queries.
    for &k in &args.ks {
        info!("Starting knn-search with k = {k} ...");

        // Run the tuned algorithm.
        let start = Instant::now();
        let results = cakes.batch_tuned_knn_search(queries, k);
        let search_time = start.elapsed().as_secs_f32();
        let throughput = queries.len().as_f32() / search_time;
        info!("With k = {k}, achieved throughput of {throughput:.2e} QPS.");

        // Run the linear search algorithm.
        let start = Instant::now();
        let linear_results = cakes.batch_linear_knn_search(queries, k);
        let linear_search_time = start.elapsed().as_secs_f32();
        let linear_throughput = queries.len().as_f32() / linear_search_time;
        info!("With k = {k}, achieved linear search throughput of {linear_throughput:.2e} QPS.",);

        let (mean_recall, hits) = recall_and_hits(
            results,
            linear_results,
            queries,
            query_headers,
            train_headers,
            train,
        );
        info!("With k = {k}, achieved mean recall of {mean_recall:.3}.");

        // Create the report.
        let report = Report {
            dataset: stem,
            metric: metric_name,
            cardinality: train.cardinality(),
            built,
            build_time,
            shard_sizes: cakes.shard_cardinalities(),
            num_queries: queries.len(),
            kind: "knn",
            val: k,
            tuned_algorithm,
            throughput,
            linear_throughput,
            hits,
            mean_recall,
        };

        // Save the report.
        report.save(output_dir)?;
    }

    // Perform range search for each value of r on all queries.
    for &r in &args.rs {
        info!("Starting range search with r = {r} ...");

        #[allow(clippy::cast_possible_truncation)]
        let radius = r as u32;

        // Run the tuned algorithm.
        let start = Instant::now();
        let results = cakes.batch_tuned_rnn_search(queries, radius);
        let search_time = start.elapsed().as_secs_f32();
        let throughput = queries.len().as_f32() / search_time;
        info!("With r = {r}, achieved throughput of {throughput:.2e} QPS.");

        // Run the linear search algorithm.
        let start = Instant::now();
        let linear_results = cakes.batch_linear_rnn_search(queries, radius);
        let linear_search_time = start.elapsed().as_secs_f32();
        let linear_throughput = queries.len().as_f32() / linear_search_time;
        info!("With r = {r}, achieved linear search throughput of {linear_throughput:.2e} QPS.",);

        let (mean_recall, hits) = recall_and_hits(
            results,
            linear_results,
            queries,
            query_headers,
            train_headers,
            train,
        );
        info!("With r = {r}, achieved mean recall of {mean_recall:.3}.");

        // Create the report.
        let report = Report {
            dataset: stem,
            metric: metric_name,
            cardinality: train.cardinality(),
            built,
            build_time,
            shard_sizes: cakes.shard_cardinalities(),
            num_queries: queries.len(),
            kind: "rnn",
            val: r,
            tuned_algorithm,
            throughput,
            linear_throughput,
            hits,
            mean_recall,
        };

        // Save the report.
        report.save(output_dir)?;
    }

    Ok(())
}

/// Compute the recall and hits of the tuned algorithm.
///
/// # Arguments
///
/// * `results`: The results of the tuned algorithm.
/// * `linear_results`: The results of linear search.
/// * `queries`: The queries.
/// * `query_headers`: The headers of the queries.
/// * `train_headers`: The headers of the training set.
/// * `train`: The training set.
///
/// # Returns
///
/// * The mean recall of the tuned algorithm.
/// * The headers of hits of the tuned algorithm.
#[allow(clippy::type_complexity)]
fn recall_and_hits(
    results: Vec<Vec<(usize, u32)>>,
    linear_results: Vec<Vec<(usize, u32)>>,
    queries: &[&String],
    query_headers: &[&String],
    train_headers: &VecDataset<String, u32>,
    train: &VecDataset<String, u32>,
) -> (f32, Vec<(String, Vec<(String, u32)>)>) {
    // Compute the recall of the tuned algorithm.
    let mean_recall = results
        .iter()
        .zip(linear_results)
        .map(|(hits, linear_hits)| compute_recall(hits.clone(), linear_hits))
        .sum::<f32>()
        / queries.len().as_f32();

    // Convert results to original indices.
    let hits = results.into_iter().map(|hits| {
        hits.into_iter()
            .map(|(index, distance)| (train.original_index(index), distance))
            .collect::<Vec<_>>()
    });

    // Collect query header, hit headers, and hit distances.
    let hits = hits
        .zip(query_headers.iter())
        .map(|(hits, &query)| {
            (
                query.clone(),
                hits.into_iter()
                    .map(|(index, distance)| (train_headers[index].clone(), distance))
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();

    (mean_recall, hits)
}

/// CLI arguments for the genomic benchmarks.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the directory containing the input data. This directory should
    /// contain the silva-18s unaligned sequences file, along with the files
    /// containing the headers and alphabet information.
    #[arg(long)]
    input_dir: PathBuf,
    /// Output directory for the report.
    #[arg(long)]
    output_dir: PathBuf,
    /// The metric to use for computing distances. One of "hamming",
    /// "levenshtein", or "needleman-wunsch".
    #[arg(long)]
    metric: String,
    /// The depth of the tree to use for auto-tuning knn-search.
    #[arg(long, default_value = "7")]
    tuning_depth: usize,
    /// The value of k to use for auto-tuning knn-search.
    #[arg(long, default_value = "10")]
    tuning_k: usize,
    /// Number of nearest neighbors to search for.
    #[arg(long, value_parser, num_args = 1.., value_delimiter = ' ', default_value = "10 100")]
    ks: Vec<usize>,
    /// Radii to use for range search.
    #[arg(long, value_parser, num_args = 1.., value_delimiter = ' ', default_value = "25 100 250")]
    rs: Vec<usize>,
    /// Seed for the random number generator.
    #[arg(long)]
    seed: Option<u64>,
}

/// Metrics for computing distances between genomic sequences.
#[derive(Debug)]
enum Metric {
    /// Hamming distance.
    Hamming,
    /// Levenshtein distance.
    Levenshtein,
    /// Needleman-Wunsch distance.
    NeedlemanWunsch,
}

impl Metric {
    /// Return the name of the metric.
    const fn name(&self) -> &str {
        match self {
            Self::Hamming => "hamming",
            Self::Levenshtein => "levenshtein",
            Self::NeedlemanWunsch => "needleman-wunsch",
        }
    }

    /// Return the metric corresponding to the given name.
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "hamming" | "ham" => Ok(Self::Hamming),
            "levenshtein" | "lev" => Ok(Self::Levenshtein),
            "needleman-wunsch" | "needlemanwunsch" | "nw" => Ok(Self::NeedlemanWunsch),
            _ => Err(format!("Unknown metric: {s}")),
        }
    }

    /// Return the metric function.
    #[allow(clippy::ptr_arg)]
    fn metric(&self) -> fn(&String, &String) -> u32 {
        match self {
            Self::Hamming => hamming,
            Self::Levenshtein => levenshtein,
            Self::NeedlemanWunsch => needleman_wunsch,
        }
    }

    /// Return whether the metric is expensive to compute.
    ///
    /// The Hamming distance is cheap to compute, while the Levenshtein and
    /// Needleman-Wunsch distances are expensive to compute.
    const fn is_expensive(&self) -> bool {
        !matches!(self, Self::Hamming)
    }
}

/// Compute the Hamming distance between two strings.
#[allow(clippy::ptr_arg)]
fn hamming(x: &String, y: &String) -> u32 {
    distances::strings::hamming(x, y)
}

/// Compute the Levenshtein distance between two strings.
#[allow(clippy::ptr_arg)]
fn levenshtein(x: &String, y: &String) -> u32 {
    distances::strings::levenshtein(x, y)
}

/// Compute the Needleman-Wunsch distance between two strings.
#[allow(clippy::ptr_arg)]
fn needleman_wunsch(x: &String, y: &String) -> u32 {
    distances::strings::needleman_wunsch::nw_distance(x, y)
}

/// A report of the results of an ANN benchmark.
#[derive(Debug, Serialize, Deserialize)]
struct Report<'a> {
    /// Name of the data set.
    dataset: &'a str,
    /// Name of the distance function.
    metric: &'a str,
    /// Number of data points in the data set.
    cardinality: usize,
    /// Whether the search tree was built or loaded from disk.
    built: bool,
    /// Time taken to build the search tree or load it from disk.
    build_time: f32,
    /// Sizes of the shards created for `ShardedCakes`.
    shard_sizes: Vec<usize>,
    /// Number of queries used for search.
    num_queries: usize,
    /// The kind of search performed. One of "knn" or "rnn".
    kind: &'a str,
    /// Value of k used for knn-search or value of r used for range search.
    val: usize,
    /// Name of the algorithm used after auto-tuning.
    tuned_algorithm: &'a str,
    /// Throughput of the tuned algorithm.
    throughput: f32,
    /// Throughput of linear search.
    linear_throughput: f32,
    /// Hits for each query.
    hits: Vec<(String, Vec<(String, u32)>)>,
    /// Mean recall of the tuned algorithm.
    mean_recall: f32,
}

impl Report<'_> {
    /// Save the report to a file in the given directory.
    ///
    /// # Arguments
    ///
    /// * `dir`: The directory to save the report to.
    ///
    /// # Errors
    ///
    /// * If the directory does not exist or cannot be written to.
    /// * If the report cannot be serialized.
    fn save(&self, dir: &Path) -> Result<(), String> {
        if !dir.exists() {
            return Err(format!("Directory {dir:?} does not exist."));
        }

        if !dir.is_dir() {
            return Err(format!("{dir:?} is not a directory."));
        }

        let path = dir.join(format!("{}_{}_{}.json", self.dataset, self.kind, self.val));
        let report = serde_json::to_string_pretty(&self).map_err(|e| e.to_string())?;
        std::fs::write(path, report).map_err(|e| e.to_string())?;
        Ok(())
    }
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
            if (hit - linear_hit).abs() <= U::epsilon() {
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
        debug!("Recall: {recall:.3}, num_common: {num_common}");

        recall
    }
}
