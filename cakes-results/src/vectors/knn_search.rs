//! Benchmark the performance of KNN search algorithms.

use std::{path::Path, time::Instant};

use abd_clam::{Cakes, Dataset, PartitionCriteria, VecDataset};
use distances::Number;
use mt_logger::{mt_log, Level};
use num_format::ToFormattedString;
use serde::{Deserialize, Serialize};

use crate::{
    utils::{compute_recall, format_f32},
    vectors::ann_datasets::AnnDatasets,
};

/// Report the results of an ANN benchmark.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub fn knn_search(
    input_dir: &Path,
    dataset: &str,
    use_shards: bool,
    tuning_depth: usize,
    tuning_k: usize,
    ks: &[usize],
    seed: Option<u64>,
    output_dir: &Path,
    run_linear: bool,
) -> Result<(), String> {
    mt_log!(Level::Info, "Start knn_search on {dataset}");

    let dataset = AnnDatasets::from_str(dataset)?;
    let metric = dataset.metric()?;
    let [train_data, queries] = dataset.read(input_dir)?;
    mt_log!(Level::Info, "Dataset: {}", dataset.name());

    let (cardinality, dimensionality) = (train_data.len(), train_data[0].len());
    mt_log!(
        Level::Info,
        "Cardinality: {}",
        cardinality.to_formatted_string(&num_format::Locale::en)
    );
    mt_log!(
        Level::Info,
        "Dimensionality: {}",
        dimensionality.to_formatted_string(&num_format::Locale::en)
    );

    let queries = queries.iter().collect::<Vec<_>>();
    let num_queries = queries.len();
    mt_log!(
        Level::Info,
        "Number of queries: {}",
        num_queries.to_formatted_string(&num_format::Locale::en)
    );

    let cakes = if use_shards {
        let max_cardinality = if cardinality < 1_000_000 {
            cardinality
        } else if cardinality < 5_000_000 {
            100_000
        } else {
            1_000_000
        };

        let shards =
            VecDataset::new(dataset.name(), train_data, metric, false).make_shards(max_cardinality);
        let mut cakes = Cakes::new_randomly_sharded(shards, seed, &PartitionCriteria::default());
        cakes.auto_tune_knn(tuning_k, tuning_depth);
        cakes
    } else {
        let data = VecDataset::new(dataset.name(), train_data, metric, false);
        let mut cakes = Cakes::new(data, seed, &PartitionCriteria::default());
        cakes.auto_tune_knn(tuning_k, tuning_depth);
        cakes
    };

    let shard_sizes = cakes.shard_cardinalities();
    mt_log!(
        Level::Info,
        "Shard sizes: [{}]",
        shard_sizes
            .iter()
            .map(|s| s.to_formatted_string(&num_format::Locale::en))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let algorithm = cakes.tuned_knn_algorithm();
    mt_log!(Level::Info, "Tuned algorithm: {}", algorithm.name());

    for &k in ks {
        mt_log!(Level::Info, "k: {k}");

        let start = Instant::now();
        let hits = cakes.batch_tuned_knn_search(&queries, k);
        let elapsed = start.elapsed().as_secs_f32();
        let throughput = queries.len().as_f32() / elapsed;
        mt_log!(Level::Info, "Throughput: {} QPS", format_f32(throughput));

        let (linear_throughput, recall) = if run_linear {
            let start = Instant::now();
            let linear_hits = cakes.batch_linear_knn_search(&queries, k);
            let linear_elapsed = start.elapsed().as_secs_f32();
            let linear_throughput = queries.len().as_f32() / linear_elapsed;
            mt_log!(
                Level::Info,
                "Linear throughput: {} QPS",
                format_f32(linear_throughput)
            );

            let speedup_factor = throughput / linear_throughput;
            mt_log!(
                Level::Info,
                "Speedup factor: {}",
                format_f32(speedup_factor)
            );

            let recall = hits
                .into_iter()
                .zip(linear_hits)
                .map(|(hits, linear_hits)| compute_recall(hits, linear_hits))
                .sum::<f32>()
                / queries.len().as_f32();
            mt_log!(Level::Info, "Recall: {}", format_f32(recall));

            (Some(linear_throughput), Some(recall))
        } else {
            (None, None)
        };

        Report {
            dataset: &dataset.name(),
            metric: dataset.metric_name(),
            cardinality,
            dimensionality,
            shard_sizes: shard_sizes.clone(),
            num_queries,
            k,
            tuned_algorithm: algorithm.name(),
            throughput,
            linear_throughput,
            recall,
        }
        .save(output_dir)?;
    }

    Ok(())
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
    /// Dimensionality of the data set.
    dimensionality: usize,
    /// Sizes of the shards created for `ShardedCakes`.
    shard_sizes: Vec<usize>,
    /// Number of queries used for search.
    num_queries: usize,
    /// Number of nearest neighbors to search for.
    k: usize,
    /// Name of the algorithm used after auto-tuning.
    tuned_algorithm: &'a str,
    /// Throughput of the tuned algorithm.
    throughput: f32,
    /// Throughput of linear search.
    linear_throughput: Option<f32>,
    /// Recall of the tuned algorithm.
    recall: Option<f32>,
}

impl Report<'_> {
    /// Save the report to a file in the given directory.
    fn save(&self, dir: &Path) -> Result<(), String> {
        let path = dir.join(format!("knn-{}-{}.json", self.dataset, self.k));
        let report = serde_json::to_string_pretty(&self).map_err(|e| e.to_string())?;
        std::fs::write(path, report).map_err(|e| e.to_string())?;
        Ok(())
    }
}
