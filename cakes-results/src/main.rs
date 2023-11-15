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

//! CLI for running Cakes experiments and benchmarks.

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use mt_logger::{mt_flush, mt_log, mt_new, Level, OutputStream};

mod genomic;
mod utils;
mod vectors;

/// CLI for running Cakes experiments and benchmarks.
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Path to the directory with the data sets. The directory should contain
    /// the hdf5 files downloaded from the ann-benchmarks repository.
    #[arg(long)]
    input_dir: PathBuf,

    /// Output directory for the report.
    #[arg(long)]
    output_dir: PathBuf,

    /// Optional seed for the random number generator.
    #[arg(long)]
    seed: Option<u64>,

    /// Name of the data set to use.
    #[arg(long)]
    dataset: String,

    /// Subcommands for the CLI.
    #[command(subcommand)]
    command: Commands,
}

/// Subcommands for the CLI.
#[derive(Subcommand)]
enum Commands {
    /// Runs RNN search.
    Rnn {
        /// Whether to shard the data set for search.
        #[arg(long)]
        use_shards: bool,

        /// The fractions of the root radius to use for rnn-search.
        #[arg(long, value_parser, num_args = 1.., value_delimiter = ' ', default_value = "0.01 0.05")]
        radius_fractions: Vec<f32>,
    },

    /// Runs KNN search.
    Knn {
        /// Whether to shard the data set for search.
        #[arg(long)]
        use_shards: bool,

        /// The depth of the tree to use for auto-tuning knn-search.
        #[arg(long, default_value = "10")]
        tuning_depth: usize,

        /// The value of k to use for auto-tuning knn-search.
        #[arg(long, default_value = "10")]
        tuning_k: usize,

        /// Number of nearest neighbors to search for.
        #[arg(long, value_parser, num_args = 1.., value_delimiter = ' ', default_value = "10 100")]
        ks: Vec<usize>,
    },

    /// Generates augmented data sets for scaling experiments.
    Scaling {
        /// Maximum scaling factor. The data set will be scaled by factors of
        /// `2 ^ i` for `i` in `0..=max_scale`.
        #[arg(long, default_value = "16")]
        max_scale: u32,

        /// Error rate used for scaling.
        #[arg(long, default_value = "0.01")]
        error_rate: f32,

        /// Maximum memory usage (in gigabytes) for the scaled data sets.
        #[arg(long, default_value = "256")]
        max_memory: f32,

        /// Whether to shard the data set for search.
        #[arg(long)]
        use_shards: bool,

        /// The depth of the tree to use for auto-tuning knn-search.
        #[arg(long, default_value = "10")]
        tuning_depth: usize,

        /// The value of k to use for auto-tuning knn-search.
        #[arg(long, default_value = "10")]
        tuning_k: usize,

        /// Number of nearest neighbors to search for.
        #[arg(long, value_parser, num_args = 1.., value_delimiter = ' ', default_value = "10 100")]
        ks: Vec<usize>,

        /// The fractions of the root radius to use for rnn-search.
        #[arg(long, value_parser, num_args = 1.., value_delimiter = ' ', default_value = "0.01 0.05")]
        radius_fractions: Vec<f32>,
    },

    /// Runs KNN and RNN search on Silva-18S dataset.
    Genomic {
        /// The metric to use for computing distances. One of "hamming",
        /// "levenshtein", or "needleman-wunsch".
        #[arg(long, default_value = "levenshtein")]
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
    },
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), String> {
    mt_new!(None, Level::Info, OutputStream::StdOut);

    let cli = Cli::parse();

    check_dir(&cli.input_dir)?;
    check_dir(&cli.output_dir)?;

    mt_log!(Level::Info, "Input directory: {}", cli.input_dir.display());
    mt_log!(
        Level::Info,
        "Output directory: {}",
        cli.output_dir.display()
    );

    let dataset = &cli.dataset;

    // You can check for the existence of subcommands, and if found use their
    // matches just as you would the top level cmd
    match &cli.command {
        Commands::Rnn {
            use_shards,
            radius_fractions,
        } => {
            vectors::rnn_search(
                &cli.input_dir,
                dataset,
                *use_shards,
                radius_fractions,
                cli.seed,
                &cli.output_dir,
            )?;
        }
        Commands::Knn {
            use_shards,
            tuning_depth,
            tuning_k,
            ks,
        } => {
            vectors::knn_search(
                &cli.input_dir,
                dataset,
                *use_shards,
                *tuning_depth,
                *tuning_k,
                ks,
                cli.seed,
                &cli.output_dir,
            )?;
        }
        Commands::Scaling {
            max_scale,
            error_rate,
            max_memory,
            use_shards,
            tuning_depth,
            tuning_k,
            ks,
            radius_fractions,
        } => {
            let scaled_names = vectors::augment_dataset(
                &cli.input_dir,
                dataset,
                *max_scale,
                *error_rate,
                *max_memory,
                &cli.input_dir,
            )?;
            for scaled_name in scaled_names {
                vectors::rnn_search(
                    &cli.input_dir,
                    &scaled_name,
                    *use_shards,
                    radius_fractions,
                    cli.seed,
                    &cli.output_dir,
                )?;
                vectors::knn_search(
                    &cli.input_dir,
                    &scaled_name,
                    *use_shards,
                    *tuning_depth,
                    *tuning_k,
                    ks,
                    cli.seed,
                    &cli.output_dir,
                )?;
            }
        }
        Commands::Genomic {
            metric,
            tuning_depth,
            tuning_k,
            ks,
            rs,
        } => {
            genomic::run(
                &cli.input_dir,
                dataset,
                metric,
                cli.seed,
                *tuning_depth,
                *tuning_k,
                ks,
                rs,
                &cli.output_dir,
            )?;
        }
    }

    mt_flush!().map_err(|e| e.to_string())?;

    Ok(())
}

/// Checks that the given path exists and is a directory.
///
/// # Arguments
///
/// * `path` - The path to check.
///
/// # Errors
///
/// * If the path does not exist.
/// * If the path is not a directory.
fn check_dir(path: &Path) -> Result<(), String> {
    if !path.exists() {
        return Err(format!(
            "The input directory '{}' does not exist.",
            path.display()
        ));
    }
    if !path.is_dir() {
        return Err(format!(
            "The input directory '{}' is not a directory.",
            path.display()
        ));
    }
    Ok(())
}
