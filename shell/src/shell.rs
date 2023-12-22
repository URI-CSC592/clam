pub(crate) mod ann_datasets;
pub(crate) mod utils;
use abd_clam::{Cakes, PartitionCriteria, VecDataset};
use ann_datasets::AnnDatasets;
use clap::Parser;
use log::{debug, info};
use std::path::Path;

// Command line arguments for SHELL
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    input_dir: std::path::PathBuf,
    #[arg(long)]
    output_dir: std::path::PathBuf,
    #[arg(long, default_value = "random")]
    dataset: String,
    #[arg(long, value_parser, num_args = 1.., value_delimiter = ' ', default_value = "10 100")]
    ks: Vec<usize>,
    #[arg(long, value_parser, num_args = 1, default_value = "42")]
    seed: Option<u64>,
    #[arg(long, value_parser, num_args = 1, default_value = "100")]
    num_trials: usize,
}

fn main() -> Result<(), String> {
    // Initialize the logger.
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    // Get the command line arguments.
    let args = Args::parse();

    // Check the input directory, output directory, and dataset.
    check_datasets(&args.input_dir, &args.dataset)?;
    check_output_dir(&args.output_dir)?;

    // Benchmark SHELL.
    benchmark_shell(
        &args.input_dir,
        &args.output_dir,
        &args.dataset,
        args.ks,
        args.seed,
        args.num_trials,
    )?;

    Ok(())
}

/// Check that the dataset exists in the input directory
///
/// # Arguments
///
/// * `input_dir` - The directory containing the datasets
/// * `dataset` - The name of the dataset
///
/// # Panics
///
/// If the dataset train and test files do not exist in the input directory
///
fn check_datasets(input_dir: &Path, dataset: &str) -> Result<(), String> {
    let data_paths = [
        input_dir.join(format!("{}-train.npy", dataset)),
        input_dir.join(format!("{}-test.npy", dataset)),
        input_dir.join(format!("{}-train.csv", dataset)),
        input_dir.join(format!("{}-test.csv", dataset)),
    ];
    for path in &data_paths {
        if !path.exists() {
            return Err(format!("File {path:?} does not exist."));
        }
    }
    Ok(())
}

/// Check that output_dir exists
///
/// # Arguments
///
/// * `output_dir` - The directory to check
///
/// # Panics
///
/// If output_dir does not exist
///
fn check_output_dir(output_dir: &Path) -> Result<(), String> {
    if !output_dir.exists() {
        return Err(format!("Output directory {:?} does not exist.", output_dir));
    }
    Ok(())
}

/// Benchmark SHELL on a given dataset
///
/// # Arguments
///
/// * `input_dir` - The directory containing the datasets
/// * `output_dir` - The path to the output directory
/// * `dataset` - The name of the dataset
/// * `ks` - The values of k to use
/// * `seed` - The random seed to use
/// * `num_trials` - The number of trials to run
///
fn benchmark_shell(
    input_dir: &Path,
    output_dir: &Path,
    dataset: &str,
    ks: Vec<usize>,
    seed: Option<u64>,
    num_trials: usize,
) -> Result<(), String> {
    // Load the dataset
    let train_path = input_dir.join(format!("{}-train.npy", dataset));
    let test_path = input_dir.join(format!("{}-test.npy", dataset));

    debug!("Dataset: {}", dataset);
    debug!("Train path: {:?}", train_path);
    debug!("Test path: {:?}", test_path);
    debug!("Output directory: {:?}", output_dir);
    debug!("ks: {:?}", ks);
    debug!("Seed: {:?}", seed);

    // Load the train and test data
    let data = AnnDatasets::from_str(dataset)?;
    let [train_data, test_data] = data.read(input_dir)?;

    // Get properties of the dataset
    let (cardinality, dimensionality, metric, queries) = (
        train_data.len(),
        train_data[0].len(),
        data.metric()?,
        test_data.iter().collect::<Vec<_>>(),
    );

    // Print the dataset properties
    info!("Dataset Name: {}", data.name());
    info!("Training Cardinality: {}", cardinality);
    info!("Training Dimensionality: {}", dimensionality);
    info!("Queries: {}", queries.len());

    // Create dummy metadata for the dataset
    // The dummy metadata is a Vec of strings from 0 to 9
    // TODO: Get metadata from a separate file
    //let metadata = (0..cardinality).map(|i| i.to_string()).collect::<Vec<_>>();

    // Get metadata from CSV files
    // TODO: Fix metadata files. The CSVs for fashion-mnist are incorrect and missing for other datasets
    let (train_metadata, _test_metadata) = read_metadata(input_dir, dataset)?;

    // Make sure the metadata is the same length as the data
    assert_eq!(
        train_metadata.len(),
        train_data.len(),
        "Metadata length does not match data length"
    );

    // Create data VecDataset
    let data: VecDataset<Vec<f32>, f32, String> = VecDataset::new(
        data.name().to_string(),
        train_data,
        metric,
        false,
        Some(train_metadata),
    );

    let entropy_criterion = utils::MinShannonEntropy {
        threshold: 0.3,
        // TODO: Remove this clone
        data: data.clone(),
    };

    // Create Partition Criteria
    let entropy_criteria = PartitionCriteria::default().with_custom(Box::new(entropy_criterion));
    let default_criteria = PartitionCriteria::default();

    // Benchmark building the models
    //benchmark_build_model(&data, seed, &entropy_criteria, &default_criteria)?;

    // Benchmark querying the models
    //benchmark_query_model(&data, &queries, &ks, seed, &entropy_criteria, &default_criteria)?;

    // Benchmark building the models
    let mut build_times_entropy = Vec::with_capacity(num_trials);
    let mut build_times_default = Vec::with_capacity(num_trials);

    for _ in 0..num_trials {
        let data_clone = data.clone();
        let start = std::time::Instant::now();
        let mut _cakes = Cakes::new(data_clone, seed, &entropy_criteria);
        let build_time = start.elapsed().as_secs_f64();
        build_times_entropy.push(build_time);

        let data_clone = data.clone();
        let start = std::time::Instant::now();
        let mut _cakes = Cakes::new(data_clone, seed, &default_criteria);
        let build_time = start.elapsed().as_secs_f64();
        build_times_default.push(build_time);
    }

    // Average the build times
    let (build_time_entropy, build_time_default) = (
        build_times_entropy.iter().sum::<f64>() / num_trials as f64,
        build_times_default.iter().sum::<f64>() / num_trials as f64,
    );

    // Print the build times in ms
    info!(
        "Build time (entropy): {:.3} ms",
        build_time_entropy * 1000.0
    );

    info!(
        "Build time (default): {:.3} ms",
        build_time_default * 1000.0
    );

    // Write the build time averages to a csv
    let mut build_writer =
        csv::Writer::from_path(output_dir.join(format!("build-{}-{}.csv", dataset, seed.unwrap())))
            .unwrap();
    build_writer
        .write_record(["default (s)", "entropy (s)", "dataset", "trials"])
        .unwrap();
    build_writer
        .write_record(&[
            build_time_default.to_string(),
            build_time_entropy.to_string(),
            dataset.to_string(),
            num_trials.to_string(),
        ])
        .unwrap();
    build_writer.flush().unwrap();

    for k in &ks {
        info!("Running query benchmarks (k = {})", k);
        // Store the throughputs
        let mut throughput_default = Vec::with_capacity(num_trials);
        let mut throughput_entropy = Vec::with_capacity(num_trials);

        let data_clone = data.clone();
        let mut cakes_default = Cakes::new(data_clone, seed, &default_criteria);

        let data_clone = data.clone();
        let mut cakes_entropy = Cakes::new(data_clone, seed, &entropy_criteria);

        cakes_default.auto_tune_knn(*k, 10);
        cakes_entropy.auto_tune_knn(*k, 10);

        for _ in 0..num_trials {
            let start = std::time::Instant::now();
            cakes_default.batch_tuned_knn_search(&queries, *k);
            let elapsed = start.elapsed().as_secs_f64();
            throughput_default.push(queries.len() as f64 / elapsed);

            let start = std::time::Instant::now();
            cakes_entropy.batch_tuned_knn_search(&queries, *k);
            let elapsed = start.elapsed().as_secs_f64();
            throughput_entropy.push(queries.len() as f64 / elapsed);
        }

        // Average the throughputs
        let (throughput_default, throughput_entropy) = (
            throughput_default.iter().sum::<f64>() / num_trials as f64,
            throughput_entropy.iter().sum::<f64>() / num_trials as f64,
        );

        // Print the throughputs in queries per second
        info!(
            "Throughput: {:.3} QPS (k = {}, default)",
            throughput_default, k
        );

        info!(
            "Throughput: {:.3} QPS (k = {}, entropy)",
            throughput_entropy, k
        );

        // Write the throughputs to a csv in the format throughputs-<dataset>-<k>.csv
        let mut throughput_writer = csv::Writer::from_path(output_dir.join(format!(
            "throughputs-{}-{}-{}.csv",
            dataset,
            k,
            seed.unwrap()
        )))
        .unwrap();
        throughput_writer
            .write_record(["QPS (default)", "QPS (entropy)", "dataset", "k", "trials"])
            .unwrap();
        throughput_writer
            .write_record(&[
                throughput_default.to_string(),
                throughput_entropy.to_string(),
                dataset.to_string(),
                k.to_string(),
                num_trials.to_string(),
            ])
            .unwrap();
        throughput_writer.flush().unwrap();
    }

    Ok(())
}

/// Read metadata from CSV files
///
/// # Arguments
///
/// * `input_dir` - The directory containing the metadata files
/// * `dataset` - The name of the dataset
///
/// # Returns
///
/// A tuple of (train_metadata, test_metadata) for the dataset
///
fn read_metadata(input_dir: &Path, dataset: &str) -> Result<(Vec<String>, Vec<String>), String> {
    let train_path = input_dir.join(format!("{}-train.csv", dataset));
    let test_path = input_dir.join(format!("{}-test.csv", dataset));

    let train_metadata = read_metadata_file(&train_path)?;
    let test_metadata = read_metadata_file(&test_path)?;

    Ok((train_metadata, test_metadata))
}

/// Read metadata from a CSV file
///
/// # Arguments
///
/// * `path` - The path to the CSV file
///
/// # Returns
///
/// A vector of strings containing the metadata
///
fn read_metadata_file(path: &std::path::PathBuf) -> Result<Vec<String>, String> {
    let mut rdr =
        csv::Reader::from_path(path).map_err(|e| format!("Error reading metadata file: {}", e))?;

    let metadata = rdr
        .records()
        .map(|r| r.map_err(|e| format!("Error reading metadata file: {}", e)))
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .map(|r| r[0].to_string())
        .collect::<Vec<_>>();

    Ok(metadata)
}
