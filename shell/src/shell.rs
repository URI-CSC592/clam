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
    run_benchmarks(
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
fn run_benchmarks(
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

    // Benchmark build times
    info!("Running build time benchmarks");

    let (build_time_naive, build_time_entropy) = benchmark_build_times(
        data.clone(),
        seed,
        num_trials,
        &default_criteria,
        &entropy_criteria,
    )?;

    print_results(
        "Build time",
        (build_time_naive * 1000.0, build_time_entropy * 1000.0),
        "ms",
    );

    let results = vec![vec![
        build_time_naive.to_string(),
        build_time_entropy.to_string(),
    ]];

    write_results(
        output_dir,
        format!(
            "build-{}-{}seed-{}trials",
            dataset,
            seed.unwrap(),
            num_trials,
        )
        .as_str(),
        vec!["Build time (naive)", "Build time (entropy)"],
        results,
    );

    // Benchmark throughput 2D array that is ks.len() x 3
    // Each element is a vector of strings and should be preassigned
    let mut throughput_results = Vec::with_capacity(ks.len());
    for k in &ks {
        throughput_results.push(vec![k.to_string(), String::new(), String::new()]);
    }

    for i in 0..ks.len() {
        let k = ks[i];

        let (throughput_naive, throughput_entropy) = benchmark_throughput(
            data.clone(),
            queries.clone(),
            k,
            seed,
            num_trials,
            &default_criteria,
            &entropy_criteria,
        )?;

        throughput_results[i][1] = throughput_naive.to_string();
        throughput_results[i][2] = throughput_entropy.to_string();

        print_results("Throughput", (throughput_naive, throughput_entropy), "QPS");
    }

    // Write the throughput results
    write_results(
        output_dir,
        format!(
            "query-{}-{}seed-{}trials",
            dataset,
            seed.unwrap(),
            num_trials,
        )
        .as_str(),
        vec!["k", "Throughput (QPS, naive)", "Throughput (QPS, entropy)"],
        throughput_results,
    );

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

/// Benchmark SHELL build times
///
/// # Arguments
///
/// * `data` - The dataset
/// * `seed` - The random seed to use
/// * `num_trials` - The number of trials to run
/// * `naive_criteria` - The naive partition criterion
/// * `entropy_criteria` - The entropy partition criterion
///
/// # Returns
///
/// A tuple of (naive_build_time, entropy_build_time)
///
fn benchmark_build_times(
    data: VecDataset<Vec<f32>, f32, String>,
    seed: Option<u64>,
    num_trials: usize,
    naive_criteria: &PartitionCriteria<f32>,
    entropy_criteria: &PartitionCriteria<f32>,
) -> Result<(f64, f64), String> {
    let mut build_times_naive = Vec::with_capacity(num_trials);
    let mut build_times_entropy = Vec::with_capacity(num_trials);

    for _ in 0..num_trials {
        let data_clone = data.clone();
        let start = std::time::Instant::now();
        let mut _cakes = Cakes::new(data_clone, seed, entropy_criteria);
        let build_time = start.elapsed().as_secs_f64();
        build_times_entropy.push(build_time);

        let data_clone = data.clone();
        let start = std::time::Instant::now();
        let mut _cakes = Cakes::new(data_clone, seed, naive_criteria);
        let build_time = start.elapsed().as_secs_f64();
        build_times_naive.push(build_time);
    }

    // Average the build times
    let (build_time_naive, build_time_entropy) = (
        build_times_naive.iter().sum::<f64>() / num_trials as f64,
        build_times_entropy.iter().sum::<f64>() / num_trials as f64,
    );

    Ok((build_time_naive, build_time_entropy))
}

/// Print benchmark results
///
/// # Arguments
///
/// * `test_name` - The name of the test
/// * `results` - The results of the test (in the format (naive, entropy))
/// * `unit` - The unit of the results
///
fn print_results(test_name: &str, results: (f64, f64), unit: &str) {
    info!("{} (naive): {:.2} {}", test_name, results.0, unit);
    info!("{} (entropy): {:.2} {}", test_name, results.1, unit);
}

/// Write benchmark results to a csv
///
/// # Arguments
///
/// * `output_dir` - The path to the output directory
/// * `test_name` - The name of the test
/// * `headers` - The headers for the csv
/// * `results` - The results of the test
///
fn write_results(
    output_dir: &Path,
    test_name: &str,
    headers: Vec<&str>,
    results: Vec<Vec<String>>,
) {
    let mut writer = csv::Writer::from_path(output_dir.join(format!("{}.csv", test_name))).unwrap();
    writer.write_record(headers).unwrap();

    for result in results {
        writer.write_record(result).unwrap();
    }

    writer.flush().unwrap();
}

/// Benchmark SHELL throughput rates
///
/// # Arguments
///
/// * `data` - The dataset
/// * `queries` - The queries
/// * `k` - The value of k
/// * `seed` - The random seed to use
/// * `num_trials` - The number of trials to run
/// * `naive_criteria` - The naive partition criterion
/// * `entropy_criteria` - The entropy partition criterion
///
/// # Returns
///
/// A tuple of (naive_throughput, entropy_throughput)
///
fn benchmark_throughput(
    data: VecDataset<Vec<f32>, f32, String>,
    queries: Vec<&Vec<f32>>,
    k: usize,
    seed: Option<u64>,
    num_trials: usize,
    naive_criteria: &PartitionCriteria<f32>,
    entropy_criteria: &PartitionCriteria<f32>,
) -> Result<(f64, f64), String> {
    info!("Running throughput benchmarks (k = {})", k);
    // Store the throughputs
    let mut throughput_naive = Vec::with_capacity(num_trials);
    let mut throughput_entropy = Vec::with_capacity(num_trials);

    // Create Cakes instance for use with naive partitioning criteria
    let data_clone = data.clone();
    let mut cakes = Cakes::new(data_clone, seed, naive_criteria);
    cakes.auto_tune_knn(k, 10);

    for _ in 0..num_trials {
        let start = std::time::Instant::now();
        cakes.batch_tuned_knn_search(&queries, k);
        let elapsed = start.elapsed().as_secs_f64();
        throughput_naive.push(queries.len() as f64 / elapsed);
    }

    // Create Cakes instance for use with entropy partitioning criteria
    let data_clone = data.clone();
    let mut cakes = Cakes::new(data_clone, seed, entropy_criteria);
    cakes.auto_tune_knn(k, 10);

    for _ in 0..num_trials {
        let start = std::time::Instant::now();
        cakes.batch_tuned_knn_search(&queries, k);
        let elapsed = start.elapsed().as_secs_f64();
        throughput_entropy.push(queries.len() as f64 / elapsed);
    }

    // Average the throughput rates
    let (throughput_naive, throughput_entropy) = (
        throughput_naive.iter().sum::<f64>() / num_trials as f64,
        throughput_entropy.iter().sum::<f64>() / num_trials as f64,
    );
    Ok((throughput_naive, throughput_entropy))
}
