pub(crate) mod ann_datasets;
pub(crate) mod utils;
use abd_clam::{Cakes, PartitionCriteria, VecDataset};
use ann_datasets::AnnDatasets;
use clap::Parser;
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
///
fn benchmark_shell(
    input_dir: &Path,
    output_dir: &Path,
    dataset: &str,
    ks: Vec<usize>,
    seed: Option<u64>,
) -> Result<(), String> {
    // Load the dataset
    let train_path = input_dir.join(format!("{}-train.npy", dataset));
    let test_path = input_dir.join(format!("{}-test.npy", dataset));

    log::debug!("Dataset: {}", dataset);
    log::debug!("Train path: {:?}", train_path);
    log::debug!("Test path: {:?}", test_path);
    log::debug!("Output directory: {:?}", output_dir);
    log::debug!("ks: {:?}", ks);
    log::debug!("Seed: {:?}", seed);

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
    log::info!("Dataset Name: {}", data.name());
    log::info!("Training Cardinality: {}", cardinality);
    log::info!("Training Dimensionality: {}", dimensionality);
    log::info!("Queries: {}", queries.len());

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
    let data2 = data.clone();

    let entropy_criterion = utils::MinShannonEntropy {
        threshold: 0.3,
        // TODO: Remove this clone
        data: data.clone(),
    };

    // Create Partition Criteria
    let entropy_criteria = PartitionCriteria::default().with_custom(Box::new(entropy_criterion));

    let default_criteria = PartitionCriteria::default();

    // Benchmark with the custom partition criteria
    // Start a timer
    let start = std::time::Instant::now();

    // Create the model
    let _model = Cakes::new(data, seed, &entropy_criteria);

    // Stop the timer
    let elapsed = start.elapsed();

    // Print the time
    log::info!("Time to build custom model: {:?}", elapsed);

    // Benchmark with the default partition criteria
    let start = std::time::Instant::now();

    // Create the model
    let _model = Cakes::new(data2, seed, &default_criteria);

    // Stop the timer
    let elapsed = start.elapsed();

    // Print the time
    log::info!("Time to build default model: {:?}", elapsed);

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
