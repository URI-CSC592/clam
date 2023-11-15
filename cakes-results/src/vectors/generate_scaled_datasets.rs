//! Generate augmented datasets by scaling the ann-datasets.

use std::path::Path;

use distances::Number;
use mt_logger::{mt_log, Level};
use rand::prelude::*;

use crate::vectors::ann_datasets::AnnDatasets;

/// Creates augmented datasets by scaling the ann-datasets.
///
/// The augmented datasets are stored in the `output_dir` directory. The
/// augmented datasets are created by scaling the ann-datasets by a factor
/// between 1 and `max_multiplier`. The scaling factor is chosen such that the
/// augmented dataset has a memory cost of at most `max_memory` GB.
///
/// For an input dataset with name "dataset.npy", the augmented datasets are
/// named "<dataset>-scale-<multiplier>.npy" where `multiplier` is the multiplicative
/// factor used to scale the dataset.
///
/// # Arguments
///
/// * `input_dir` - The directory containing the ann-datasets.
/// * `dataset` - The name of the ann-dataset to augment.
/// * `max_multiplier` - The maximum cardinality multiplier.
/// * `error_rate` - The maximum error rate.
/// * `max_memory` - The maximum memory cost of the augmented dataset in GB.
/// * `output_dir` - The directory to store the augmented datasets.
///
/// # Returns
///
/// The names of the augmented datasets.
///
/// # Errors
///
/// * If the dataset does not exist.
/// * If the dataset cannot be read.
/// * If the dataset cannot be saved.
pub fn augment_dataset(
    input_dir: &Path,
    dataset: &str,
    max_multiplier: u32,
    error_rate: f32,
    max_memory: f32,
    output_dir: &Path,
) -> Result<Vec<String>, String> {
    // Read the dataset.
    let dataset = AnnDatasets::from_str(dataset)?;
    let [train_data, _] = dataset.read(input_dir)?;

    let base_cardinality = train_data.len();
    let dimensionality = train_data[0].len();
    mt_log!(
        Level::Info,
        "Read dataset {} with {base_cardinality} points and {dimensionality} dimensions.",
        dataset.name()
    );

    let mut scaled_names = Vec::new();

    for multiplier in (1..=max_multiplier).map(|s| 2_usize.pow(s)) {
        let memory_cost = memory_cost(base_cardinality, dimensionality);
        if memory_cost > max_memory {
            mt_log!(
                Level::Info,
                "Stopping at multiplier {} because memory cost is {} GB.",
                multiplier,
                memory_cost
            );
            break;
        }
        mt_log!(
            Level::Info,
            "Augmenting dataset {} with multiplier {multiplier}.",
            dataset.name()
        );

        let data = if dataset.name().starts_with("random") {
            let cardinality = base_cardinality * multiplier;
            let (min_val, max_val) = (-1.0, 1.0);
            let seed = rand::thread_rng().gen();
            symagen::random_data::random_tabular_seedable(
                cardinality,
                dimensionality,
                min_val,
                max_val,
                seed,
            )
        } else {
            symagen::augmentation::augment_data(&train_data, multiplier - 1, error_rate)
        };

        let scaled_name = format!("{}-scale-{multiplier}", dataset.name());
        let output_path = output_dir.join(format!("{scaled_name}-train.npy"));
        save_npy(data, &output_path)?;
        scaled_names.push(scaled_name);

        mt_log!(
            Level::Info,
            "Saved augmented dataset {} with multiplier {multiplier} to {}.",
            dataset.name(),
            output_path.display()
        );
    }

    Ok(scaled_names)
}

/// Compute the memory cost of a dataset in GB.
fn memory_cost(cardinality: usize, dimensionality: usize) -> f32 {
    let gb = 1024.0 * 1024.0 * 1024.0;
    let c = cardinality.as_f32() / gb;
    let d = dimensionality.as_f32();
    let f = std::mem::size_of::<f32>().as_f32();
    c * d * f
}

/// Save a vector of vectors to a numpy `.npy` file.
///
/// # Arguments
///
/// * `data` - The data to save.
/// * `path` - The path to the file to save.
///
/// # Errors
///
/// * If the file cannot be saved.
fn save_npy(data: Vec<Vec<f32>>, path: &Path) -> Result<(), String> {
    let data = ndarray::Array2::from_shape_vec(
        (data.len(), data[0].len()),
        data.into_iter().flatten().collect(),
    )
    .map_err(|e| e.to_string())?;

    ndarray_npy::write_npy(path, &data).map_err(|e| e.to_string())?;

    Ok(())
}
