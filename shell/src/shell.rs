pub(crate) mod utils;
use abd_clam::{Cakes, PartitionCriteria, VecDataset};

#[allow(clippy::ptr_arg)]
fn euclidean(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
    distances::simd::euclidean_f32(x, y)
}

fn main() {
    let seed = 42;
    let (cardinality, dimensionality) = (1_000, 10);
    let (min_val, max_val) = (-0.5, 0.5);
    let data: Vec<Vec<f32>> = symagen::random_data::random_tabular_seedable(
        cardinality,
        dimensionality,
        min_val,
        max_val,
        seed,
    );

    // We will generate some random labels for each point.
    let metadata: Vec<bool> = data.iter().map(|v| v[0] > 0.0).collect();

    // The name of the dataset.
    let name = "demo".to_string();

    // We will assume that our distance function is cheap to compute.
    let is_metric_expensive = false;

    // The metric function itself will be given to Cakes.
    let data = VecDataset::<Vec<f32>, f32, bool>::new(
        name,
        data,
        euclidean,
        is_metric_expensive,
        Some(metadata),
    );

    // Create Partition Criteria
    let criteria = PartitionCriteria::default();

    // The Cakes struct provides the functionality described in the CHESS paper.
    // We use a single shard here because the demo data is small.
    let _model = Cakes::new(data, Some(seed), &criteria);
    // This line performs a non-trivial amount of work. #understatement
}
