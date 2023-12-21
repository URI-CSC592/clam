pub(crate) mod utils;
use abd_clam::{knn, rnn, Cakes, PartitionCriteria, VecDataset};

#[allow(clippy::ptr_arg)]
fn euclidean(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
    distances::simd::euclidean_f32(x, y)
}

fn main() {
    let seed = 69;
    let (cardinality, dimensionality) = (100, 10);
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

    // We will use the origin as our query.
    let query: Vec<f32> = vec![0.0; dimensionality];

    // RNN search will use a radius of 0.05.
    let radius: f32 = 0.5;

    // KNN search will find the 10 nearest neighbors.
    let k = 10;

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

    let entropy_criterion = utils::MinShannonEntropy {
        threshold: 0.3,
        // TODO: Remove this clone
        data: data.clone(),
    };

    // Create Partition Criteria
    let criteria = PartitionCriteria::default().with_custom(
        Box::new(entropy_criterion)
    );

    // The Cakes struct provides the functionality described in the CHESS paper.
    // We use a single shard here because the demo data is small.
    let model = Cakes::new(data, Some(seed), &criteria);
    // This line performs a non-trivial amount of work. #understatement


    // At this point, the dataset has been reordered to improve search performance.

    // We can now perform RNN search on the model.
    let rnn_results: Vec<(usize, f32)> =
        model.rnn_search(&query, radius, rnn::Algorithm::default());

    // We can also perform KNN search on the model.
    let knn_results: Vec<(usize, f32)> = model.knn_search(&query, k, knn::Algorithm::default());

    // Both results are a Vec of 2-tuples where the first element is the index of
    // the point in the dataset and the second element is the distance from the
    // query point.

    // We can get the reordered metadata from the model.
    let metadata: &[bool] = model.shards()[0].metadata().unwrap();

    // Print how many trues and falses there are in the metadata
    println!(
        "Metadata: {:?} trues, {:?} falses",
        metadata.iter().filter(|&&x| x).count(),
        metadata.iter().filter(|&&x| !x).count()
    );

    // We can use the results to get the labels of the points that are within the
    // radius of the query point.
    let rnn_labels: Vec<bool> = rnn_results.iter().map(|&(i, _)| metadata[i]).collect();

    // We can use the results to get the labels of the points that are the k nearest
    // neighbors of the query point.
    let knn_labels: Vec<bool> = knn_results.iter().map(|&(i, _)| metadata[i]).collect();

    // Print the rnn labels
    println!("RNN labels: {:?}", rnn_labels);

    // Print the knn labels
    println!("KNN labels: {:?}", knn_labels);
}
