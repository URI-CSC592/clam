//! Data sets for the ANN experiments.

/// The data sets to use for the experiments.
///
/// TODO: Add functions to read from hdf5 files.
pub enum AnnDatasets {
    /// The deep-image data set.
    DeepImage(usize),
    /// The fashion-mnist data set.
    FashionMnist(usize),
    /// The gist data set.
    Gist(usize),
    /// The glove-25 data set.
    Glove25(usize),
    /// The glove-50 data set.
    Glove50(usize),
    /// The glove-100 data set.
    Glove100(usize),
    /// The glove-200 data set.
    Glove200(usize),
    /// The mnist data set.
    Mnist(usize),
    /// The sift data set.
    Sift(usize),
    /// The lastfm data set.
    LastFm(usize),
    /// The NYTimes data set.
    NYTimes(usize),
    /// The kosarak data set.
    Kosarak(usize),
    /// The MovieLens10m data set.
    MovieLens10m(usize),
    /// A random data set. (cardinality, dimensionality, metric_name)
    Random(usize, usize, String),
}

impl AnnDatasets {
    /// Return the data set corresponding to the given string.
    pub fn from_str(s: &str) -> Result<Self, String> {
        let s = s.to_lowercase();

        if s.starts_with("random") {
            let parts = s.split('-').collect::<Vec<_>>();
            let cardinality = parts[1].parse::<usize>().map_err(|reason| {
                format!("Failed to parse dimensionality from random dataset: {s} ({reason})")
            })?;
            let dimensionality = parts[2].parse::<usize>().map_err(|reason| {
                format!("Failed to parse dimensionality from random dataset: {s} ({reason})")
            })?;
            let metric_name = parts[3].to_string();
            let dataset = Self::Random(cardinality, dimensionality, metric_name);
            let _ = dataset.metric()?;
            Ok(dataset)
        } else {
            let (dataset, scale) = if s.contains("scale") {
                let parts = s.split("-scale-").collect::<Vec<_>>();
                let scale = parts[1].parse::<usize>().map_err(|reason| {
                    format!("Failed to parse scale from scaled dataset: {s} ({reason})")
                })?;
                (parts[0], scale)
            } else {
                (s.as_str(), 1)
            };
            match dataset {
                "deep-image" => Ok(Self::DeepImage(scale)),
                "fashion-mnist" => Ok(Self::FashionMnist(scale)),
                "gist" => Ok(Self::Gist(scale)),
                "glove-25" => Ok(Self::Glove25(scale)),
                "glove-50" => Ok(Self::Glove50(scale)),
                "glove-100" => Ok(Self::Glove100(scale)),
                "glove-200" => Ok(Self::Glove200(scale)),
                "mnist" => Ok(Self::Mnist(scale)),
                "sift" => Ok(Self::Sift(scale)),
                "lastfm" => Ok(Self::LastFm(scale)),
                "nytimes" => Ok(Self::NYTimes(scale)),
                "kosarak" => Ok(Self::Kosarak(scale)),
                "movielens10m" => Ok(Self::MovieLens10m(scale)),
                _ => Err(format!("Unknown dataset: {s}")),
            }
        }
    }

    /// Return the cardinality multiplier for the dataset.
    #[allow(clippy::match_same_arms)]
    const fn scale(&self) -> usize {
        match self {
            Self::DeepImage(scale) => *scale,
            Self::FashionMnist(scale) => *scale,
            Self::Gist(scale) => *scale,
            Self::Glove25(scale) => *scale,
            Self::Glove50(scale) => *scale,
            Self::Glove100(scale) => *scale,
            Self::Glove200(scale) => *scale,
            Self::Mnist(scale) => *scale,
            Self::Sift(scale) => *scale,
            Self::LastFm(scale) => *scale,
            Self::NYTimes(scale) => *scale,
            Self::Kosarak(scale) => *scale,
            Self::MovieLens10m(scale) => *scale,
            Self::Random(..) => 1,
        }
    }

    /// Return the base name of the data set.
    pub fn base_name(&self) -> String {
        match self {
            Self::DeepImage(_) => "deep-image".to_string(),
            Self::FashionMnist(_) => "fashion-mnist".to_string(),
            Self::Gist(_) => "gist".to_string(),
            Self::Glove25(_) => "glove-25".to_string(),
            Self::Glove50(_) => "glove-50".to_string(),
            Self::Glove100(_) => "glove-100".to_string(),
            Self::Glove200(_) => "glove-200".to_string(),
            Self::Mnist(_) => "mnist".to_string(),
            Self::Sift(_) => "sift".to_string(),
            Self::LastFm(_) => "lastfm".to_string(),
            Self::NYTimes(_) => "nytimes".to_string(),
            Self::Kosarak(_) => "kosarak".to_string(),
            Self::MovieLens10m(_) => "movielens10m".to_string(),
            Self::Random(c, d, _) => format!("random-{c}-{d}"),
        }
    }

    /// Return the name of the data set including the multiplier, if any.
    pub fn name(&self) -> String {
        let base_name = self.base_name();

        if let Self::Random(..) = self {
            base_name
        } else {
            let scale = self.scale();
            if scale > 1 {
                format!("{base_name}-scale-{scale}")
            } else {
                base_name
            }
        }
    }

    /// Return the metric to use for this data set.
    pub fn metric_name(&self) -> &str {
        match self {
            Self::DeepImage(_)
            | Self::Glove25(_)
            | Self::Glove50(_)
            | Self::Glove100(_)
            | Self::Glove200(_)
            | Self::LastFm(_)
            | Self::NYTimes(_) => "cosine",
            Self::FashionMnist(_) | Self::Gist(_) | Self::Mnist(_) | Self::Sift(_) => "euclidean",
            Self::Kosarak(_) | Self::MovieLens10m(_) => "jaccard",
            Self::Random(_, _, metric_name) => metric_name,
        }
    }

    /// Return the metric to use for this data set.
    #[allow(clippy::type_complexity)]
    pub fn metric(&self) -> Result<fn(&Vec<f32>, &Vec<f32>) -> f32, String> {
        match self.metric_name() {
            "cosine" => Ok(cosine),
            "euclidean" => Ok(euclidean),
            "jaccard" => Err(
                "We are still merging Jaccard distance. Generic distances are hard.".to_string(),
            ),
            _ => Err(format!("Unknown metric: {}", self.metric_name())),
        }
    }

    /// Read the data set from the given directory.
    pub fn read(&self, dir: &std::path::Path) -> Result<[Vec<Vec<f32>>; 2], String> {
        let [train_data, test_data] = if let Self::Random(c, d, _) = self {
            let cardinality = *c;
            let dimensionality = *d;
            let (min_val, max_val) = (-1.0, 1.0);
            let seed = 42;

            let train_data = symagen::random_data::random_tabular_seedable(
                cardinality,
                dimensionality,
                min_val,
                max_val,
                seed,
            );
            let test_data = symagen::random_data::random_tabular_seedable(
                10_000,
                dimensionality,
                min_val,
                max_val,
                seed + 1,
            );

            [train_data, test_data]
        } else {
            let train_path = dir.join(format!("{}-train.npy", self.name()));
            let train_data = Self::read_npy(&train_path)?;

            let test_path = dir.join(format!("{}-test.npy", self.name()));
            let test_data = Self::read_npy(&test_path)?;

            [train_data, test_data]
        };

        Ok([train_data, test_data])
    }

    /// Read a numpy file into a vector of vectors.
    fn read_npy(path: &std::path::Path) -> Result<Vec<Vec<f32>>, String> {
        let data: ndarray::Array2<f32> = ndarray_npy::read_npy(path).map_err(|error| {
            format!(
                "Error: Failed to read your dataset at {}. {}",
                path.display(),
                error
            )
        })?;

        Ok(data.outer_iter().map(|row| row.to_vec()).collect())
    }

    /// The link from which to download the data set.
    fn download_link<'a>(&self) -> Result<&'a str, String> {
        match self {
            Self::DeepImage(_) => Ok("http://ann-benchmarks.com/deep-image-96-angular.hdf5"),
            Self::FashionMnist(_) => {
                Ok("http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5")
            }
            Self::Gist(_) => Ok("http://ann-benchmarks.com/gist-960-euclidean.hdf5"),
            Self::Glove25(_) => Ok("http://ann-benchmarks.com/glove-25-angular.hdf5"),
            Self::Glove50(_) => Ok("http://ann-benchmarks.com/glove-50-angular.hdf5"),
            Self::Glove100(_) => Ok("http://ann-benchmarks.com/glove-100-angular.hdf5"),
            Self::Glove200(_) => Ok("http://ann-benchmarks.com/glove-200-angular.hdf5"),
            Self::Kosarak(_) => Ok("http://ann-benchmarks.com/kosarak-jaccard.hdf5"),
            Self::Mnist(_) => Ok("http://ann-benchmarks.com/mnist-784-euclidean.hdf5"),
            Self::MovieLens10m(_) => Ok("http://ann-benchmarks.com/movielens10m-jaccard.hdf5"),
            Self::NYTimes(_) => Ok("http://ann-benchmarks.com/nytimes-256-angular.hdf5"),
            Self::Sift(_) => Ok("http://ann-benchmarks.com/sift-128-euclidean.hdf5"),
            Self::LastFm(_) => Ok("http://ann-benchmarks.com/lastfm-64-dot.hdf5"),
            Self::Random(..) => Err("Random datasets cannot be downloaded.".to_string()),
        }
    }

    /// The name of the hdf5 file.
    #[allow(dead_code)]
    fn hdf5_name<'a>(&self) -> Result<&'a str, String> {
        self.download_link()
            .map(|link| link.split(".com/").collect::<Vec<_>>()[1])
    }

    /// Read the data set from the given directory.
    #[allow(dead_code, unused_variables)]
    pub fn read_hdf5(&self, dir: &std::path::Path) -> Result<[Vec<Vec<f32>>; 2], String> {
        let data_path = dir.join(self.hdf5_name()?);
        // let file = hdf5::File::open(data_path).map_err(|error| {
        //     format!(
        //         "Error: Failed to read your dataset at {}. {}",
        //         data_path.display(),
        //         error
        //     )
        // })?;

        todo!()
    }
}

/// A wrapper around the cosine distance function.
#[allow(clippy::ptr_arg)]
fn cosine(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
    distances::simd::cosine_f32(x, y)
}

/// A wrapper around the euclidean distance function.
#[allow(clippy::ptr_arg)]
fn euclidean(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
    distances::simd::euclidean_f32(x, y)
}
