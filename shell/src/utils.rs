use std::ops::Neg;

/// Calculate the Shannon entropy for a given array of probabilities
///
/// The Shannon entropy is calculated as the negative sum of the probabilities
/// multiplied by their logarithms. The base of the logarithm for this
/// implementation is 2.
///
/// # Arguments
///
/// - `probabilities`: A reference to an array of probabilities
///
/// # Returns
///
/// The Shannon entropy for the given probabilities
///
#[allow(dead_code)]
fn shannon_entropy(probabilities: &[f64]) -> f64 {
    // Handle the base case where there are no probabilities
    if probabilities.is_empty() {
        return f64::NAN;
    }

    // Otherwise, calculate the Shannon entropy which is the negative sum of the probabilities
    // multiplied by their logarithms
    probabilities
        .iter()
        .map(|&p| if p > 0.0 { p * p.log2() } else { 0.0 }) // If statement handles the case where p := 0
        .sum::<f64>()
        .neg()
}

/// Calculate the Gini impurity for a given array of probabilities
///
/// The Gini impurity value is calculated as 1 - the sum of the probabilities squared.
///
/// # Arguments
///
/// - `probabilities`: A reference to an array of probabilities
///
/// # Returns
///
/// The Gini impurity for the given probabilities
///
#[allow(dead_code)]
fn gini_impurity(probabilities: &[f64]) -> f64 {
    // Handle the base case where there are no probabilities
    if probabilities.is_empty() {
        return f64::NAN;
    }

    // Otherwise, calculate the Gini impurity which is 1 - the sum of the probabilities squared
    1.0 - probabilities.iter().map(|&p| p * p).sum::<f64>()
}

/// Count occurrences of each value in a vector
/// # Arguments
///
/// - `values`: A reference to a vector of values
///
/// # Returns
///
/// A vector of tuples containing the value and the number of occurrences
///
#[allow(dead_code)]
fn count_occurrences<T: Eq + Copy>(values: &[T]) -> Vec<(T, usize)> {
    values.iter().fold(Vec::new(), |mut acc, &v| {
        if let Some((_, count)) = acc.iter_mut().find(|(v2, _)| *v2 == v) {
            *count += 1;
        } else {
            acc.push((v, 1));
        }
        acc
    })
}

/// Calculate the Shannon entropy of a given vector of metadata
///
/// The Shannon entropy is calculated as the negative sum of the probabilities
/// multiplied by their logarithms. The base of the logarithm for this
/// implementation is 2.
///
/// We are calculating the entropy on a vector of metadata, which are of
/// a generic type.
///
/// # Arguments
///
/// - `metadata`: A reference to a vector of metadata
///
/// # Returns
///
/// The Shannon entropy for the given metadata
///
#[allow(dead_code)]
#[allow(clippy::cast_precision_loss)]
pub fn shannon_entropy_from_metadata<T: Eq + Copy>(metadata: &[T]) -> f64 {
    // Put the metadata values and counts into an array of tuples
    let counts = count_occurrences(metadata);

    // Get the probabilities from the counts and calculate the Shannon entropy
    counts
        .iter()
        .fold(0.0, |acc, (_, count)| {
            let probability = *count as f64 / metadata.len() as f64;
            probability.mul_add(probability.log2(), acc)
        })
        .neg()
}

/// Calculate the Gini impurity of a given vector of metadata
///
/// The Gini impurity value is calculated as 1 - the sum of the probabilities squared.
///
/// We are calculating the impurity on a vector of metadata, which are of
/// a generic type.
///
/// # Arguments
///
/// - `metadata`: A reference to a vector of metadata
///
/// # Returns
///
/// The Gini impurity for the given metadata
///
#[allow(dead_code)]
#[allow(clippy::cast_precision_loss)]
pub fn gini_impurity_from_metadata<T: Eq + Copy>(metadata: &[T]) -> f64 {
    // Put the metadata values and counts into an array of tuples
    let counts = count_occurrences(metadata);

    // Get the probabilities from the counts and calculate the Gini impurity
    1.0 - counts.iter().fold(0.0, |acc, (_, count)| {
        let probability = *count as f64 / metadata.len() as f64;
        probability.mul_add(probability, acc)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::EPSILON;

    #[test]
    fn test_shannon_entropy() {
        // Test data: (probabilities, expected entropy)
        // Expected entropy calculated using WolframAlpha
        let test_cases = vec![
            (vec![1.0], 0.0),      // Base case. One event with probability 1.0
            (vec![0.5, 0.5], 1.0), // Base case. Two events with equal probabilities
            (vec![0.0, 1.0], 0.0), // Similar to the first base case, but with a zero probability event
            (vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], 1.5849625007211561),
            (vec![0.25, 0.25, 0.25, 0.25], 2.0),
            (vec![1.0, 1.0], 0.0), // Probabilities don't add to 1.0
        ];

        test_cases.iter().for_each(|(probabilities, expected)| {
            let calculated = shannon_entropy(&probabilities);
            assert!(
                float_cmp::approx_eq!(f64, calculated, *expected, epsilon = EPSILON),
                "{}, {} not equal for probabilities {:?}",
                calculated,
                expected,
                probabilities
            );
        });

        // Check the case where there are no probabilities
        let entropy = shannon_entropy(&[]);
        assert!(entropy.is_nan());
    }

    #[test]
    fn test_gini_impurity() {
        // Test data: (probabilities, expected gini impurity)
        // Expected Gini impurity calculated using WolframAlpha
        let test_cases = vec![
            (vec![1.0], 0.0),      // Base case. One event with probability 1.0
            (vec![0.5, 0.5], 0.5), // Base case. Two events with equal probabilities
            (vec![0.0, 1.0], 0.0), // Similar to the first base case, but with a zero probability event
            (vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], 2.0 / 3.0),
            (vec![0.25, 0.25, 0.25, 0.25], 0.75),
            (vec![1.0, 1.0], -1.0), // Probabilities don't add to 1.0
        ];

        test_cases.iter().for_each(|(probabilities, expected)| {
            let calculated = gini_impurity(&probabilities);
            assert!(
                float_cmp::approx_eq!(f64, calculated, *expected, ulps = 2),
                "{}, {} not equal for probabilities {:?}",
                calculated,
                expected,
                probabilities
            );
        });

        // Check the case where there are no probabilities
        // Technically per the equation this should be 1.  But for our
        // purposes, we return NaN
        let gini_impurity = gini_impurity(&[]);
        assert!(gini_impurity.is_nan());
    }

    #[test]
    fn test_shannon_entropy_from_metadata() {
        // Integer test cases
        vec![
            (vec![0], shannon_entropy(&[1.0])),
            (vec![0, 0], shannon_entropy(&[1.0])),
            (vec![0, 1], shannon_entropy(&[0.5, 0.5])),
            (vec![0, 1, 0, 1], shannon_entropy(&[0.5, 0.5])),
            (
                vec![0, 1, 2],
                shannon_entropy(&[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
            ),
            (vec![0, 1, 2, 3], shannon_entropy(&[0.25, 0.25, 0.25, 0.25])),
        ]
        .iter()
        .for_each(|(metadata, expected)| {
            let calculated = shannon_entropy_from_metadata(&metadata);
            assert!(
                float_cmp::approx_eq!(f64, calculated, *expected, epsilon = EPSILON),
                "{}, {} not equal for metadata {:?}",
                calculated,
                expected,
                metadata
            );
        });

        // String test cases
        vec![
            (vec!["a"], shannon_entropy(&[1.0])),
            (vec!["a", "a"], shannon_entropy(&[1.0])),
            (vec!["a", "b"], shannon_entropy(&[0.5, 0.5])),
            (vec!["a", "b", "a", "b"], shannon_entropy(&[0.5, 0.5])),
            (
                vec!["a", "b", "c"],
                shannon_entropy(&[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
            ),
            (
                vec!["a", "b", "c", "d"],
                shannon_entropy(&[0.25, 0.25, 0.25, 0.25]),
            ),
        ]
        .iter()
        .for_each(|(metadata, expected)| {
            let calculated = shannon_entropy_from_metadata(&metadata);
            assert!(
                float_cmp::approx_eq!(f64, calculated, *expected, epsilon = EPSILON),
                "{}, {} not equal for metadata {:?}",
                calculated,
                expected,
                metadata
            );
        });
    }

    #[test]
    fn test_gini_impurity_from_metadata() {
        // Integer test cases
        vec![
            (vec![0], gini_impurity(&[1.0])),
            (vec![0, 0], gini_impurity(&[1.0])),
            (vec![0, 1], gini_impurity(&[0.5, 0.5])),
            (vec![0, 1, 0, 1], gini_impurity(&[0.5, 0.5])),
            (
                vec![0, 1, 2],
                gini_impurity(&[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
            ),
            (vec![0, 1, 2, 3], gini_impurity(&[0.25, 0.25, 0.25, 0.25])),
        ]
        .iter()
        .for_each(|(metadata, expected)| {
            let calculated = gini_impurity_from_metadata(&metadata);
            assert!(
                float_cmp::approx_eq!(f64, calculated, *expected, ulps = 2),
                "{}, {} not equal for metadata {:?}",
                calculated,
                expected,
                metadata
            );
        });

        // String test cases
        vec![
            (vec!["a"], gini_impurity(&[1.0])),
            (vec!["a", "a"], gini_impurity(&[1.0])),
            (vec!["a", "b"], gini_impurity(&[0.5, 0.5])),
            (vec!["a", "b", "a", "b"], gini_impurity(&[0.5, 0.5])),
            (
                vec!["a", "b", "c"],
                gini_impurity(&[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
            ),
            (
                vec!["a", "b", "c", "d"],
                gini_impurity(&[0.25, 0.25, 0.25, 0.25]),
            ),
        ]
        .iter()
        .for_each(|(metadata, expected)| {
            let calculated = gini_impurity_from_metadata(&metadata);
            assert!(
                float_cmp::approx_eq!(f64, calculated, *expected, ulps = 2),
                "{}, {} not equal for metadata {:?}",
                calculated,
                expected,
                metadata
            );
        });
    }

    #[test]
    fn test_count_occurrences() {
        let test_cases = vec![
            (vec![], vec![]),
            (vec![0], vec![(0, 1)]),
            (vec![0, 0], vec![(0, 2)]),
            (vec![0, 1], vec![(0, 1), (1, 1)]),
            (vec![0, 1, 0, 1], vec![(0, 2), (1, 2)]),
            (vec![0, 1, 2], vec![(0, 1), (1, 1), (2, 1)]),
            (vec![0, 1, 2, 3], vec![(0, 1), (1, 1), (2, 1), (3, 1)]),
        ];

        test_cases.iter().for_each(|(values, expected)| {
            let calculated = count_occurrences(&values);
            assert_eq!(calculated, *expected);
        });
    }
}
