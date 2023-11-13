//! Sequences of nucleotides for the Silva-18S dataset.

use core::fmt::Display;

use abd_clam::Instance;
use rand::distributions::uniform::UniformSampler;

use super::nucleotides::Nucleotide;

/// A genomic sequence is a list of nucleotides.
#[derive(Debug, Clone)]
pub struct Sequence(Vec<Nucleotide>);

impl Sequence {
    /// Parses the given string into a sequence.
    pub fn from_str(s: &str) -> Result<Self, String> {
        let seq = s
            .chars()
            .map(Nucleotide::from_char)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self(seq))
    }

    /// The number of nucleotides in the sequence.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns the underlying nucleotide list.
    pub fn neucloetides(&self) -> &[Nucleotide] {
        &self.0
    }
}

impl Display for Sequence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.0
                .iter()
                .map(|&b| Nucleotide::to_char(b))
                .collect::<String>()
        )
    }
}

impl Instance for Sequence {
    fn to_bytes(&self) -> Vec<u8> {
        self.0.iter().map(|&c| Nucleotide::to_byte(c)).collect()
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, String>
    where
        Self: Sized,
    {
        let seq = bytes
            .iter()
            .map(|b| Nucleotide::from_byte(*b))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self(seq))
    }

    fn type_name() -> String {
        "Sequence".to_string()
    }
}
