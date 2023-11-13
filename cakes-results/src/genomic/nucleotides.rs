//! Nucleotides for Silva-18S sequences.

/// A nucleotide for Silva-18S sequences.
///
/// These are as defined in the [IUPAC nucleotide code](https://www.bioinformatics.org/sms/iupac.html).
#[derive(Debug, Clone, Copy)]
pub enum Nucleotide {
    /// Adenine
    A,
    /// Cytosine
    C,
    /// Guanine
    G,
    /// Thymine (or Uracil)
    U,
    /// A or G
    R,
    /// C or T
    Y,
    /// G or C
    S,
    /// A or T
    W,
    /// G or T
    K,
    /// A or C
    M,
    /// C or G or T
    B,
    /// A or G or T
    D,
    /// A or C or T
    H,
    /// A or C or G
    V,
    /// Any base
    N,
    /// Gap
    Gap,
}

impl Nucleotide {
    /// Returns the nucleotide corresponding to the given character.
    ///
    /// # Arguments
    ///
    /// * `c` - The character to convert.
    ///
    /// # Errors
    ///
    /// If the given character is not a valid nucleotide.
    pub fn from_char(c: char) -> Result<Self, String> {
        match c {
            'A' => Ok(Self::A),
            'C' => Ok(Self::C),
            'G' => Ok(Self::G),
            'U' | 'T' => Ok(Self::U),
            'R' => Ok(Self::R),
            'Y' => Ok(Self::Y),
            'S' => Ok(Self::S),
            'W' => Ok(Self::W),
            'K' => Ok(Self::K),
            'M' => Ok(Self::M),
            'B' => Ok(Self::B),
            'D' => Ok(Self::D),
            'H' => Ok(Self::H),
            'V' => Ok(Self::V),
            'N' => Ok(Self::N),
            '-' | '.' => Ok(Self::Gap),
            _ => Err(format!("Invalid nucleotide: {c}")),
        }
    }

    /// Returns the character corresponding to the given nucleotide.
    pub const fn to_char(self) -> char {
        match self {
            Self::A => 'A',
            Self::C => 'C',
            Self::G => 'G',
            Self::U => 'U',
            Self::R => 'R',
            Self::Y => 'Y',
            Self::S => 'S',
            Self::W => 'W',
            Self::K => 'K',
            Self::M => 'M',
            Self::B => 'B',
            Self::D => 'D',
            Self::H => 'H',
            Self::V => 'V',
            Self::N => 'N',
            Self::Gap => '-',
        }
    }

    /// Returns the byte corresponding to the given nucleotide.
    pub const fn to_byte(self) -> u8 {
        self.to_char() as u8
    }

    /// Returns the nucleotide corresponding to the given byte.
    pub fn from_byte(b: u8) -> Result<Self, String> {
        Self::from_char(b as char)
    }

    /// Returns the gap penalty.
    pub const fn gap_penalty() -> u32 {
        180
    }

    /// Returns the penalty for aligning the given nucleotides.
    ///
    /// # Arguments
    ///
    /// * `other` - The other nucleotide.
    #[allow(clippy::too_many_lines)]
    pub const fn penalty(self, other: Self) -> u32 {
        match self {
            Self::A => match other {
                Self::A => 0,
                Self::R | Self::W | Self::M => 90,
                Self::D | Self::H | Self::V => 120,
                Self::N => 135,
                _ => Self::gap_penalty(),
            },
            Self::C => match other {
                Self::C => 0,
                Self::R | Self::S | Self::K => 90,
                Self::B | Self::D | Self::H => 120,
                Self::N => 135,
                _ => Self::gap_penalty(),
            },
            Self::G => match other {
                Self::G => 0,
                Self::R | Self::S | Self::K => 90,
                Self::B | Self::D | Self::V => 120,
                Self::N => 135,
                _ => Self::gap_penalty(),
            },
            Self::U => match other {
                Self::U => 0,
                Self::Y | Self::W | Self::K => 90,
                Self::B | Self::D | Self::H => 120,
                Self::N => 135,
                _ => Self::gap_penalty(),
            },
            Self::R => match other {
                Self::A | Self::G | Self::R => 90,
                Self::D | Self::V => 120,
                Self::S | Self::W | Self::K | Self::M | Self::N => 135,
                Self::B | Self::H => 150,
                _ => Self::gap_penalty(),
            },
            Self::Y => match other {
                Self::C | Self::U | Self::Y => 90,
                Self::B | Self::H => 120,
                Self::S | Self::W | Self::K | Self::M | Self::N => 135,
                Self::D | Self::V => 150,
                _ => Self::gap_penalty(),
            },
            Self::S => match other {
                Self::C | Self::G | Self::S => 90,
                Self::B | Self::V => 120,
                Self::R | Self::W | Self::K | Self::M | Self::N => 135,
                Self::D | Self::H => 150,
                _ => Self::gap_penalty(),
            },
            Self::W => match other {
                Self::A | Self::U | Self::W => 90,
                Self::D | Self::H => 120,
                Self::R | Self::Y | Self::K | Self::M | Self::N => 135,
                Self::B | Self::V => 150,
                _ => Self::gap_penalty(),
            },
            Self::K => match other {
                Self::G | Self::U | Self::K => 90,
                Self::B | Self::D => 120,
                Self::R | Self::Y | Self::S | Self::W | Self::N => 135,
                Self::H | Self::V => 150,
                _ => Self::gap_penalty(),
            },
            Self::M => match other {
                Self::A | Self::C | Self::M => 90,
                Self::H | Self::V => 120,
                Self::R | Self::Y | Self::S | Self::W | Self::N => 135,
                Self::B | Self::D => 150,
                _ => Self::gap_penalty(),
            },
            Self::B => match other {
                Self::C | Self::G | Self::U | Self::Y | Self::S | Self::K | Self::B => 120,
                Self::N => 135,
                Self::D | Self::H | Self::V => 140,
                Self::R | Self::W | Self::M => 150,
                _ => Self::gap_penalty(),
            },
            Self::D => match other {
                Self::A | Self::G | Self::U | Self::R | Self::W | Self::K | Self::D => 120,
                Self::N => 135,
                Self::B | Self::H | Self::V => 140,
                Self::Y | Self::S | Self::M => 150,
                _ => Self::gap_penalty(),
            },
            Self::H => match other {
                Self::A | Self::C | Self::U | Self::Y | Self::W | Self::M | Self::H => 120,
                Self::N => 135,
                Self::B | Self::D | Self::V => 140,
                Self::R | Self::S | Self::K => 150,
                _ => Self::gap_penalty(),
            },
            Self::V => match other {
                Self::A | Self::C | Self::G | Self::R | Self::S | Self::M | Self::V => 120,
                Self::N => 135,
                Self::B | Self::D | Self::H => 140,
                Self::Y | Self::W | Self::K => 150,
                _ => Self::gap_penalty(),
            },
            Self::N => match other {
                Self::Gap => Self::gap_penalty(),
                _ => 135,
            },
            Self::Gap => match other {
                Self::Gap => 0,
                _ => Self::gap_penalty(),
            },
        }
    }
}
