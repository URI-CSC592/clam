"""Parses the SILVA-18S sequences from a FASTA file into a plain text file."""

import collections
import logging
import pathlib

from fasta_reader import read_fasta
import typer

logger = logging.getLogger("parse_silva")
logger.setLevel("INFO")

app = typer.Typer()


@app.command()
def parse(
    fasta_file: pathlib.Path = typer.Option(
        ...,
        "--fasta-file",
        "-i",
        help="The FASTA file containing the SILVA-18S sequences.",
        exists=True,
        readable=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    output_dir: pathlib.Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="The directory where the output files will be saved.",
        exists=True,
        readable=True,
        writable=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    progress_interval: int = typer.Option(
        1_000,
        "--progress-interval",
        "-p",
        help="The number of sequences to parse before logging a progress update.",
    ),
) -> None:
    """Parses the SILVA-18S sequences from a FASTA file.

    The FASTA file is expected to contain the SILVA-18S sequences. The sequences
    are parsed into four files:

    1. A plain text file containing the sequences as they appear in the FASTA
    file. The sequences are separated by newlines.

    2. A plain text file containing the sequences with all gaps removed. The
    sequences are separated by newlines.

    3. A plain text file containing the headers of the sequences. The headers
    are separated by newlines.

    4. A plain text file containing the alphabet and the number of times each
    character appears in the sequences. The first line contains the alphabet
    in sorted order. The remaining lines contain the character and the number
    of times it appears in the sequences.
    """
    logger.info(f"parsing: {fasta_file}")

    stem = fasta_file.stem
    pre_aligned_path = output_dir.joinpath(f"{stem}-pre-aligned.txt")
    unaligned_path = output_dir.joinpath(f"{stem}-unaligned.txt")
    headers_path = output_dir.joinpath(f"{stem}-headers.txt")

    alphabet: dict[str, int] = collections.defaultdict(int)
    num_sequences = 0

    with (
        pre_aligned_path.open("w") as pre_aligned_file,
        unaligned_path.open("w") as unaligned_file,
        headers_path.open("w") as headers_file,
    ):
        for item in read_fasta(str(fasta_file)):
            header = item.defline
            headers_file.write(header + "\n")

            sequence = item.sequence
            pre_aligned_file.write(sequence + "\n")

            sequence = sequence.replace("-", "").replace(".", "")
            unaligned_file.write(sequence + "\n")

            alphabet_counts = collections.Counter(sequence)
            for character, count in alphabet_counts.items():
                alphabet[character] += count

            num_sequences += 1
            if num_sequences % progress_interval == 0:
                logger.info(f"parsed {num_sequences} sequences ...")

    logger.info(f"parsed {num_sequences} sequences")

    alphabet_path = output_dir.joinpath(f"{stem}-alphabet.txt")

    with alphabet_path.open("w") as alphabet_file:
        characters = list(sorted(alphabet.keys()))
        alphabet_file.write("".join(characters) + "\n")

        for character in characters:
            alphabet_file.write(f"{character} {alphabet[character]}\n")
