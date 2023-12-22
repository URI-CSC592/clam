#!/usr/bin/env bash

cargo build --release --bin shell

for dataset in "fashion-mnist"
do
    ./target/release/shell \
        --input-dir "../data/ann-benchmarks/datasets" \
        --output-dir "../data/shell-reports" \
        --dataset $dataset \
        --ks 10 100 \
        --num-trials 1
done
