#!/bin/bash
set -ex

PYTHON="/usr/local/bin/python3.10"

if [[ ! -f "./synthetic_dataset/imageAno/1636.txt" ]]; then
    curl -O https://cs.hunter.cuny.edu/~khatchad/media/synthetic_dataset.zip
    unzip synthetic_dataset.zip
fi
