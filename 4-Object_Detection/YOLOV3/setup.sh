#!/bin/bash
set -ex

PYTHON="/usr/local/bin/python3.10"

if [[ ! -f "./data/dataset/yymnist_train.txt" ]]; then
    $PYTHON yymnist/make_data.py --images_num 1000 --images_path ./data/dataset/train --labels_txt ./data/dataset/yymnist_train.txt
fi

if [[ ! -f "./data/dataset/yymnist_test.txt" ]]; then
    $PYTHON yymnist/make_data.py --images_num 200  --images_path ./data/dataset/test --labels_txt ./data/dataset/yymnist_test.txt
fi
