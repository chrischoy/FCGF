#!/usr/bin/env bash

DATA_DIR=$1
REPO_ID="chrischoy/FCGF-3DMatch"

function download() {
    echo "#################################"
    echo "Data Root Dir: ${DATA_DIR}"
    echo "Dataset repo : ${REPO_ID} (Hugging Face)"
    echo "#################################"

    if [ ! -d "$DATA_DIR" ]; then
        echo ">> Create data directory: ${DATA_DIR}"
        mkdir -p "$DATA_DIR"
    fi

    echo ">> Downloading the preprocessed 3DMatch dataset from Hugging Face"
    echo ">> (requires huggingface_hub: pip install huggingface_hub)"
    python -c "from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='${REPO_ID}', repo_type='dataset', local_dir='${DATA_DIR}')"

    echo "#################################"
    echo "Done! Data is in ${DATA_DIR}/threedmatch"
    echo "#################################"
}

function main() {
    echo $DATA_DIR
    if [ -z "$DATA_DIR" ]; then
        echo "DATA_DIR is required config!"
    else
        download
    fi
}

main;
