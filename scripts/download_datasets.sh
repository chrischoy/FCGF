#!/usr/bin/env bash

DATA_DIR=$1

function download() {
    TMP_PATH="$DATA_DIR/tmp"
    echo "#################################"
    echo "Data Root Dir: ${DATA_DIR}"
    echo "Download Path: ${TMP_PATH}"
    echo "#################################"
    urls=(
        'http://node2.chrischoy.org/data/datasets/registration/threedmatch.tgz'
    )

    if [ ! -d "$TMP_PATH" ]; then
        echo ">> Create temporary directory: ${TMP_PATH}"
        mkdir -p "$TMP_PATH"
    fi
    cd "$TMP_PATH"

    echo ">> Start downloading"
    echo ${urls[@]} | xargs -n 1 -P 3 wget --no-check-certificate -q -c --show-progress $0 

    echo ">> Unpack .zip file"
    for filename in *.tgz
    do
        tar -xvzf $filename -C ../
    done

    echo ">> Clear tmp directory"
    cd ..
    rm -rf ./tmp

    echo "#################################"
    echo "Done!"
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
