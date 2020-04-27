export DATA_DIR=$1
export BASE_PATH=http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments

function download() {
    TMP_PATH="$DATA_DIR/tmp"
    echo "#################################"
    echo "Data Root Dir: ${DATA_DIR}"
    echo "Download Path: ${TMP_PATH}"
    echo "#################################"
    urls=(
	'7-scenes-redkitchen'
	'7-scenes-redkitchen-evaluation'
	'sun3d-home_at-home_at_scan1_2013_jan_1'
	'sun3d-home_at-home_at_scan1_2013_jan_1-evaluation'
	'sun3d-home_md-home_md_scan9_2012_sep_30'
	'sun3d-home_md-home_md_scan9_2012_sep_30-evaluation'
	'sun3d-hotel_uc-scan3'
	'sun3d-hotel_uc-scan3-evaluation'
	'sun3d-hotel_umd-maryland_hotel1'
	'sun3d-hotel_umd-maryland_hotel1-evaluation'
	'sun3d-hotel_umd-maryland_hotel3'
	'sun3d-hotel_umd-maryland_hotel3-evaluation'
	'sun3d-mit_76_studyroom-76-1studyroom2'
	'sun3d-mit_76_studyroom-76-1studyroom2-evaluation'
	'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
	'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika-evaluation'
    )

    if [ ! -d "$TMP_PATH" ]; then
        echo ">> Create temporary directory: ${TMP_PATH}"
        mkdir -p "$TMP_PATH"
    fi
    cd "$TMP_PATH"

    echo ">> Start downloading"
    for url in ${urls[@]}
    do
    	echo $BASE_PATH/$url
	wget --no-check-certificate --show-progress $BASE_PATH/$url.zip ./
    done

    echo ">> Unpack .zip file"
    for filename in *.zip
    do
        unzip $filename
    done

    rm *.zip
    mv * ../

    rm -r tmp
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
