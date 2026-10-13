#!/usr/bin/env bash
set -euo pipefail

# 使用已校正反射率影像生成 M3M 四波段正射影像。
# 工作数据复制到 WSL 文件系统以提高处理速度，不修改 E: 盘源影像。

work=/root/m3m_ortho
raw_src=/mnt/e/Projects/DJi/M3M_WinterBarley_Dataset/M3M_WinterBarley_Dataset/90m_RGBMSP_RTK_Overlap7080/Dataset
reflectance_src=/mnt/e/Projects/DJi/M3M_Project/outputs/radiometric_full_approx
project=/root/m3m_ortho/odm_project_v2
dataset=winterbarley
images="$project/$dataset/images"
deliver=/mnt/e/Projects/DJi/M3M_Project/outputs/orthomosaic
manifest=/mnt/e/Projects/DJi/M3M_Project/outputs/radiometric_full_approx/radiometric_manifest.csv
tools_dir=/mnt/e/Projects/DJi/M3M_Project/scripts
repair_log=/mnt/e/Projects/DJi/M3M_Project/outputs/odm_processing_repair.log

mkdir -p "$work/raw" "$work/radiometric" "$images" "$deliver"
cp -n "$raw_src"/*_MS_*.TIF "$work/raw/"
cp -n "$reflectance_src"/*_REFLECTANCE_APPROX.tif "$work/radiometric/"

shopt -s nullglob
corrected=("$work/radiometric"/*_REFLECTANCE_APPROX.tif)
if [[ ${#corrected[@]} -ne 1236 ]]; then
    echo "Expected 1,236 corrected band frames, found ${#corrected[@]}" >&2
    exit 2
fi

count=0
if [[ ! -f "$work/.metadata_prepared" ]]; then
    for corrected_path in "${corrected[@]}"; do
        base=$(basename "$corrected_path" _REFLECTANCE_APPROX.tif)
        source_path="$work/raw/$base.TIF"
        if [[ ! -f "$source_path" ]]; then
            echo "Missing source TIFF for $corrected_path" >&2
            exit 3
        fi
        # 复制相机参数、RTK 位置与姿态、波段标识及拍摄元数据。
        # ExifTool 只写入元数据，不改变已校正的栅格像元值。
        exiftool -q -q -overwrite_original -TagsFromFile "$source_path" \
            -XMP -EXIF:GPS:All -EXIF:DateTimeOriginal -EXIF:DateTimeDigitized \
            -EXIF:ExposureTime -EXIF:FocalLength -EXIF:FNumber -EXIF:Make -EXIF:Model \
            "$corrected_path"

        band=${base##*_MS_}
        capture=${base%_MS_*}
        image_link="$images/${capture}_${band}.tif"
        if [[ ! -e "$image_link" ]]; then
            ln "$corrected_path" "$image_link"
        elif ! cmp -s "$corrected_path" "$image_link"; then
            echo "Existing ODM image differs: $image_link" >&2
            exit 4
        fi

        ((count += 1))
        if (( count % 100 == 0 )); then
            echo "XMP attached and linked: $count/${#corrected[@]}"
        fi
    done
    touch "$work/.metadata_prepared"
else
    echo "Reusing prepared XMP metadata and ODM hard links."
fi

image_count=$(find "$images" -maxdepth 1 -type f -name '*.tif' | wc -l)
if [[ "$image_count" -ne 1236 ]]; then
    echo "Expected 1,236 ODM image files, found $image_count" >&2
    exit 5
fi

if [[ ! -f "$work/.nodata_filled" ]]; then
    # Texrecon 会将 -9999 饱和占位值扩散成大块纹理异常。
    # 仅填补 WSL 工作副本中少量无效像元，保留其元数据。
    docker run --rm \
        --entrypoint /code/venv/bin/python \
        -v "$images:/images" \
        -v "$tools_dir:/tools:ro" \
        -v "$manifest:/manifest.csv:ro" \
        -v "$deliver:/audit" \
        opendronemap/odm:latest \
        /tools/prepare_odm_inputs.py \
        --images-dir /images \
        --manifest /manifest.csv \
        --audit-csv /audit/odm_input_filling_audit.csv
    touch "$work/.nodata_filled"
else
    echo "Reusing working copies with saturated pixels filled."
fi

echo "Starting ODM: 309 captures x G/R/RE/NIR; Blue excluded."
docker run --rm \
    --memory=6g \
    --memory-swap=6g \
    -v "$project:/datasets" \
    opendronemap/odm:latest "$dataset" \
    --project-path /datasets \
    --radiometric-calibration none \
    --primary-band NIR \
    --max-concurrency 2 \
    --feature-quality medium \
    --matcher-neighbors 8 \
    --use-hybrid-bundle-adjustment \
    --skip-3dmodel \
    --optimize-disk-space \
    --orthophoto-resolution 5 \
    --orthophoto-compression DEFLATE \
    --rerun-from opensfm \
    2>&1 | tee "$repair_log"

dataset_dir="$project/$dataset"
ortho="$dataset_dir/odm_orthophoto/odm_orthophoto.tif"
if [[ ! -f "$ortho" ]]; then
    echo "ODM finished without creating the expected orthophoto: $ortho" >&2
    exit 6
fi
repaired="$deliver/winterbarley_reflectance_orthomosaic_4band_repaired.tif"
cp -f "$ortho" "$repaired"

# Float32 概览需使用无损压缩；ODM 的 JPEG 概览设置会失败。
docker run --rm \
    --entrypoint /code/SuperBuild/install/bin/gdaladdo \
    -v "$deliver:/output" \
    opendronemap/odm:latest \
    -r average \
    --config COMPRESS_OVERVIEW DEFLATE \
    --config BIGTIFF_OVERVIEW IF_SAFER \
    /output/winterbarley_reflectance_orthomosaic_4band_repaired.tif \
    2 4 8 16

if [[ -d "$dataset_dir/odm_report" ]]; then
    cp -a "$dataset_dir/odm_report" "$deliver/"
fi
if [[ -f "$dataset_dir/odm_dem/dsm.tif" ]]; then
    cp -f "$dataset_dir/odm_dem/dsm.tif" "$deliver/winterbarley_dsm.tif"
fi

echo "Repaired orthomosaic copied to $repaired"
