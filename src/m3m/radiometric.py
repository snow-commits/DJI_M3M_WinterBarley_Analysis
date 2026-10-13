import csv
import re

import numpy as np
import rasterio

from src.m3m.config import img_dir, panel_dir, panel_reflectance, project_dir
from src.m3m.io import find_band, read_dji_xmp, read_raster
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


BANDS = ("G", "R", "RE", "NIR")

# 各波段影像中白板内部的近似范围，坐标暂定，需目视确认。
# 坐标顺序为整幅 TIFF 中的 (x_min, y_min, x_max, y_max)。
PANEL_ROIS = {
    "G": (990, 1150, 1200, 1380),
    "R": (948, 1150, 1166, 1378),
    "RE": (938, 1185, 1158, 1410),
    "NIR": (926, 1160, 1144, 1390),
}

# 当前 TIFF 中观察到的暂定阈值，尚未确认是否为传感器规定的饱和值。
SATURATION_DN = 65520


def inspect_panel_image(path, band):
    """输出影像、ROI、饱和情况及相关 XMP 元数据以供检查。"""
    data, _ = read_raster(path)
    height, width = data.shape

    roi_bounds = PANEL_ROIS[band]
    x_min, y_min, x_max, y_max = roi_bounds
    if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
        raise ValueError(
            f"ROI {roi_bounds} is outside {path.name} dimensions {width}x{height}"
        )

    roi = data[y_min:y_max, x_min:x_max]
    saturated = roi >= SATURATION_DN

    print(f"\n{band} 波段：{path.name}")
    print(f"  整图尺寸/类型：{width} x {height} / {data.dtype}")
    print(f"  白板 ROI (x0,y0,x1,y1)：{roi_bounds}")
    print(
        "  ROI DN min/median/mean/max："
        f"{roi.min()} / {np.median(roi):.1f} / {roi.mean():.1f} / {roi.max()}"
    )
    print(
        f"  ROI 达到 {SATURATION_DN} 的比例："
        f"{100 * saturated.mean():.2f}% ({saturated.sum()}/{roi.size})"
    )

    metadata = read_dji_xmp(path)
    print("  DJI XMP：")
    if metadata:
        for key, value in metadata.items():
            print(f"    {key}: {value}")
    else:
        print("    未找到 DJI XMP 参数")


def dji_relative_signal(data, metadata):
    """
    对每张图像应用 DJI M3M 的信号归一化和暗角模型。
    返回的数组是经过辐照度归一化（irradiance-normalized）的信号。
    """
    bits = np.iinfo(data.dtype).bits
    scale = float(2**bits)
    image = data.astype(np.float32) / scale
    black_level = float(metadata["BlackLevel"]) / scale
    gain = float(metadata["SensorGain"])
    exposure_seconds = float(metadata["ExposureTime"]) / 1_000_000.0
    band_gain_adjustment = float(metadata["SensorGainAdjustment"])
    irradiance = float(metadata["Irradiance"])

    if gain <= 0 or exposure_seconds <= 0 or irradiance <= 0:
        raise ValueError("Gain, exposure time, and irradiance must be positive")

    coefficients = [
        float(value.strip())
        for value in metadata["VignettingData"].split(",")
    ]
    if len(coefficients) != 6:
        raise ValueError("Expected six DJI vignetting coefficients")

    center_x = float(metadata["CalibratedOpticalCenterX"])
    center_y = float(metadata["CalibratedOpticalCenterY"])
    rows, columns = np.indices(data.shape, dtype=np.float32)
    radius = np.hypot(columns - center_x, rows - center_y)
    vignette_factor = np.ones(data.shape, dtype=np.float32)
    radius_power = radius.copy()
    for coefficient in coefficients:
        vignette_factor += coefficient * radius_power
        radius_power *= radius

    if not np.all(np.isfinite(vignette_factor)) or np.any(vignette_factor <= 0):
        raise ValueError("DJI vignetting model produced invalid correction factors")

    # 按 DJI 流程依次进行暗角校正、黑电平校正、曝光/增益归一化、波段增益调整和 DLS 辐照度归一化。
    corrected = np.maximum(image * vignette_factor - black_level, 0.0)
    corrected /= gain * exposure_seconds
    corrected *= band_gain_adjustment
    corrected /= irradiance
    return corrected


def calibrate_nir_anchor():
    """根据 NIR 白板估算暂定的共用反射率尺度。"""
    path = find_band(panel_dir, "NIR")[0]
    data, _ = read_raster(path)
    metadata = read_dji_xmp(path)
    x0, y0, x1, y1 = PANEL_ROIS["NIR"]
    panel = data[y0:y1, x0:x1]
    valid = np.isfinite(panel) & (panel < SATURATION_DN)
    if not np.any(valid):
        raise ValueError("No unsaturated pixels in the NIR panel ROI")

    corrected = dji_relative_signal(data, metadata)
    panel_signal = corrected[y0:y1, x0:x1][valid]
    panel_mean_signal = float(panel_signal.mean())

    panel_reflectance_nir = float(panel_reflectance["NIR"])
    scale = panel_reflectance_nir / panel_mean_signal
    return scale, panel_reflectance_nir, panel_mean_signal, valid.mean()


def preview_one_flight(scale):
    """没有经过正射校正的临时样本"""
    output_dir = project_dir / "outputs" / "radiometric_preview_approx"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n单组主航飞影像试算：")
    print("预览输出目录：", output_dir)

    for band in BANDS:
        path = find_band(img_dir, band)[0]
        data, profile = read_raster(path)
        metadata = read_dji_xmp(path)
        reflectance = dji_relative_signal(data, metadata) * scale
        saturated = data >= SATURATION_DN
        reflectance[saturated] = -9999.0

        output_path = output_dir / f"{path.stem}_REFLECTANCE_APPROX.tif"
        profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            nodata=-9999.0,
            compress="deflate",
            predictor=3,
        )
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(reflectance.astype(np.float32), 1)
            dst.update_tags(
                PRODUCT="Provisional M3M reflectance preview",
                CALIBRATION="DJI per-image metadata, anchored to the NIR panel",
                PANEL_NIR_REFLECTANCE=f"{panel_reflectance['NIR']:.6f}",
                NIR_SCALE_FACTOR=f"{scale:.8f}",
                NOTE=(
                    "Approximate NIR panel anchor; not orthorectified"
                ),
            )

        valid = ~saturated
        values = reflectance[valid]
        print(
            f"  {band}: min={values.min():.4f}, "
            f"median={np.median(values):.4f}, "
            f"mean={values.mean():.4f}, "
            f"max={values.max():.4f}, "
            f"saturated={100 * saturated.mean():.3f}%, "
            f"saved={output_path}"
        )


def collect_flight_groups(directory=img_dir):
    """收集每个拍摄序列对应的四个多光谱文件"""
    groups = {}
    pattern = re.compile(r"_(\d{4})_MS_(G|R|RE|NIR)$", re.IGNORECASE)

    for path in directory.glob("*_MS_*.TIF"):
        match = pattern.search(path.stem)
        if not match:
            continue
        group_id, band = match.group(1), match.group(2).upper()
        band_paths = groups.setdefault(group_id, {})
        if band in band_paths:
            raise ValueError(f"重复的 {band} 波段文件：{group_id}")
        band_paths[band] = path

    if not groups:
        raise FileNotFoundError(f"没有在 {directory} 找到多光谱 TIFF")

    expected = set(BANDS)
    incomplete = {
        group_id: sorted(expected - set(paths))
        for group_id, paths in groups.items()
        if set(paths) != expected
    }
    if incomplete:
        examples = "; ".join(
            f"{group_id}:缺{','.join(missing)}"
            for group_id, missing in list(sorted(incomplete.items()))[:10]
        )
        raise ValueError(
            f"发现 {len(incomplete)} 组缺少波段，已停止以免产生不完整输出。{examples}"
        )

    return dict(sorted(groups.items()))


def calibrate_and_write(path, band, scale, output_dir):
    """对一幅影像进行辐射校正，并将其保存为单波段浮点型 GeoTIFF 文件。"""
    data, profile = read_raster(path)
    metadata = read_dji_xmp(path)
    required = (
        "BlackLevel", "SensorGain", "ExposureTime", "SensorGainAdjustment",
        "Irradiance", "VignettingData", "CalibratedOpticalCenterX",
        "CalibratedOpticalCenterY",
    )
    missing = [name for name in required if name not in metadata]
    if missing:
        raise ValueError(f"{path.name}: XMP 缺少参数 {', '.join(missing)}")

    reflectance = dji_relative_signal(data, metadata) * scale
    saturated = data >= SATURATION_DN
    valid = np.isfinite(reflectance) & ~saturated
    if not np.any(valid):
        raise ValueError(f"{path.name}: 没有可用像素")
    reflectance[~valid] = -9999.0

    output_path = output_dir / f"{path.stem}_REFLECTANCE_APPROX.tif"
    profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=-9999.0,
        compress="deflate",
        predictor=3,
    )
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(reflectance.astype(np.float32), 1)
        dst.update_tags(
            PRODUCT="Provisional M3M reflectance image",
            CALIBRATION="DJI metadata normalization anchored to NIR panel",
            PANEL_NIR_REFLECTANCE=f"{panel_reflectance['NIR']:.6f}",
            NIR_SCALE_FACTOR=f"{scale:.8f}",
            SATURATION_DN=f"{SATURATION_DN}",
            NOTE=(
                "Per-image reflectance estimate; saturation marked nodata; "
                "not orthorectified and not a multispectral orthomosaic."
            ),
        )

    values = reflectance[valid]
    return {
        "band": band,
        "source": str(path),
        "output": str(output_path),
        "valid_fraction": float(valid.mean()),
        "saturated_fraction": float(saturated.mean()),
        "reflectance_min": float(values.min()),
        "reflectance_median": float(np.median(values)),
        "reflectance_mean": float(values.mean()),
        "reflectance_max": float(values.max()),
    }


def process_dataset(scale, max_groups=None, output_dir=None):
    """按顺序处理所有拍摄组，并可选择设置一个试运行数量上限。"""
    groups = collect_flight_groups()
    if max_groups is not None:
        groups = dict(list(groups.items())[:max_groups])

    if output_dir is None:
        folder = (
            "radiometric_full_approx"
            if max_groups is None
            else f"radiometric_pilot_{max_groups}_approx"
        )
        output_dir = project_dir / "outputs" / folder
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "radiometric_manifest.csv"
    fieldnames = [
        "group_id", "band", "source", "output", "status", "error",
        "valid_fraction", "saturated_fraction", "reflectance_min",
        "reflectance_median", "reflectance_mean", "reflectance_max",
    ]
    manifest_exists = manifest_path.exists()

    print(f"待处理影像组：{len(groups)}")
    print(f"输出目录：{output_dir}")
    print("逐张、顺序处理，不会将整个数据集载入内存。")
    print("注意：这些仍是单张影像反射率结果，不是正射影像。")

    completed = 0
    failed = 0
    with manifest_path.open(
        "a", newline="", encoding="utf-8-sig"
    ) as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        if not manifest_exists:
            writer.writeheader()
            manifest_file.flush()

        for group_index, (group_id, paths) in enumerate(groups.items(), start=1):
            group_failed = False
            for band in BANDS:
                path = paths[band]
                output_path = output_dir / f"{path.stem}_REFLECTANCE_APPROX.tif"
                if output_path.exists():
                    completed += 1
                    continue

                try:
                    result = calibrate_and_write(path, band, scale, output_dir)
                    writer.writerow({
                        "group_id": group_id,
                        **result,
                        "status": "ok",
                        "error": "",
                    })
                    completed += 1
                except Exception as exc:
                    writer.writerow({
                        "group_id": group_id,
                        "band": band,
                        "source": str(path),
                        "output": str(output_path),
                        "status": "failed",
                        "error": repr(exc),
                        "valid_fraction": "",
                        "saturated_fraction": "",
                        "reflectance_min": "",
                        "reflectance_median": "",
                        "reflectance_mean": "",
                        "reflectance_max": "",
                    })
                    failed += 1
                    group_failed = True
                    print(f"失败：{group_id} {band}：{exc}")
                manifest_file.flush()

            if group_index == 1 or group_index % 10 == 0 or group_index == len(groups):
                print(
                    f"进度：{group_index}/{len(groups)} 组；"
                    f"完成影像 {completed}，失败 {failed}"
                    + (f"；最近失败组 {group_id}" if group_failed else "")
                )

    print(f"处理结束：完成影像 {completed}，失败 {failed}")
    print(f"处理清单：{manifest_path}")
    return completed, failed

def plot_roi(path,ROI):
    img,_ = read_raster(path)
    x0,y0,x1,y1 = ROI
    fig,ax = plt.subplots()
    ax.imshow(img,cmap="gray")
    ax.add_patch(
        Rectangle(
            (x0,y0),
            x1-x0,
            y1-y0,
            fill=False,
            edgecolor="red",
            linewidth=1,
        )
    )
    ax.set_xlim(max(0,x0-100),min(img.shape[1],x1+100))
    ax.set_ylim(min(img.shape[0], y1 + 100), max(0, y0 - 100))
    ax.set_title(path.name)
    plt.show()

def check_with_nir_anchor(scale):
    print("NIR检查R,RE的预测值")
    for band in ("R","RE"):
        path = find_band(panel_dir,band)[0]
        data,_ = read_raster(path)
        metadata = read_dji_xmp(path)
        x0,y0,x1,y1 = PANEL_ROIS[band]
        roi_dn = data[y0:y1,x0:x1]
        valid = roi_dn < SATURATION_DN

        if not np.any(valid):
            print(f"{band}:ROI 没有饱和")
            continue

        reflectance = dji_relative_signal(data,metadata)*scale
        roi_reflectance = reflectance[y0:y1,x0:x1][valid]

        predicted = float(roi_reflectance.mean())
        target = float(panel_reflectance[band])
        error = (predicted - target) / target * 100
        print(
            f"  {band}：板标称值={target:.4f}，"
            f"预测均值={predicted:.4f}，"
            f"相对差异={error:+.2f}%"
        )

def main():
    print("反射率板目录：", panel_dir)
    print("各波段 ROI 根据各自影像预览分别估计，仍需人工确认。")

    for band in BANDS:
        paths = find_band(panel_dir, band)
        if len(paths) > 1:
            print(f"警告：{band} 找到 {len(paths)} 张板照，将逐张检查。")
        for path in paths:
            inspect_panel_image(path, band)
            # plot_roi(path, PANEL_ROIS[band])

    scale, target, panel_signal, valid_fraction = calibrate_nir_anchor()
    check_with_nir_anchor(scale)
    # print("\nDJI M3M NIR 锚定系数初步试算：")
    # print(f"  面板标称 NIR 反射率：{target:.4f}")
    # print(f"  白板未饱和像素比例：{100 * valid_fraction:.2f}%")
    # print(f"  白板校正后相对信号均值：{panel_signal:.8f}")
    # print(f"  初步绝对尺度系数：{scale:.8f}")

    preview_one_flight(scale)


if __name__ == "__main__":
    main()
