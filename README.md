# m3m — DJI Mavic 3 Multispectral processing toolkit

[![CI](https://github.com/snow-commits/DJI_M3M_WinterBarley_Analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/snow-commits/DJI_M3M_WinterBarley_Analysis/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

A label-free processing toolkit for **DJI Mavic 3 Multispectral (M3M)** imagery:
radiometric calibration, orthomosaic quality assessment, vegetation indices,
within-field relative zones, and a sensitivity analysis of the zoning itself.

Everything here is `pip install`-able, needs **no external binaries** (no
ExifTool, no GDAL CLI, no Docker) for the core pipeline, and never loads a whole
orthomosaic into memory — indices and quantiles are computed window by window
into fixed-range histograms.

> **What this is not.** Nothing in this toolkit predicts biomass or yield, and
> nothing here has been validated against field-measured ground truth. The
> default pipeline ends at *relative* within-field zoning. See
> [Methodology and limitations](#methodology-and-limitations) before quoting any
> number it produces.

---

## Install

```bash
git clone https://github.com/snow-commits/DJI_M3M_WinterBarley_Analysis.git
cd DJI_M3M_WinterBarley_Analysis
python -m pip install -e .
```

Requires Python 3.10+. Pulls in `numpy`, `rasterio`, `geopandas`, `shapely`,
`matplotlib` and `scipy`.

## Quick start — no flight data required

The repository ships a synthetic M3M dataset generator so you can run the entire
pipeline offline, in about ten seconds:

```bash
m3m synthetic --dest demo

m3m audit       --config demo/m3m.toml
m3m panel       --config demo/m3m.toml
m3m calibrate   --config demo/m3m.toml
m3m indices     --config demo/m3m.toml --raster demo/synthetic_reflectance_orthomosaic.tif
m3m zones       --config demo/m3m.toml
m3m compare     --config demo/m3m.toml
m3m sensitivity --config demo/m3m.toml
m3m prescribe   --config demo/m3m.toml \
  --rate 1=0,2=40,3=80,4=120,5=160 --unit "kg N/ha" \
  --target-mean-rate 100 --product-price 4.5 --application-cost-per-ha 25
```

The generated captures are written as the exact inverse of the DJI signal
model, so `calibrate` must return the reflectance the generator asked for.
That round trip is asserted in the test suite — it is the strongest guarantee
available without proprietary flight data.

## Quick start — your own data

```bash
m3m config init            # writes a commented m3m.toml template
$EDITOR m3m.toml           # point [paths] at your captures
m3m config show            # confirm what was resolved, and from where

m3m audit                  # how many capture groups, which bands are missing
m3m panel                  # panel ROI statistics, saturation, NIR anchor
m3m calibrate              # raw DN -> reflectance GeoTIFFs
m3m indices --raster <reflectance_orthomosaic.tif>
m3m zones
```

`m3m panel` is not optional the first time. Panel ROIs depend on your capture
geometry and cannot be guessed; the command tells you exactly which
`[radiometric.panel_roi]` entry to fill in and can render an overlay PNG
(`--show-roi`) so you can measure the panel position on your own images.

## Commands

| Command | What it does |
| --- | --- |
| `m3m config show` / `config init` | Inspect the resolved configuration, or write a starter `m3m.toml` |
| `m3m audit` | Inventory capture groups; flag missing bands, size and dtype mismatches |
| `m3m panel` | Panel ROI statistics, saturation fraction, NIR anchor, cross-band consistency |
| `m3m calibrate` | Apply the DJI vignetting/black-level/gain/exposure/irradiance chain and anchor to the panel |
| `m3m orthorectify` | Render (and optionally run) an OpenDroneMap script for the four-band reflectance set |
| `m3m fill-nodata` | Fill saturation placeholders in the throw-away ODM working copy |
| `m3m assess` | Check an orthomosaic for physically impossible values and boundary coverage |
| `m3m indices` | NDVI / NDRE / EVI2 rasters with memory-bounded statistics |
| `m3m zones` | Within-field relative quintiles, area table, maps |
| `m3m compare` | Index pairwise agreement, or a repaired orthomosaic against the original |
| `m3m sensitivity` | How much the zoning depends on the chosen index weights |
| `m3m prescribe` | Turn the zones into a variable-rate map with areas, product quantities and costs |
| `m3m synthetic` | Generate a synthetic dataset |

Run `m3m <command> --help` for the full flag list.

## Configuration

Resolution order, lowest to highest priority:

1. built-in defaults
2. a TOML file — `--config`, `$M3M_CONFIG`, or `./m3m.toml` searched upwards
3. `M3M_*` environment variables
4. explicit command-line flags

```toml
[paths]
img_dir     = "data/Dataset"
panel_dir   = "data/Reflectance_Panel"
boundary    = "data/boundary.gpkg"
out_dir     = "outputs"

[radiometric]
saturation_dn = 65520

[radiometric.panel_reflectance]      # from your panel's certificate
G = 0.493
R = 0.494
RE = 0.494
NIR = 0.493

[radiometric.panel_roi]              # [x_min, y_min, x_max, y_max] in panel-capture pixels
NIR = [926, 1160, 1144, 1390]

[output]
prefix = "field"                     # -> field_ndvi.tif
```

Relative paths inside the TOML file resolve against that file's own directory.
`m3m.toml` is git-ignored on purpose. See [`m3m.example.toml`](m3m.example.toml)
for the fully commented version.

When a required path is missing, the error names the flag, the environment
variable and the TOML key that would supply it — the code contains no
machine-specific fallback paths.

## Input data

Raw captures must be named `<name>_<NNNN>_MS_<BAND>.TIF` with `BAND` one of
`G`, `R`, `RE`, `NIR`, accompanied by the true-colour frame `<name>_<NNNN>_D.JPG`.
Each capture must carry DJI's XMP block; `m3m audit` reports any group that is
incomplete, and calibration refuses to run on a group with missing bands rather
than silently emitting partial results.

The reflectance orthomosaic consumed by `m3m indices` must have five bands with
descriptions `Red`, `Green`, `NIR`, `RedEdge`, `Alpha` in that order. Band
positions are validated against the descriptions, so a mislabelled export fails
loudly instead of producing plausible-looking nonsense.

Note that the M3M has **no independent blue band**. Blue-dependent indices such
as the standard three-band EVI cannot be computed; this toolkit uses EVI2 for
that reason.

## Outputs

```
outputs/
├─ dataset_inventory.csv              m3m audit
├─ radiometric/                       m3m calibrate
│  ├─ *_REFLECTANCE_APPROX.tif
│  └─ radiometric_manifest.csv        per-image status, saturation, min/median/mean/max
├─ indices/                           m3m indices, zones, compare
│  ├─ field_ndvi.tif, field_ndre.tif, field_evi2.tif
│  ├─ field_ndvi_relative_5class.tif  ...
│  ├─ vegetation_index_summary.csv
│  ├─ vegetation_index_relative_classes.csv
│  └─ *.png
├─ sensitivity/                       m3m sensitivity
├─ prescription/                      m3m prescribe
│  ├─ field_ndvi_prescription.tif     float32 application rate
│  ├─ field_ndvi_prescription_summary.csv   per-zone rate, area, quantity, cost
│  ├─ field_ndvi_prescription_totals.json   totals + limitations, machine-readable
│  └─ field_ndvi_prescription.png     rate map + per-zone quantity chart
├─ orthomosaic/                       m3m orthorectify, assess, compare
└─ orthorectify.sh                    rendered, review before running
```

## Methodology and limitations

These are the constraints on what the outputs mean. They are stated here rather
than buried, because most of them cannot be checked from the imagery alone.

1. **Reflectance is provisional.** The chain follows DJI's documented per-image
   model (vignetting polynomial, black level, gain/exposure, per-band gain
   adjustment, DLS irradiance) and then applies a single scene scale anchored on
   the NIR panel. Per-band absolute reflectance is **not** independently
   ground-calibrated, and the panel ROI and saturation threshold still deserve
   verification on your own data. Do not describe the output as fully validated
   absolute reflectance.

2. **The green band saturates on some flights.** On the reference dataset the
   green panel ROI saturated, so no reliable independent uncertainty could be
   estimated for it and GNDVI was excluded from the formal statistics. Check
   `m3m panel` output before trusting any green-band product.

3. **Index uncertainty is not propagated.** Per-band uncertainties and
   inter-band covariances are unavailable, so the reported statistics are not a
   substitute for a proper uncertainty budget.

4. **There is no ground truth.** Without field-measured biomass, LAI or yield
   paired to pixels, no supervised model can be trained or validated and no
   meaningful predictive R², RMSE or MAE can be reported. The toolkit therefore
   reports none, and the zoning it produces is relative to one field on one
   date.

5. **Relative classes are not agronomic thresholds.** The five classes are
   equal-frequency quantiles within the analysed field. "Highest relative" means
   greener than 80% of *this* field *today* — not "high yield" and not a
   fertiliser prescription.

6. **The prescription rate schedule is an operator input, not a model output.**
   `m3m prescribe` applies a schedule you supply and reports the resulting areas,
   product quantities and costs. It does not derive, optimise or validate those
   rates, because it has no ground truth to do so with. Two further consequences
   are easy to miss: because the zones are equal-frequency, each class always
   covers roughly a fifth of the field, so a relative-mode prescription can
   redistribute product within a field-level decision but can never conclude that
   a field needs less overall — use `target_mean_rate` (or absolute rates) to
   keep the total under operator control. And computed areas are pixel counts
   times the nominal pixel area, so they exclude headland, overlap and implement
   turning losses.

7. **`m3m sensitivity` measures weight sensitivity, not accuracy.** It re-weights
   the same index ranks and reports how similar the resulting maps are. Two
   candidate scores agreeing does not make either closer to biomass: with no
   ground truth in the comparison there is nothing for the agreement to be
   measured against. It is a self-referential robustness check, and the module
   docstring says so.

8. **Geometric accuracy is not verified.** Orthorectification quality checks
   compare grids and coverage; they are not an independent assessment of
   absolute positional error, which needs surveyed ground control points.

9. **Temporal and environmental coverage is narrow.** Results describe the
   captured dates only and do not carry over to the whole growing season.

### What would lift these limits

- Field-measured biomass / LAI / yield samples paired to pixels — turns the
  relative ranking into a validated regression.
- Independent ground control points — turns grid comparison into a positional
  accuracy statement.
- Per-band uncertainty and covariance estimates — enables a real error
  propagation through the indices.

## Development

```bash
python -m pip install -e ".[dev]"
pytest              # full suite: no network, no proprietary data
ruff check src tests
```

The test suite generates its own synthetic data, so it runs identically on a
laptop and in CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml), Ubuntu
and Windows × Python 3.10/3.12).

## Migrating from the pre-0.2 scripts

The original one-off scripts are now library modules behind a single CLI. Their
logic is unchanged; only paths, configuration and packaging moved.

| Before | Now |
| --- | --- |
| `python audit_dataset.py` | `m3m audit` |
| `python main.py [--limit N]` | `m3m calibrate [--limit N]` |
| `python -m src.m3m.radiometric` | `m3m panel` |
| `scripts/calculate_vegetation_indices.py` | `m3m indices` |
| `scripts/map_vegetation_indices.py` | `m3m zones` |
| `scripts/compare_vegetation_indices.py` | `m3m compare` |
| `scripts/compare_proxy_indicators.py` | `m3m sensitivity` |
| `scripts/compare_repair_results.py` | `m3m compare --before A --after B` |
| `scripts/assess_orthomosaic.py` | `m3m assess` |
| `scripts/prepare_odm_inputs.py` | `m3m fill-nodata` |
| `scripts/orthorectify_wsl.sh` | `m3m orthorectify` (rendered from config) |

Output filenames lost their dataset-specific literal. Pass
`--prefix winterbarley` (or set `[output] prefix`) to reproduce the old names.

## License

MIT — see [LICENSE](LICENSE).

---

## 中文说明

### 这个项目是什么

一个面向 **DJI Mavic 3 Multispectral（M3M）** 多光谱影像的处理工具包：辐射校正、正射影像质检、
植被指数、田内相对分区，以及分区对权重选择的敏感性分析。

核心流程 `pip install` 即可运行，**不需要 ExifTool、不需要 GDAL 命令行、不需要 Docker**；
指数与分位数统计全部按窗口流式计算，几千万像元也不会把内存吃满。

### 三分钟跑通（不需要任何飞行数据）

```bash
m3m synthetic --dest demo                    # 生成一套合成数据集
m3m calibrate  --config demo/m3m.toml        # 辐射校正
m3m indices    --config demo/m3m.toml --raster demo/synthetic_reflectance_orthomosaic.tif
m3m zones      --config demo/m3m.toml
m3m prescribe  --config demo/m3m.toml --rate 1=0,2=40,3=80,4=120,5=160 \
               --unit "kg N/ha" --target-mean-rate 100 --product-price 4.5
```

合成影像是按 DJI 信号模型的**精确逆运算**生成的，所以校正结果必须还原出生成时设定的反射率——
这条往返断言写在测试里，是在没有真实飞行数据时能做的最强保证。

### 用自己的数据

```bash
m3m config init     # 生成带注释的 m3m.toml
# 编辑 [paths] 指向你的数据
m3m panel           # 先跑这个！面板 ROI 必须按你自己的影像量取
m3m calibrate
m3m indices --raster <你的反射率正射影像.tif>
m3m zones
```

配置优先级：默认值 < `m3m.toml` < `M3M_*` 环境变量 < 命令行参数。
代码里**没有任何写死的本机路径**；缺少必需路径时，报错会明确告诉你该设哪个参数、哪个环境变量、哪个 TOML 键。

### 必须知道的限制

- **反射率是近似结果**：按 DJI 文档的逐影像模型（暗角、黑电平、增益/曝光、波段增益、DLS 辐照度）
  处理后，以 NIR 反射率板锚定一个共用尺度；**没有做逐波段独立的地面反射率标定**。
- **没有田间实测真值**：因此不训练、也不报告任何有监督预测指标（R²/RMSE/MAE）。
  输出的分区是"同一田块、同一日期内的相对等级"，**不是农学阈值，也不是施肥处方**。
- **`m3m sensitivity` 测的是权重敏感性，不是精度**：它把同一份指数秩重新加权后比较结果相似度，
  没有真值参与，所以不能解读为"某个方案更接近生物量"。
- **`m3m prescribe` 的施肥/施药量是你给的，不是模型算的**：工具负责把这份方案一致地落到每个分区，
  并算出各自面积、用量和成本；它没有能力推导或验证这些数值。另外两个容易踩的点：分区是等频的，
  每级恒定约占田块 1/5，所以相对模式下的处方只能在"总量已定的前提下重新分配"，**永远得不出
  "这块田整体要少施"的结论**（要么用 `--target-mean-rate` 锁住总量，要么直接用绝对用量）；
  面积是按像元数 × 名义像元面积算的，**不含地头、重叠和机具转弯损失**。
- **几何精度未验证**：缺少独立测量的地面控制点。
- M3M **没有独立蓝波段**，因此无法计算标准三波段 EVI，本工具用 EVI2 替代。

### 与旧脚本的对应关系

原来的独立脚本已全部迁移为库模块并统一到一个 CLI，**算法逻辑未改动**，只是路径、配置和打包方式变了。
详细的对照表见上方 "Migrating from the pre-0.2 scripts" 一节。
输出文件名去掉了 `winterbarley` 这个数据集专有前缀，可用 `--prefix winterbarley` 还原。
