from xml.etree import ElementTree

import geopandas as gpd
import rasterio


DJI_XMP_NAMESPACE = "http://www.dji.com/drone-dji/1.0/"
RDF_NAMESPACE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
XMP_TAGS = (
    "BandName",
    "BandFreq",
    "GpsStatus",
    "RtkFlag",
    "Irradiance",
    "SensorGain",
    "ExposureTime",
    "SensorGainAdjustment",
    "BlackLevel",
    "LS_status",
    "VignettingFlag",
    "VignettingData",
    "CalibratedOpticalCenterX",
    "CalibratedOpticalCenterY",
)


def read_boundary(path):
    return gpd.read_file(path)


def find_band(directory, band):
    """返回排序后的多光谱TIFF"""
    files = sorted(directory.glob(f"*MS_{band}.TIF"))
    if not files:
        raise FileNotFoundError(f"File {band} not found in {directory}")
    return files


def read_raster(path):
    """读取栅格影像的第一个波段，并返回该波段的数据以及影像的配置/元数据信息"""
    with rasterio.open(path) as src:
        data = src.read(1)
        profile = src.profile.copy()
    return data, profile


def read_dji_xmp(path):
    """返回栅格影像文件中指定的XMP属性"""
    with rasterio.open(path) as src:
        xmp = src.tags(ns="xml:XMP").get("xml:XMP")

    if not xmp:
        return {}

    root = ElementTree.fromstring(xmp)
    description = root.find(f".//{{{RDF_NAMESPACE}}}Description")
    if description is None:
        return {}

    tags = {}
    for key, value in description.attrib.items():
        if key.startswith(f"{{{DJI_XMP_NAMESPACE}}}"):
            local_name = key.rsplit("}", 1)[-1]
            if local_name in XMP_TAGS:
                tags[local_name] = value
    return tags
