import laspy
import numpy as np
import open3d as o3d
import json, pdal
import string
from pathlib import Path
# Example: xxx_filter("input.las", "output.laz")

def smrf_filter(input: string, output: string):
    input_str = str(input).replace('\\', '/') if isinstance(input, Path) else str(input)
    output_str = str(output).replace('\\', '/') if isinstance(output, Path) else str(output)
    smrf_json = f"""
    [
        "{input_str}",
        {{
            "type":"filters.smrf",
            "scalar":1.2,
            "slope":0.2,
            "threshold":0.45,
            "window":16.0
        }},
        {{
            "type":"filters.expression",
            "expression":"Classification == 1 && Classification != 7"
        }},
        "{output_str}"
    ]
    """
    pipeline = pdal.Pipeline(smrf_json)
    count = pipeline.execute()
    print(f"pdal executed. ({output_str})   point count: {count}")

def csf_filter(input: string, output: string):
    input_str = str(input).replace('\\', '/') if isinstance(input, Path) else str(input)
    output_str = str(output).replace('\\', '/') if isinstance(output, Path) else str(output)
    csf_json = f"""
    [
        "{input_str}",
        {{
            "type":"filters.csf",
            "hdiff":1.0,
            "threshold":4.0
        }},
        {{
            "type":"filters.expression",
            "expression":"Classification == 1 && Classification != 7"
        }},
        "{output_str}"
    ]
    """
    pipeline = pdal.Pipeline(csf_json)
    count = pipeline.execute()
    print(f"pdal executed. ({output_str})   point count: {count}")

def csf_hag_filter(input: string, output: string):
    # Pathオブジェクトを文字列に変換し、Windowsパスのバックスラッシュをスラッシュに変換
    input_str = str(input).replace('\\', '/') if isinstance(input, Path) else str(input)
    output_str = str(output).replace('\\', '/') if isinstance(output, Path) else str(output)
    json = f"""
    [
        "{input_str}",
        {{
            "type":"filters.csf"
        }}
        ,
        {{
            "type":"filters.hag_delaunay"
        }},
        {{
            "type":"filters.expression",
            "expression":"HeightAboveGround >= 1.5 && HeightAboveGround < 3.5"
        }},
        {{
            "type":"writers.las",
            "filename":"{output_str}",
            "extra_dims":"HeightAboveGround=float32"
        }}
    ]
    """
    pipeline = pdal.Pipeline(json)
    count = pipeline.execute()
    print(f"output count: {count}")

def smrf_hag_filter(input: string, output: string):
    # Pathオブジェクトを文字列に変換し、Windowsパスのバックスラッシュをスラッシュに変換
    input_str = str(input).replace('\\', '/') if isinstance(input, Path) else str(input)
    output_str = str(output).replace('\\', '/') if isinstance(output, Path) else str(output)
    json = f"""
    [
        "{input_str}",
        {{
            "type":"filters.smrf",
            "scalar":1.2,
            "slope":0.2,
            "threshold":0.45,
            "window":16.0
        }},
        {{
            "type":"filters.hag_delaunay"
        }},
        {{
            "type":"filters.expression",
            "expression":"HeightAboveGround >= 1.5 && HeightAboveGround < 3.5"
        }},
        {{
            "type":"writers.las",
            "filename":"{output_str}",
            "extra_dims":"HeightAboveGround=float32"
        }}
    ]
    """
    pipeline = pdal.Pipeline(json)
    count = pipeline.execute()
    print(f"output count: {count}")

# NOTE: 重ければ downsample を最初にしてもいいかも
# smrf -> downsample -> hag -> shift to origin
def hag_preprocess(input: string, output: string):
    # Pathオブジェクトを文字列に変換し、Windowsパスのバックスラッシュをスラッシュに変換
    input_str = str(input).replace('\\', '/') if isinstance(input, Path) else str(input)
    output_str = str(output).replace('\\', '/') if isinstance(output, Path) else str(output)
    json = f"""
    [
        "{input_str}",
        {{
            "type":"filters.smrf",
            "scalar":1.2,
            "slope":0.2,
            "threshold":0.45,
            "window":16.0
        }},
        {{
            "type":"filters.voxeldownsize",
            "cell":0.2,
            "mode":"first"
        }},
        {{
            "type":"filters.hag_delaunay"
        }},
        {{
            "type":"filters.assign",
            "value": [
                "Z = HeightAboveGround"
            ]
        }},
        {{
            "type":"filters.expression",
            "expression":"Classification == 1 && Classification != 7"
        }},
        {{
            "type":"writers.las",
            "filename":"{output_str}",
            "extra_dims":"HeightAboveGround=float32"
        }}
    ]
    """
    pipeline = pdal.Pipeline(json)
    count = pipeline.execute()
    print(f"output count: {count}")
    
def add_hag_dim(input: string, output: string):
    input_str = str(input).replace('\\', '/') if isinstance(input, Path) else str(input)
    output_str = str(output).replace('\\', '/') if isinstance(output, Path) else str(output)
    json = f"""
    [
        "{input_str}",
        {{
            "type":"filters.smrf",
            "scalar":1.2,
            "slope":0.2,
            "threshold":0.45,
            "window":16.0
        }},
        {{
            "type":"filters.hag_delaunay"
        }},
        {{
            "type":"writers.las",
            "filename":"{output_str}",
            "extra_dims":"HeightAboveGround=float32"
        }}
    ]
    """
    pipeline = pdal.Pipeline(json)
    count = pipeline.execute()
    print(f"output count: {count}")

if __name__ == "__main__":
    # pdal
    smrf_filter("akan2.las", "smrf.laz")
    # hag_preprocess("akan2.las", "hag.laz")