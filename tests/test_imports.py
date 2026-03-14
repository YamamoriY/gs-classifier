"""全モジュールのインポートが正しく解決されることを検証する"""
# ruff: noqa: F401, F811


def test_import_models():
    from gs_classifier.models import GSplatData, TrunkLocation
    from gs_classifier.models.gsplat import GSplatData
    from gs_classifier.models.trunk import TrunkLocation


def test_import_core():
    from gs_classifier.core import (
        KDTree,
        load_ply_file,
        load_splat_file,
        save_ply_file,
    )
    from gs_classifier.core.gsloader import load_ply_file
    from gs_classifier.core.kdtree import KDTree


def test_import_core_submodules():
    from gs_classifier.core.denoise import NoiseRemover
    from gs_classifier.core.denoise.denoisecore import DenoiseCore
    from gs_classifier.core.ground import GroundDetector
    from gs_classifier.core.ground.groundcore import GroundLerp, SegGround
    from gs_classifier.core.image2d import Image2D
    from gs_classifier.core.leaf import LeafDetector
    from gs_classifier.core.leaf.leafcore import LeafClassifier
    from gs_classifier.core.trunk import TrunkClassifier


def test_import_viewer():
    from gs_classifier.viewer import Viewer
    from gs_classifier.viewer.colors import ColorCycle
    from gs_classifier.viewer.types import (
        GSplatFolder,
        GSplatHandle,
        GSplatMode,
    )


def test_import_util():
    from gs_classifier.util import pcd
