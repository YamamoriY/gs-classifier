"""gs-classifier CLI entry point.

Usage:
    gs-classifier ground    [--ply PATH] [--no-viewer]
    gs-classifier trunk     [--no-viewer]
    gs-classifier tree      [--no-viewer]
    gs-classifier identify  <image> [--no-display]
    gs-classifier view      <file> [<file> ...]
"""

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    """CLI のサブコマンドパーサーを構築する。"""
    parser = argparse.ArgumentParser(
        prog="gs-classifier",
        description="3D Gaussian Splatting forest classifier",
    )
    sub = parser.add_subparsers(dest="command")

    # --- ground ---
    p_ground = sub.add_parser(
        "ground", help="PLY -> denoise -> ground detection"
    )
    p_ground.add_argument(
        "--ply", default="data/takino.ply", help="input PLY file path"
    )
    p_ground.add_argument(
        "--no-viewer", action="store_true", help="skip viewer"
    )

    # --- trunk ---
    p_trunk = sub.add_parser("trunk", help="ground result -> trunk detection")
    p_trunk.add_argument(
        "--no-viewer", action="store_true", help="skip viewer"
    )

    # --- tree ---
    p_tree = sub.add_parser(
        "tree", help="trunk result -> individual tree extraction"
    )
    p_tree.add_argument("--no-viewer", action="store_true", help="skip viewer")

    # --- identify ---
    p_id = sub.add_parser(
        "identify", help="identify plant species from image via PlantNet"
    )
    p_id.add_argument("image", help="input image file path")
    p_id.add_argument(
        "--no-display", action="store_true", help="skip matplotlib display"
    )

    # --- view ---
    p_view = sub.add_parser("view", help="view PLY / NPZ files in 3D viewer")
    p_view.add_argument("files", nargs="+", help="PLY or NPZ files to view")

    return parser


def main() -> None:
    """CLI のメインエントリポイント。"""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "ground":
        from gs_classifier.pipeline.detect_ground import main as run_ground

        run_ground(ply_path=args.ply, no_viewer=args.no_viewer)

    elif args.command == "trunk":
        from gs_classifier.pipeline.detect_trunk import main as run_trunk

        run_trunk(no_viewer=args.no_viewer)

    elif args.command == "tree":
        from gs_classifier.pipeline.detect_tree import main as run_tree

        run_tree(no_viewer=args.no_viewer)

    elif args.command == "identify":
        from gs_classifier.api.plantnet import main as run_identify

        run_identify(image_path=args.image, no_display=args.no_display)

    elif args.command == "view":
        _run_view(args.files)


def _run_view(files: list[str]) -> None:
    """PLY/NPZ ファイルを 3D ビューアーで表示する。

    Args:
        files: 表示するファイルパスのリスト。
    """
    from pathlib import Path

    import numpy as np

    from gs_classifier.core import load_ply_file
    from gs_classifier.models import GSplatData
    from gs_classifier.viewer import Viewer

    viewer = Viewer()
    for filepath in files:
        path = Path(filepath)
        if path.suffix == ".ply":
            gs = load_ply_file(path, center=True)
            # 座標変換（x軸周り-90°）
            R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            gs.coordinate_transform(R)
        elif path.suffix == ".npz":
            gs = GSplatData.load_from_npz(str(path))
        else:
            print(
                f"unsupported file format: {path.suffix} (expected .ply or .npz)"
            )
            continue
        viewer.add_gsplat(gs, name=path.stem, folder_name=path.stem)
        print(f"loaded: {filepath}")

    viewer.run()


if __name__ == "__main__":
    main()
