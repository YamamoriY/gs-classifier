from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import tyro
from plyfile import PlyData, PlyElement

import viser
from viser import transforms as tf

from src.lib.types.gstype import GSplatData

# viser 公式リポジトリ参考

def load_splat_file(splat_path: Path, center: bool = False) -> GSplatData:
    start_time = time.time()
    splat_buffer = splat_path.read_bytes()
    bytes_per_gaussian = (
        # Each Gaussian is serialized as:
        # - position (vec3, float32)
        3 * 4
        # - xyz (vec3, float32)
        + 3 * 4
        # - rgba (vec4, uint8)
        + 4
        # - ijkl (vec4, uint8), where 0 => -1, 255 => 1.
        + 4
    )
    assert len(splat_buffer) % bytes_per_gaussian == 0
    num_gaussians = len(splat_buffer) // bytes_per_gaussian

    # Reinterpret cast to dtypes that we want to extract.
    splat_uint8 = np.frombuffer(splat_buffer, dtype=np.uint8).reshape(
        (num_gaussians, bytes_per_gaussian)
    )
    scales = splat_uint8[:, 12:24].copy().view(np.float32)
    wxyzs = splat_uint8[:, 28:32] / 255.0 * 2.0 - 1.0
    Rs = tf.SO3(wxyzs).as_matrix()
    covariances = np.einsum(
        "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )
    centers = splat_uint8[:, 0:12].copy().view(np.float32)
    if center:
        centers -= np.mean(centers, axis=0, keepdims=True)
    print(
        f"Splat file with {num_gaussians=} loaded in {time.time() - start_time} seconds"
    )
    return GSplatData(
        centers=centers,
        # Colors should have shape (N, 3).
        rgbs=splat_uint8[:, 24:27] / 255.0,
        opacities=splat_uint8[:, 27:28] / 255.0,
        # Covariances should have shape (N, 3, 3).
        covariances=covariances,
    )


def load_ply_file(ply_file_path: Path, center: bool = False) -> GSplatData:
    start_time = time.time()

    SH_C0 = 0.28209479177387814

    plydata = PlyData.read(ply_file_path)
    v = plydata["vertex"]
    positions = np.stack([v["x"], v["y"], v["z"]], axis=-1)
    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1))
    wxyzs = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1)
    colors = 0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
    opacities = 1.0 / (1.0 + np.exp(-v["opacity"][:, None]))

    Rs = tf.SO3(wxyzs).as_matrix()
    covariances = np.einsum(
        "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )
    if center:
        positions -= np.mean(positions, axis=0, keepdims=True)

    num_gaussians = len(v)
    print(
        f"PLY file with {num_gaussians=} loaded in {time.time() - start_time} seconds"
    )
    return GSplatData(
        centers=positions,
        rgbs=colors,
        opacities=opacities,
        covariances=covariances,
    )

# 書き出し用 from chat gpt
def _rotation_matrix_to_quat_wxyz_batch(Rs: np.ndarray) -> np.ndarray:
    """
    Rs: (N, 3, 3) rotation matrices
    return: (N, 4) quaternions [w, x, y, z]
    """
    N = Rs.shape[0]
    quats = np.empty((N, 4), dtype=np.float32)

    for i in range(N):
        R = Rs[i]
        trace = np.trace(R)
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0  # s = 4 * qw
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        else:
            if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
                qw = (R[2, 1] - R[1, 2]) / s
                qx = 0.25 * s
                qy = (R[0, 1] + R[1, 0]) / s
                qz = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
                qw = (R[0, 2] - R[2, 0]) / s
                qx = (R[0, 1] + R[1, 0]) / s
                qy = 0.25 * s
                qz = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
                qw = (R[1, 0] - R[0, 1]) / s
                qx = (R[0, 2] + R[2, 0]) / s
                qy = (R[1, 2] + R[2, 1]) / s
                qz = 0.25 * s

        q = np.array([qw, qx, qy, qz], dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-8
        quats[i] = q

    return quats


def save_ply_file(ply_file_path: Path, gs: "GSplatData") -> None:
    """
    GSplatData から load_ply_file と互換な PLY を書き出す。
    """
    start_time = time.time()

    SH_C0 = 0.28209479177387814

    centers = np.asarray(gs.centers, dtype=np.float32)          # (N,3)
    colors  = np.asarray(gs.rgbs, dtype=np.float32)             # (N,3)
    covs    = np.asarray(gs.covariances, dtype=np.float32)      # (N,3,3)
    opac    = np.asarray(gs.opacities, dtype=np.float32).reshape(-1)  # (N,)

    N = centers.shape[0]
    assert covs.shape == (N, 3, 3)

    # colors = 0.5 + SH_C0 * f_dc → f_dc に戻す
    f_dc = (colors - 0.5) / SH_C0  # (N,3)

    # opacities = sigmoid(opacity_param) の逆
    eps = 1e-6
    o_clipped = np.clip(opac, eps, 1.0 - eps)
    opacity_param = np.log(o_clipped / (1.0 - o_clipped))  # (N,)

    # covariances -> R, scales
    eigvals, eigvecs = np.linalg.eigh(covs)  # eigvecs: (N,3,3)
    eigvals = np.clip(eigvals, 1e-12, None)
    scales = np.sqrt(eigvals).astype(np.float32)  # (N,3)

    Rs = eigvecs.astype(np.float32)
    # det(R)=+1 に補正
    dets = np.linalg.det(Rs)
    neg_mask = dets < 0
    Rs[neg_mask, :, 0] *= -1.0

    # load 側で scales = exp([scale_*]) なので逆に log を取る
    scale_params = np.log(scales)

    # R -> 四元数 wxyz
    wxyzs = _rotation_matrix_to_quat_wxyz_batch(Rs)  # (N,4)

    # PLY の頂点配列を構築
    vertex_dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    vertex = np.empty(N, dtype=vertex_dtype)

    vertex["x"] = centers[:, 0]
    vertex["y"] = centers[:, 1]
    vertex["z"] = centers[:, 2]

    vertex["f_dc_0"] = f_dc[:, 0]
    vertex["f_dc_1"] = f_dc[:, 1]
    vertex["f_dc_2"] = f_dc[:, 2]

    vertex["opacity"] = opacity_param.astype(np.float32)

    vertex["scale_0"] = scale_params[:, 0]
    vertex["scale_1"] = scale_params[:, 1]
    vertex["scale_2"] = scale_params[:, 2]

    vertex["rot_0"] = wxyzs[:, 0]
    vertex["rot_1"] = wxyzs[:, 1]
    vertex["rot_2"] = wxyzs[:, 2]
    vertex["rot_3"] = wxyzs[:, 3]

    el = PlyElement.describe(vertex, "vertex")
    PlyData([el], text=False).write(ply_file_path)

    print(
        f"PLY file with num_gaussians={N} saved in {time.time() - start_time} seconds"
    )

def main(
    splat_paths: tuple[Path, ...] = (
        # Path(__file__).absolute().parent.parent / "assets" / "train.splat",
        Path(__file__).absolute().parent.parent / "assets" / "nike.splat",
    ),
) -> None:
    server = viser.ViserServer()

    for i, splat_path in enumerate(splat_paths):
        if splat_path.suffix == ".splat":
            splat_data = load_splat_file(splat_path, center=True)
        elif splat_path.suffix == ".ply":
            splat_data = load_ply_file(splat_path, center=True)
        else:
            raise SystemExit("Please provide a filepath to a .splat or .ply file.")

        server.scene.add_transform_controls(f"/{i}")
        gs_handle = server.scene.add_gaussian_splats(
            f"/{i}/gaussian_splats",
            centers=splat_data.centers,
            rgbs=splat_data.rgbs,
            opacities=splat_data.opacities,
            covariances=splat_data.covariances,
        )

        remove_button = server.gui.add_button(f"Remove splat object {i}")

        @remove_button.on_click
        def _(_, gs_handle=gs_handle, remove_button=remove_button) -> None:
            gs_handle.remove()
            remove_button.remove()

    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    tyro.cli(main)