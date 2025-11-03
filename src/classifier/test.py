from pathlib import Path
import numpy as np
from src.lib.gsloader import load_ply_file
from src.viewer.viewer import Viewer
from src.classifier.util.pcd import *
from src.classifier.util.pdal import *
from src.viewer.util.gstypes import GSplatDataWithClass
from src.lib.ground import SegGround
import matplotlib.pyplot as plt
import time
if __name__ == "__main__":
    ply_path = Path(__file__).parent / "../../data/akan.ply"
    splat_data = load_ply_file(ply_path, center=True)
    print("load ply file")
    splat_data.print_shape()

    # 座標変換（x軸周り-90°）
    R = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0],
    ])
    splat_data.centers = splat_data.centers @ R
    splat_data.covariances = np.einsum("ij,njk,kl->nil", R.T, splat_data.covariances, R)

    seg_ground = SegGround(splat_data.centers)
    root = seg_ground.ground_heights()
    print(root.height)
    print(root.children[0].height)
    print(root.children[1].height)
    print(root.children[2].height)
    print(root.children[3].height)

    viewer = Viewer()
    # viewer.add_gsplats(
    #     gsplatData=GSplatDataWithClass(
    #         centers=splat_data.centers,
    #         rgbs=splat_data.rgbs,
    #         opacities=splat_data.opacities,
    #         covariances=splat_data.covariances,
    #     ),
    # )
    viewer.add_gsplat(splat_data, name="ground", folder_name="akan")
    viewer.server.scene.add_box(
        name="ground1",
        dimensions=(10, 10, 3),
        color=(255, 0, 0),
        position=(root.children[0].point[0], root.children[0].point[1], root.children[0].height),
    )
    viewer.server.scene.add_box(
        name="ground2",
        dimensions=(10, 10, 3),
        color=(255, 0, 0),
        position=(root.children[1].point[0], root.children[1].point[1], root.children[1].height),
    )
    viewer.server.scene.add_box(
        name="ground3",
        dimensions=(10, 10, 3),
        color=(255, 0, 0),
        position=(root.children[2].point[0], root.children[2].point[1], root.children[2].height),
    )
    viewer.server.scene.add_box(
        name="ground4",
        dimensions=(10, 10, 3),
        color=(255, 0, 0),
        position=(root.children[3].point[0], root.children[3].point[1], root.children[3].height),
    )
    viewer.run()
    exit()
    # てきとうcenter
    center = np.array([(np.max(splat_data.centers[:, 0]) + np.min(splat_data.centers[:, 0])) / 2, (np.max(splat_data.centers[:, 1]) + np.min(splat_data.centers[:, 1])) / 2, 0])
    ground_height = seg_ground.ground_height(center, 10)
    print(f"ground height: {ground_height}")

    indices = np.where((splat_data.centers[:, 2] > ground_height - 0.3) & (splat_data.centers[:, 2] < ground_height + 0.3))[0]
    under_indices = np.where(splat_data.centers[:, 2] < ground_height - 0.3)[0]
    labels = np.zeros(len(splat_data.centers), dtype=int)
    labels[:] = 2
    labels[indices] = 1
    labels[under_indices] = 0

    viewer = Viewer()
    viewer.add_gsplats(
        gsplatData=GSplatDataWithClass(
            centers=splat_data.centers,
            rgbs=splat_data.rgbs,
            opacities=splat_data.opacities,
            covariances=splat_data.covariances,
            class_ids=labels,
        ),
        name="ground",
        folder_name="akan",
    )
    viewer.run()
    exit()


    # 時間を計測
    # start_time = time.time()
    # zs = splat_data.centers[:, 2]
    # plt.hist(zs, bins=1000)
    # print(f"time: {time.time() - start_time} seconds")
    # plt.show()
    # exit()

    pcd = to_pcd(splat_data.centers)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=1000)
    )

    z_threshold = 0.5
    normals = np.asarray(pcd.normals)
    horizontal_labels = np.abs(normals[:, 2]) > z_threshold



    # lasで保存
    # save_points_las(Path(__file__).parent / "../../tmp/akan.las", splat_data.centers)

    # smrf filter
    # add_hag_dim(Path(__file__).parent / "../../tmp/akan.las", Path(__file__).parent / "../../tmp/akan_smrf.las")

    # points, las = read_points_las(Path(__file__).parent / "../../tmp/akan_smrf.las")
    # hags = las.HeightAboveGround
    # print(points.shape)
    # print(hags.shape)
    # print(hags.min(), hags.max())
    # labels = np.zeros(len(points), dtype=int)
    # labels[hags < 1.5] = 1
    # labels[hags >= 1.5] = 2
    # labels[hags >= 3.5] = 3
    viewer = Viewer()
    viewer.add_gsplats(
        gsplatData=GSplatDataWithClass(
            centers=splat_data.centers,
            rgbs=splat_data.rgbs,
            opacities=splat_data.opacities,
            covariances=splat_data.covariances,
            class_ids=horizontal_labels,
        ),
        name="forest",
        folder_name="akan",
    )
    viewer.run()