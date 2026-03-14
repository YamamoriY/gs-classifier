"""GSplatData / TrunkLocation のユニットテスト"""

import tempfile
from pathlib import Path

import numpy as np

from gs_classifier.models import GSplatData, TrunkLocation


class TestGSplatDataInit:
    def test_default_labels_are_zero(self, sample_gs):
        assert np.all(sample_gs.labels == 0)
        assert len(sample_gs.labels) == len(sample_gs.centers)

    def test_explicit_labels(self, labeled_gs):
        assert list(labeled_gs.labels) == [0, 0, 0, 1, 1]

    def test_additional_data_defaults_to_empty(self, sample_gs):
        assert sample_gs.additional_data == {}


class TestGSplatDataSaveLoad:
    def test_round_trip_npz(self, sample_gs):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.npz")
            sample_gs.save_to_npz(path)
            loaded = GSplatData.load_from_npz(path)

            np.testing.assert_array_equal(loaded.centers, sample_gs.centers)
            np.testing.assert_array_equal(loaded.rgbs, sample_gs.rgbs)
            np.testing.assert_array_equal(
                loaded.opacities, sample_gs.opacities
            )
            np.testing.assert_array_equal(
                loaded.covariances, sample_gs.covariances
            )
            np.testing.assert_array_equal(loaded.labels, sample_gs.labels)

    def test_round_trip_with_additional_data(self, sample_gs):
        sample_gs.additional_data["hags"] = np.arange(
            len(sample_gs.centers), dtype=np.float32
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.npz")
            sample_gs.save_to_npz(path)
            loaded = GSplatData.load_from_npz(path)

            assert "hags" in loaded.additional_data
            np.testing.assert_array_equal(
                loaded.additional_data["hags"],
                sample_gs.additional_data["hags"],
            )


class TestGSplatDataCopy:
    def test_copy_is_independent(self, sample_gs):
        copied = sample_gs.copy()
        copied.centers[0] = [999, 999, 999]
        assert not np.array_equal(copied.centers[0], sample_gs.centers[0])

    def test_copy_preserves_additional_data(self, sample_gs):
        sample_gs.additional_data["test"] = np.ones(len(sample_gs.centers))
        copied = sample_gs.copy()
        copied.additional_data["test"][0] = 999
        assert sample_gs.additional_data["test"][0] == 1.0


class TestGSplatDataLabels:
    def test_reset_labels(self, labeled_gs):
        labeled_gs.reset_labels()
        assert np.all(labeled_gs.labels == 0)

    def test_split_by_label(self, labeled_gs):
        splits = labeled_gs.split_by_label()
        assert len(splits) == 2
        assert len(splits[0].centers) == 3  # label 0
        assert len(splits[1].centers) == 2  # label 1

    def test_split_preserves_additional_data(self, labeled_gs):
        labeled_gs.additional_data["score"] = np.array(
            [10, 20, 30, 40, 50], dtype=np.float32
        )
        splits = labeled_gs.split_by_label()
        np.testing.assert_array_equal(
            splits[0].additional_data["score"], [10, 20, 30]
        )
        np.testing.assert_array_equal(
            splits[1].additional_data["score"], [40, 50]
        )

    def test_split_resets_new_labels_to_zero(self, labeled_gs):
        splits = labeled_gs.split_by_label()
        for s in splits:
            assert np.all(s.labels == 0)


class TestGSplatDataConcatenate:
    def test_concatenate_doubles_size(self, sample_gs):
        result = sample_gs.concatenate(sample_gs)
        assert len(result.centers) == 2 * len(sample_gs.centers)

    def test_concatenate_preserves_data(self, labeled_gs):
        splits = labeled_gs.split_by_label()
        merged = splits[0].concatenate(splits[1])
        assert len(merged.centers) == len(labeled_gs.centers)


class TestGSplatDataTransform:
    def test_identity_transform(self, sample_gs):
        original_centers = sample_gs.centers.copy()
        original_covs = sample_gs.covariances.copy()
        R = np.eye(3)
        sample_gs.coordinate_transform(R)
        np.testing.assert_allclose(sample_gs.centers, original_centers)
        np.testing.assert_allclose(sample_gs.covariances, original_covs)

    def test_90deg_rotation_z(self, sample_gs):
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        original_centers = sample_gs.centers.copy()
        sample_gs.coordinate_transform(R)
        # x -> -y, y -> x にマッピングされる (centers @ R)
        np.testing.assert_allclose(
            sample_gs.centers[:, 0], original_centers[:, 1], atol=1e-5
        )
        np.testing.assert_allclose(
            sample_gs.centers[:, 1], -original_centers[:, 0], atol=1e-5
        )


class TestTrunkLocation:
    def test_from_gs(self, labeled_gs):
        trunk_loc = TrunkLocation.from_gs(labeled_gs)
        assert trunk_loc.trunk_locations.shape == (2, 3)
        # label 0: mean of [0,0,0],[1,0,0],[2,0,0] = [1,0,0]
        np.testing.assert_allclose(trunk_loc.trunk_locations[0], [1, 0, 0])
        # label 1: mean of [10,0,0],[11,0,0] = [10.5,0,0]
        np.testing.assert_allclose(trunk_loc.trunk_locations[1], [10.5, 0, 0])

    def test_from_gs_excludes_label_minus_one(self):
        gs = GSplatData(
            centers=np.array(
                [[0, 0, 0], [1, 1, 1], [5, 5, 5]], dtype=np.float32
            ),
            rgbs=np.zeros((3, 3), dtype=np.float32),
            opacities=np.ones((3, 1), dtype=np.float32),
            covariances=np.tile(np.eye(3, dtype=np.float32), (3, 1, 1)),
            labels=np.array([-1, 0, 1]),
        )
        trunk_loc = TrunkLocation.from_gs(gs)
        assert trunk_loc.trunk_locations.shape == (2, 3)

    def test_coordinate_transform(self, labeled_gs):
        trunk_loc = TrunkLocation.from_gs(labeled_gs)
        R = np.eye(3)
        R[0, 0] = -1  # x軸を反転
        trunk_loc.coordinate_transform(R)
        np.testing.assert_allclose(trunk_loc.trunk_locations[0], [-1, 0, 0])

    def test_save_to_json(self, labeled_gs, tmp_path):
        import json

        trunk_loc = TrunkLocation.from_gs(labeled_gs)
        path = str(tmp_path / "trunks.json")
        trunk_loc.save_to_json(path)

        with open(path) as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]["name"] == "trunk_0"
        assert "x" in data[0] and "y" in data[0] and "z" in data[0]
