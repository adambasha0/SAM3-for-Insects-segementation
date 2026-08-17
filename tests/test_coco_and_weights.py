"""COCO export, rendering and the weight split/join round trip."""
import json

import pytest
from PIL import Image

from sam3_insect.coco import annotations_to_coco, empty_coco, merge_coco, save_coco
from sam3_insect.inference import PredictionResult
from sam3_insect.viz import crop_detections, render_overview
from sam3_insect.weights import join_parts, resolve_checkpoint, sha256sum, split_file


def make_result(n=2, name="img.jpg"):
    annotations = [
        {
            "bbox": [10.0 * i, 20.0 * i, 30.0, 40.0],
            "segmentation": [[0.0, 0.0, 10.0, 0.0, 10.0, 10.0]],
            "area": 1200.0,
            "score": 0.9 - 0.1 * i,
        }
        for i in range(n)
    ]
    return PredictionResult(annotations=annotations, width=640, height=480, file_name=name)


class TestCoco:
    def test_empty_coco_has_the_required_sections(self):
        coco = empty_coco()
        assert set(coco) >= {"info", "licenses", "images", "annotations", "categories"}
        assert coco["categories"][0]["name"] == "insect"

    def test_single_result_export(self):
        coco = annotations_to_coco(make_result(3))
        assert len(coco["images"]) == 1
        assert len(coco["annotations"]) == 3
        assert coco["images"][0]["width"] == 640

    def test_confidence_is_written_under_both_keys(self):
        ann = annotations_to_coco(make_result(1))["annotations"][0]
        assert ann["score"] == ann["conf"] == pytest.approx(0.9)

    def test_ids_are_unique_and_one_based(self):
        coco = annotations_to_coco([make_result(2, "a.jpg"), make_result(2, "b.jpg")])
        assert [i["id"] for i in coco["images"]] == [1, 2]
        assert [a["id"] for a in coco["annotations"]] == [1, 2, 3, 4]

    def test_annotations_point_at_the_right_image(self):
        coco = annotations_to_coco([make_result(1, "a.jpg"), make_result(1, "b.jpg")])
        by_image = {a["image_id"] for a in coco["annotations"]}
        assert by_image == {1, 2}

    def test_merge_renumbers_without_collisions(self):
        merged = merge_coco([
            annotations_to_coco(make_result(2, "a.jpg")),
            annotations_to_coco(make_result(3, "b.jpg")),
        ])
        assert len(merged["images"]) == 2
        assert len(merged["annotations"]) == 5
        assert len({a["id"] for a in merged["annotations"]}) == 5
        assert {a["image_id"] for a in merged["annotations"]} == {1, 2}

    def test_merge_of_nothing_is_still_valid_coco(self):
        assert merge_coco([])["images"] == []

    def test_saved_json_round_trips(self, tmp_path):
        path = save_coco(annotations_to_coco(make_result(2)), tmp_path / "out.json")
        assert json.loads(path.read_text())["annotations"][0]["iscrowd"] == 0

    def test_export_is_json_serialisable_with_numpy_free_floats(self):
        coco = annotations_to_coco(make_result(2))
        json.dumps(coco)  # raises if a numpy scalar slipped through


class TestRendering:
    def test_overview_keeps_the_image_size_and_mode(self):
        image = Image.new("RGB", (200, 100), "white")
        out = render_overview(image, make_result(2).annotations)
        assert out.size == (200, 100)
        assert out.mode == "RGB"

    def test_overview_draws_something(self):
        image = Image.new("RGB", (200, 100), "white")
        out = render_overview(image, make_result(1).annotations)
        assert out.convert("RGB").getcolors(maxcolors=1 << 16) is None or len(
            out.getcolors(maxcolors=1 << 16) or [1, 2]
        ) > 1

    def test_score_threshold_hides_weak_detections(self):
        image = Image.new("RGB", (200, 100), "white")
        blank = render_overview(image, [], draw_masks=False, draw_boxes=False)
        filtered = render_overview(image, make_result(2).annotations, score_threshold=1.1)
        assert filtered.tobytes() == blank.tobytes()

    def test_crops_skip_degenerate_boxes(self):
        image = Image.new("RGB", (200, 200), "white")
        anns = [{"bbox": [0, 0, 2, 2], "segmentation": [], "area": 4, "score": 1.0}]
        assert crop_detections(image, anns, min_size=8) == []
        assert len(crop_detections(image, make_result(2).annotations)) == 2


class TestWeights:
    def test_split_then_join_is_byte_identical(self, tmp_path):
        source = tmp_path / "weights.pt"
        source.write_bytes(bytes(range(256)) * 400)
        digest = sha256sum(source)

        parts = split_file(source, n_parts=3, out_dir=tmp_path / "parts")
        assert len(parts) == 3
        assert sum(p.stat().st_size for p in parts) == source.stat().st_size

        joined = join_parts(parts, tmp_path / "rebuilt.pt", expected_sha256=digest)
        assert joined.read_bytes() == source.read_bytes()

    def test_split_writes_the_checksum_file(self, tmp_path):
        source = tmp_path / "w.pt"
        source.write_bytes(b"abc" * 1000)
        split_file(source, 2, out_dir=tmp_path)
        assert (tmp_path / "w.pt.sha256").read_text().strip() == sha256sum(source)

    def test_a_corrupt_part_is_rejected(self, tmp_path):
        source = tmp_path / "w.pt"
        source.write_bytes(b"x" * 5000)
        digest = sha256sum(source)
        parts = split_file(source, 2, out_dir=tmp_path / "p")
        parts[1].write_bytes(b"y" * parts[1].stat().st_size)

        with pytest.raises(RuntimeError, match="checksum mismatch"):
            join_parts(parts, tmp_path / "bad.pt", expected_sha256=digest)
        assert not (tmp_path / "bad.pt").exists()  # the bad file is removed

    def test_an_lfs_pointer_is_diagnosed_clearly(self, tmp_path):
        pointer = tmp_path / "part-00"
        pointer.write_text(
            "version https://git-lfs.github.com/spec/v1\noid sha256:deadbeef\nsize 1\n"
        )
        with pytest.raises(RuntimeError, match="Git LFS pointer"):
            join_parts([pointer], tmp_path / "out.pt")

    def test_a_missing_part_is_named(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            join_parts([tmp_path / "nope.part-00"], tmp_path / "out.pt")

    def test_local_source_requires_an_existing_file(self, tmp_path):
        with pytest.raises(ValueError, match="requires path"):
            resolve_checkpoint("local")
        with pytest.raises(FileNotFoundError):
            resolve_checkpoint("local", path=tmp_path / "absent.pt")

    def test_unknown_source_is_rejected_by_name(self, tmp_path):
        with pytest.raises(ValueError, match="unknown source"):
            resolve_checkpoint("bittorrent", cache_dir=tmp_path)
