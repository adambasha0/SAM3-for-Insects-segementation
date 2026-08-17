"""Tiling and polygon geometry — no model, no weights, no GPU."""
import numpy as np
import pytest

from sam3_insect.inference import (
    build_cfg,
    calculate_dynamic_tolerance,
    calculate_pyramid_scales,
    calculate_tile_offsets,
    check_min_mask_area,
    compute_ios_matrix,
    equal_allocate_overlaps,
    expand_polygon,
    filter_by_edge_margin,
    filter_by_image_boundary,
    filter_by_object_size,
    linear_interpolate_polygon,
    mask_to_polygons,
    nms_ios,
    pad_bbox,
)


class TestTiling:
    def test_single_segment_needs_no_overlap(self):
        assert equal_allocate_overlaps(1024, 1, 1024) == [0]

    def test_offsets_start_at_zero_and_cover_the_image(self):
        offsets = calculate_tile_offsets((3000, 2000), 1024, 384)
        starts = {(y, x) for _, (y, x) in offsets}
        assert (0, 0) in starts
        # the last tile must reach the far edge, or content would be missed
        assert max(x for _, x in starts) + 1024 >= 3000
        assert max(y for y, _ in starts) + 1024 >= 2000

    def test_tiles_overlap_by_at_least_the_minimum(self):
        offsets = calculate_tile_offsets((3000, 1024), 1024, 384)
        xs = sorted({x for _, (_, x) in offsets})
        gaps = [b - a for a, b in zip(xs, xs[1:])]
        assert all(1024 - g >= 384 for g in gaps), gaps

    def test_exact_tile_size_yields_one_tile(self):
        assert len(calculate_tile_offsets((1024, 1024), 1024, 384)) == 1

    def test_pyramid_ends_at_full_resolution_and_ascends(self):
        scales = calculate_pyramid_scales(4000, 3000, 1024)
        assert scales[-1] == 1.0
        assert scales == sorted(scales)
        assert all(0 < s <= 1.0 for s in scales)

    def test_small_image_gets_a_single_level(self):
        assert calculate_pyramid_scales(800, 600, 1024) == [1.0]

    def test_coarsest_level_fits_the_image_in_one_tile(self):
        scales = calculate_pyramid_scales(4000, 3000, 1024)
        assert max(4000, 3000) * min(scales) <= 1024 + 1


class TestFilters:
    def test_synthetic_edges_drop_but_real_borders_keep(self):
        boxes = np.array([[0.0, 500.0, 20.0, 560.0]])  # hugs the left edge
        # tile_x == 0: that edge is the real image border, so keep
        assert filter_by_edge_margin(boxes, 1024, 16, 0, 0, 4000, 4000)[0]
        # tile_x == 640: the edge is a cut, a neighbouring tile sees it whole
        assert not filter_by_edge_margin(boxes, 1024, 16, 640, 0, 4000, 4000)[0]

    def test_zero_margin_disables_the_boundary_filter(self):
        boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
        assert filter_by_image_boundary(boxes, 100, 100, 0).all()
        assert not filter_by_image_boundary(boxes, 100, 100, 5).any()

    def test_object_size_uses_sqrt_area(self):
        boxes = np.array([[0, 0, 10, 10], [0, 0, 100, 100]], dtype=float)
        keep = filter_by_object_size(boxes, 32, 1e8)
        assert list(keep) == [False, True]

    def test_filters_tolerate_no_boxes(self):
        empty = np.zeros((0, 4))
        assert len(filter_by_object_size(empty, 1, 2)) == 0
        assert len(filter_by_image_boundary(empty, 10, 10, 1)) == 0
        assert len(filter_by_edge_margin(empty, 1024, 16, 0, 0, 10, 10)) == 0


class TestSuppression:
    def test_ios_matrix_is_symmetric_with_zero_diagonal(self):
        boxes = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [50, 50, 60, 60]], dtype=float)
        ios = compute_ios_matrix(boxes)
        assert np.allclose(ios, ios.T)
        assert np.allclose(np.diag(ios), 0)

    def test_contained_box_scores_one_on_intersection_over_smaller(self):
        boxes = np.array([[0, 0, 100, 100], [10, 10, 20, 20]], dtype=float)
        assert compute_ios_matrix(boxes)[0, 1] == pytest.approx(1.0)

    def test_ios_nms_keeps_the_higher_score(self):
        boxes = np.array([[0, 0, 100, 100], [10, 10, 20, 20]], dtype=float)
        keep = nms_ios(boxes, np.array([0.9, 0.5]), 0.2)
        assert list(keep) == [0]

    def test_ios_nms_on_empty_input(self):
        assert len(nms_ios(np.zeros((0, 4)), np.zeros(0))) == 0


class TestMasksAndPolygons:
    def test_mask_becomes_a_polygon_in_padded_coordinates(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:48, 16:48] = 1
        polys = mask_to_polygons(mask, x_off=100, y_off=200, scale=1.0, tile_size=64,
                                use_dynamic_tolerance=False)
        assert len(polys) == 1
        xs = polys[0][0::2]
        ys = polys[0][1::2]
        # offsets applied, and the square's extent preserved
        assert min(xs) == pytest.approx(116, abs=2)
        assert min(ys) == pytest.approx(216, abs=2)
        assert max(xs) - min(xs) == pytest.approx(31, abs=2)

    def test_scale_maps_polygons_back_to_full_resolution(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:48, 16:48] = 1
        half = mask_to_polygons(mask, 0, 0, 0.5, tile_size=64, use_dynamic_tolerance=False)
        full = mask_to_polygons(mask, 0, 0, 1.0, tile_size=64, use_dynamic_tolerance=False)
        assert max(half[0][0::2]) == pytest.approx(2 * max(full[0][0::2]), rel=0.05)

    def test_empty_mask_yields_no_polygon(self):
        assert mask_to_polygons(np.zeros((32, 32), np.uint8), 0, 0, 1.0) == []

    def test_min_mask_area(self):
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[0, 0] = 1
        assert check_min_mask_area(mask, 1)
        assert not check_min_mask_area(mask, 3)

    def test_dynamic_tolerance_never_below_one(self):
        assert calculate_dynamic_tolerance(1024, 1024, 1024, 1024) >= 1.0
        assert calculate_dynamic_tolerance(64, 64, 1024, 1024) > 1.0

    def test_expand_polygon_grows_outward(self):
        square = [0, 0, 10, 0, 10, 10, 0, 10]
        grown = expand_polygon(square, 2.0)
        assert max(grown[0::2]) > 10 and min(grown[0::2]) < 0

    def test_expand_polygon_ignores_degenerate_input(self):
        assert expand_polygon([0, 0, 1, 1], 5) == [0, 0, 1, 1]

    def test_interpolation_multiplies_the_vertex_count(self):
        square = [0, 0, 10, 0, 10, 10, 0, 10]
        assert len(linear_interpolate_polygon(square, 3)) == len(square) * 4

    def test_pad_bbox_clamps_to_the_image(self):
        assert pad_bbox([5, 5, 95, 95], 10, 100, 100) == (0, 0, 100, 100)
        assert pad_bbox([5, 5, 95, 95], 0, 100, 100) == (5, 5, 95, 95)


class TestConfig:
    def test_defaults_are_copied_not_shared(self):
        a, b = build_cfg(), build_cfg()
        a["TILE_SIZE"] = 512
        assert b["TILE_SIZE"] == 1024

    def test_override_applies(self):
        assert build_cfg({"SCORE_THRESHOLD": 0.5})["SCORE_THRESHOLD"] == 0.5

    def test_typo_raises_rather_than_being_ignored(self):
        with pytest.raises(KeyError, match="SCORE_TRESHOLD"):
            build_cfg({"SCORE_TRESHOLD": 0.5})
