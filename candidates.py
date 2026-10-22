from __future__ import annotations

from typing import Sequence, Union

import cv2
import numpy as np


def crop_centered_roi(image: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    """
    Crop the region of interest with the given center from the full frame. 

    Args:
        image (np.ndarray): Input frame (1 channel).
        x (int): Input point's x coordinate.
        y (int): Input point's y coordinate.
        size (int): The fixed size of the region of interest.

    Returns:
        np.ndarray: Cropped part of the frame where the further search will be performed.
    """

    h, w = image.shape[:2]
    half = size // 2

    # ensure roi fits within frame boundaries.
    # if yes => cut the rectangle that fits into the frame, not a square.
    x1 = max(x - half, 0)
    y1 = max(y - half, 0)
    x2 = min(x + half, w)
    y2 = min(y + half, h)

    return image[y1:y2, x1:x2]


def combine_boxes_and_scores(detections: list[tuple[int, int, int, int]], scores: list[float]) -> list[dict[str, Union[int, float]]]:
    """
    Function to combine boxes and their corresponding scores into dicts.

    Args:
        detections (list): Contains tuples of format (int, int, int, int) for every bounding box' 
        coordinates (x1, y1, x2, y2) in the coordinate space of the region of interest.
        scores (list): Contains the final scores for each box in the float format.

    Returns:
        list[dict[str, Union[int, float]]]: [{"x1": int, "y1": int, "x2": int, "y2": int, "score": float}].
    """

    boxes = []
    for box, score in zip(detections, scores):
        full_x1, full_y1, full_x2, full_y2 = box

        boxes.append({
            "x1": full_x1,
            "y1": full_y1,
            "x2": full_x2,
            "y2": full_y2,
            "score": score
        })

    return boxes


def contains_center(x: int, y: int, box: dict[str, int], margin: float = 0) -> bool:
    """
    Boolean check if the candidate box contains the input point or no.

    Args:
        x (int): Input point's x coordinate.
        y (int): Input point's y coordinate.
        box (dict): {"x1": int, "y1": int, "x2": int, "y2": int}
        margin (float): how far (px) outside the box the point may still be
            and count as contained.

    Returns:
        bool: True, if the box contains the input. False otherwise.
    """

    return (box["x1"] - margin <= x <= box["x2"] + margin
            and box["y1"] - margin <= y <= box["y2"] + margin)


def get_zd_score(box_edgemap: np.ndarray) -> float:
    """
    Compute the zd score, inspired by the paper "Edge Boxes:Locating Object Proposals from Edges"
    by Lawrence Zitnick and Piotr Dollar.

    Args:
        box_edgemap (np.ndarray): Edgemap of the proposed candidate.

    Returns:
        float: The zd score.
    """

    if box_edgemap.size == 0:
        return 0.0

    # the quantity of the strong edge pixels in the candidate box's edgemap
    internal_strength = np.sum(box_edgemap > 0)

    h, w = box_edgemap.shape
    border_mask = np.zeros_like(box_edgemap, dtype=np.uint8)
    # 5% of the minimal shape of the box will be taken as the "border"
    border_width = max(1, int(min(w, h) * 0.05))

    border_mask[:border_width, :] = 1
    border_mask[-border_width:, :] = 1
    border_mask[:, :border_width] = 1
    border_mask[:, -border_width:] = 1

    # if some edges intersect the border of the bounding box => probably the box doesn't contain
    # the full object => gets penalized 
    border_penalty = np.sum((box_edgemap > 0) & (border_mask > 0))
    
    # the full score is being computed by the strong edges as a positive score and penalty for the
    # intersection...
    score = internal_strength - border_penalty
    # and normalized
    normalized_score = score / (w * h)

    # returns the zd score
    # makes sure a big negative zd score won't collapse the final score
    return max(normalized_score, -2.0)
    

def get_all_candidates(roi: np.ndarray, roi_size: int, around_center: dict[str, int], reach: float = 0) -> list[dict[str, int]]:
    """
    Get all candidates from the region of interest.

    Args:
        roi (np.ndarray): Region of interest (grayscaled, blurred and cut).
        roi_size (int): Size of the region of interest.
        around_center (dict): {"x": int, "y": int} Coordinates to center the search around.
        reach (float): a box is kept if it contains the center point or lies
            within this many pixels of it.

    Returns:
        list[dict]: List of candidate dictionaries.
    """

    # compute dynamic thresholds for Canny edge detection  
    median = np.median(roi)
    low = int(max(0, 0.66 * median))
    high = int(min(255, 1.33 * median))

    # if the median is close to 0, makes sure to get different thresholds
    if high - low < 20:
        low = 50
        high = 150

    # edge detection
    edges = cv2.Canny(roi, low, high, apertureSize=3, L2gradient=True)

    # save original edgemap for zd score computations
    canny = edges

    # morphological operations for smoothening broken edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)

    # extract ONLY OUTER contours (with cv2.RETR_EXTERNAL)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # if nothing was found => roi doesn't contain full objects
    if hierarchy is None:
        return []

    boxes = []
    scores_raw = []

    # center coordinates in the full frame coordinate system
    x_center = around_center['x']
    y_center = around_center['y']

    # get the roi's top left point's coordinates in the full frame system 
    half = roi_size // 2
    top_left_x = max(x_center - half, 0)
    top_left_y = max(y_center - half, 0)

    for j, cnt in enumerate(contours):
        # get the bounding box for each outer contour
        x, y, w, h = cv2.boundingRect(cnt)

        # map the bbox's coordinates from roi's system into full frame's system
        full_x1 = top_left_x + x
        full_y1 = top_left_y + y
        full_x2 = full_x1 + w
        full_y2 = full_y1 + h

        if not contains_center(x_center, y_center, {"x1": full_x1, "y1": full_y1, "x2": full_x2, 
                                                    "y2": full_y2}, reach):
            continue

        # parameters for computing the solidity score
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)

        # computes the solidity score
        solidity = float(area) / hull_area if hull_area != 0 else 0

        # computes the sharpness score (can be removed if the speed is low)
        lap = cv2.Laplacian(roi[y:y+h, x:x+w], cv2.CV_64F)
        sharpness = np.log1p(lap.var()) / np.log1p(10000)

        # computes the distance from the input point
        distance = np.sqrt(
            (int((full_x1 + full_x2) // 2) - x_center) ** 2 +
            (int((full_y1 + full_y2) // 2) - y_center) ** 2
        )

        # computes the zd score
        zd = get_zd_score(canny[y:y+h, x:x+w])

        # save the box and every score in a tuple format
        boxes.append((full_x1, full_y1, full_x2, full_y2))
        scores_raw.append((solidity, sharpness, distance, zd))

    if not boxes:
        return []

    # gets the maximum distance from the input point for further normalization
    max_dist = max(s[2] for s in scores_raw) or 1
        
    # solidity weight
    alpha = 0.8
    # sharpness weight
    beta = 0.2
    # distance penalty weight
    gamma = 0.4
    # zd weight
    delta = 0.5

    final_scores = []

    for solidity, sharpness, dist, zd in scores_raw:
        # normalized the distance
        norm_dist = dist / max_dist
        
        # computes the weighted sum using every score
        # (distance is a penalty => negative coefficient)
        combined_score = alpha * solidity + beta * sharpness - gamma * (norm_dist ** 2) + delta * zd
        
        final_scores.append(combined_score)

    return combine_boxes_and_scores(boxes, final_scores)


def box_iou(box_a: dict[str, int], box_b: dict[str, int]) -> float:
    """
    Intersection-over-union of two candidate boxes.
    """

    ix1 = max(box_a["x1"], box_b["x1"])
    iy1 = max(box_a["y1"], box_b["y1"])
    ix2 = min(box_a["x2"], box_b["x2"])
    iy2 = min(box_a["y2"], box_b["y2"])

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0

    area_a = (box_a["x2"] - box_a["x1"]) * (box_a["y2"] - box_a["y1"])
    area_b = (box_b["x2"] - box_b["x1"]) * (box_b["y2"] - box_b["y1"])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


# Candidates scoring below this are considered too weak to be useful and get
# dropped, regardless of how they rank against the others.
MIN_CANDIDATE_SCORE = 0.15

# When a smaller box overlaps a bigger one past this IoU, it's treated as the
# same object found at a smaller scale, not a distinct candidate.
CONTAINMENT_IOU_THRESHOLD = 0.8


def non_max_suppression(candidates: list[dict[str, Union[int, float]]], iou_threshold: float = 0.5) -> list[dict[str, Union[int, float]]]:
    """
    Greedily keep the highest-scoring candidates, dropping any later candidate
    that overlaps an already-kept one past `iou_threshold`. The same ROI
    searched at different sizes tends to rediscover the same object as
    near-duplicate boxes, so this keeps the top-n genuinely distinct.
    """

    ordered = sorted(candidates, key=lambda c: c["score"], reverse=True)
    kept: list[dict[str, Union[int, float]]] = []

    for candidate in ordered:
        if all(box_iou(candidate, k) < iou_threshold for k in kept):
            kept.append(candidate)

    return kept


def box_area(box: dict[str, int]) -> int:
    return max(0, box["x2"] - box["x1"]) * max(0, box["y2"] - box["y1"])


def remove_contained_boxes(candidates: list[dict[str, Union[int, float]]], iou_threshold: float = CONTAINMENT_IOU_THRESHOLD) -> list[dict[str, Union[int, float]]]:
    """
    When a smaller box overlaps a bigger one past `iou_threshold`, drop the
    smaller one and keep only the bigger box, regardless of their scores.
    Unlike `non_max_suppression` (which keeps the higher-scoring box of an
    overlapping pair), this always prefers size.
    """

    ordered = sorted(candidates, key=box_area, reverse=True)
    kept: list[dict[str, Union[int, float]]] = []

    for candidate in ordered:
        if all(box_iou(candidate, k) < iou_threshold for k in kept):
            kept.append(candidate)

    return kept


# Candidates whose color variation (see `color_variation`) falls outside
# [MIN, MAX] are dropped: below MIN is a near-flat patch (sky, water, ground),
# above MAX is busy texture (grass, waves, clutter). Lab units, 0 = perfectly
# flat color. Tune on real footage; None disables that side.
MIN_COLOR_VARIATION = 8.0
MAX_COLOR_VARIATION = None

# Fraction of the box trimmed from each side before measuring color, so the
# background caught in the box corners doesn't count as variation.
COLOR_INNER_MARGIN = 0.2


def color_variation(frame: np.ndarray, box: dict[str, Union[int, float]]) -> float:
    """
    How much the color varies inside a box: the combined standard deviation of
    the L, a and b channels (CIE Lab, so distances track perceived color) over
    the box's inner area. 0 means a perfectly flat color.
    """

    mx = int((box["x2"] - box["x1"]) * COLOR_INNER_MARGIN)
    my = int((box["y2"] - box["y1"]) * COLOR_INNER_MARGIN)
    patch = frame[box["y1"] + my:box["y2"] - my, box["x1"] + mx:box["x2"] - mx]
    if patch.size == 0:
        return float("inf")

    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    return float(np.sqrt(lab.var(axis=0).sum()))


def filter_by_color_variation(frame: np.ndarray, candidates: list[dict[str, Union[int, float]]], min_variation: float | None = MIN_COLOR_VARIATION, max_variation: float | None = MAX_COLOR_VARIATION) -> list[dict[str, Union[int, float]]]:
    """
    Keep candidates whose color variation lies within [min_variation,
    max_variation]; None leaves that side open. Each candidate gets its
    measured "color_variation" added, for tuning.
    """

    if min_variation is None and max_variation is None:
        return candidates

    for c in candidates:
        c["color_variation"] = color_variation(frame, c)
    return [
        c for c in candidates
        if (min_variation is None or c["color_variation"] >= min_variation)
        and (max_variation is None or c["color_variation"] <= max_variation)
    ]


# Weight of the penalty for a candidate's center being far from the clicked
# point (0 disables it). Same quadratic shape as the per-ROI distance penalty.
CLICK_DISTANCE_WEIGHT = 0.4


def penalize_click_distance(candidates: list[dict[str, Union[int, float]]], x: int, y: int, max_dist: float, weight: float = CLICK_DISTANCE_WEIGHT) -> None:
    """
    Subtract `weight * (dist / max_dist) ** 2` from each candidate's score,
    where dist is its box center's distance to (x, y); max_dist is the
    distance at which the full penalty applies. Adds "click_distance" (px)
    to each candidate and edits them in place.
    """

    for c in candidates:
        dist = float(np.hypot((c["x1"] + c["x2"]) / 2 - x, (c["y1"] + c["y2"]) / 2 - y))
        c["click_distance"] = dist
        c["score"] -= weight * min(dist / max_dist, 1.0) ** 2


def _prepare_gray(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


# Side lengths (pixels) of the square ROIs searched around each point. The
# largest one is the most space a single point (e.g. each grid point) gets.
# DEFAULT_ROI_SIZES = tuple(range(20, 100, 10))
DEFAULT_ROI_SIZES = tuple((50, 100, 150))


def _all_candidates_at_point(gray: np.ndarray, x: int, y: int, roi_sizes: Sequence[int] = DEFAULT_ROI_SIZES, reach: float = 0) -> list[dict[str, Union[int, float]]]:
    """
    Pool every candidate found around (x, y) across all `roi_sizes` (no
    de-duplication yet).
    """

    all_candidates = []
    for size in roi_sizes:
        roi = crop_centered_roi(gray, x, y, size)
        all_candidates.extend(get_all_candidates(roi, size, {"x": x, "y": y}, reach))
    return all_candidates


def top_n_candidates(x: int, y: int, frame: np.ndarray, n: int = 5, iou_threshold: float = 0.5, min_score: float = MIN_CANDIDATE_SCORE, roi_sizes: Sequence[int] = DEFAULT_ROI_SIZES, min_color_variation: float | None = MIN_COLOR_VARIATION, max_color_variation: float | None = MAX_COLOR_VARIATION) -> list[dict[str, Union[int, float]]]:
    """
    Find the top n candidates around the given point, pooling candidates
    found across every ROI size and de-duplicating overlapping boxes.

    Args:
        x (int): center's x coordinate.
        y (int): center's y coordinate.
        frame (np.ndarray): full frame
        n (int): how many candidates to return, highest score first.
        iou_threshold (float): overlap above which a lower-scoring candidate
            is treated as a duplicate of a kept one and dropped.
        min_score (float): candidates scoring below this are dropped.
        roi_sizes (Sequence[int]): square ROI side lengths searched around the point.
        min_color_variation (float | None): candidates with less color variation
            than this are dropped before ranking; None leaves it open.
        max_color_variation (float | None): same for more variation than this.

    Returns:
        list[dict]: Up to n candidate dictionaries, highest score first.
        [{"x1": int, "y1": int, "x2": int, "y2": int, "score": float}]
    """

    gray = _prepare_gray(frame)
    all_candidates = filter_by_color_variation(frame, _all_candidates_at_point(gray, x, y, roi_sizes), min_color_variation, max_color_variation)

    if not all_candidates:
        return []

    deduped = non_max_suppression(all_candidates, iou_threshold)
    deduped = remove_contained_boxes(deduped)
    filtered = [c for c in deduped if c["score"] >= min_score]
    return filtered[:n]


def best_candidate(x: int, y: int, frame: np.ndarray) -> dict[str, int]:
    """
    Find the best (with the highest combined score) candidate to the given coordinates.

    Args:
        x (int): center's x coordinate.
        y (int): center's y coordinate.
        frame (np.ndarray): full frame

    Returns:
        dict: The closest candidate dictionary.
        {"x1": int, "y1": int, "x2": int, "y2": int}
    """

    top = top_n_candidates(x, y, frame, n=1)
    if not top:
        return {"x1": -1, "y1": -1, "x2": -1, "y2": -1}

    return {"x1": top[0]["x1"], "y1": top[0]["y1"], "x2": top[0]["x2"], "y2": top[0]["y2"]}


# Fixed side length (in pixels) of the square sampled around a clicked point.
GRID_REGION_SIZE = 200

# How far (px) from a grid point a box may lie and still be accepted by that
# point. Bigger than the grid spacing (region / (cols - 1), ~33 px for 49 points
# over 200 px) means neighboring points' reach overlaps, so no gap is missed.
POINT_REACH = 30

# How many candidates grid_candidates returns at most, highest score first.
MAX_RESULTS = 10


def grid_points(x: int, y: int, n: int, region_size: int, width: int, height: int) -> list[tuple[int, int]]:
    """
    n points laid out on an even grid covering a `region_size` x `region_size`
    square centered on (x, y), clamped into the frame bounds. If n isn't a
    perfect square, the grid is sized to just cover it (cols * rows >= n)
    and trimmed down to exactly n points.
    """

    half = region_size / 2.0
    cols = max(1, int(round(np.sqrt(n))))
    rows = int(np.ceil(n / cols))

    xs = np.linspace(x - half, x + half, cols)
    ys = np.linspace(y - half, y + half, rows)

    points = [(int(round(px)), int(round(py))) for py in ys for px in xs][:n]

    return [
        (min(max(px, 0), width - 1), min(max(py, 0), height - 1))
        for px, py in points
    ]


def grid_candidates(x: int, y: int, frame: np.ndarray, n: int, region_size: int = GRID_REGION_SIZE, iou_threshold: float = 0.5, min_score: float = MIN_CANDIDATE_SCORE, max_results: int = MAX_RESULTS, roi_sizes: Sequence[int] = DEFAULT_ROI_SIZES, min_color_variation: float | None = MIN_COLOR_VARIATION, max_color_variation: float | None = MAX_COLOR_VARIATION, click_distance_weight: float = CLICK_DISTANCE_WEIGHT, point_reach: float = POINT_REACH) -> list[dict[str, Union[int, float]]]:
    """
    Sample n points on an even grid covering a `region_size` x `region_size`
    square centered on (x, y), run the single-point best-candidate search
    (same logic as `best_candidate`) at each sampled point, and return the
    de-duplicated, score-filtered results.

    Args:
        x (int): click's x coordinate.
        y (int): click's y coordinate.
        frame (np.ndarray): full frame.
        n (int): number of grid sample points searched.
        region_size (int): side length (pixels) of the square sampled around
            (x, y).
        iou_threshold (float): overlap above which a lower-scoring candidate
            is treated as a duplicate of a kept one and dropped.
        min_score (float): candidates scoring below this are dropped.
        max_results (int): at most this many candidates are returned,
            highest score first.
        roi_sizes (Sequence[int]): square ROI side lengths searched around each
            grid point; the largest is the space each point gets.
        min_color_variation (float | None): candidates with less color variation
            than this are dropped before each point picks its best; None
            leaves it open.
        max_color_variation (float | None): same for more variation than this.
        click_distance_weight (float): how strongly a candidate is penalized
            for its center being far from (x, y), the clicked point; 0 disables.
        point_reach (float): a grid point also accepts boxes within this many
            pixels of it, not only boxes containing it. Set above the grid
            spacing so small objects between points are still found.

    Returns:
        list[dict]: Up to max_results candidate dictionaries, highest score
        first. [{"x1": int, "y1": int, "x2": int, "y2": int, "score": float}]
    """

    gray = _prepare_gray(frame)
    height, width = gray.shape[:2]

    # a candidate at the far corner of the sampled region gets the full penalty
    max_dist = region_size * np.sqrt(2) / 2

    results = []
    for px, py in grid_points(x, y, n, region_size, width, height):
        candidates = filter_by_color_variation(frame, _all_candidates_at_point(gray, px, py, roi_sizes, point_reach), min_color_variation, max_color_variation)
        penalize_click_distance(candidates, x, y, max_dist, click_distance_weight)
        if candidates:
            results.append(max(candidates, key=lambda c: c["score"]))

    deduped = non_max_suppression(results, iou_threshold)
    deduped = remove_contained_boxes(deduped)
    filtered = [c for c in deduped if c["score"] >= min_score]
    return filtered[:max_results]