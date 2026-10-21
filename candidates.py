from typing import Union

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


def contains_center(x: int, y: int, box: dict[str, int]) -> bool:
    """
    Boolean check if the candidate box contains the input point or no.

    Args:
        x (int): Input point's x coordinate.
        y (int): Input point's y coordinate.
        box (dict): {"x1": int, "y1": int, "x2": int, "y2": int}

    Returns:
        bool: True, if the box contains the input. False otherwise.
    """

    return x >= box["x1"] and x <= box["x2"] and y >= box["y1"] and y <= box["y2"]


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
    

def get_all_candidates(roi: np.ndarray, roi_size: int, around_center: dict[str, int]) -> list[dict[str, int]]:
    """
    Get all candidates from the region of interest.

    Args:
        roi (np.ndarray): Region of interest (grayscaled, blurred and cut).
        roi_size (int): Size of the region of interest.
        around_center (dict): {"x": int, "y": int} Coordinates to center the search around.

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
                                                    "y2": full_y2}):
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

    best_score = -1
    best_box = {
        "x1": -1,
        "y1": -1,
        "x2": -1,
        "y2": -1
    }

    # grayscale + blur the frame ONCE
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # go through every size for roi to avoid thresholding the score
    for size in range(20, 200, 10):
        
        # crop the region of interest        
        roi = crop_centered_roi(frame, x, y, size)

        # get all candidates from that region
        candidates = get_all_candidates(roi, size, {"x": x, "y": y})

        if not candidates:
            continue

        top = max(candidates, key=lambda c: c["score"])
        if top["score"] > best_score:
            best_score = top["score"]
            best_box = {"x1": top["x1"], "y1": top["y1"], "x2": top["x2"],
                        "y2": top["y2"]}
    
    return best_box