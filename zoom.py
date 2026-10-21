import argparse
import sys

import cv2
import numpy as np

# How far past the circle (in multiples of its radius) the warp fades out over,
# so the boundary blends smoothly instead of cutting off sharply.
FALLOFF_RATIO = 1.5

# Fixed-size border (in pixels), inset from each frame edge, that the
# rectangular zoom never touches.
RECT_ZOOM_PADDING = 0

# Fraction of the available center-to-padding distance that the flat-zoom
# rectangle covers. Fixed regardless of zoom level, so the box stays a
# constant size on screen and only the squeeze band absorbs more source
# content as zoom increases.
RECT_ZOOM_INNER_RATIO = 0.75

# Opacity of the inner-rectangle guide overlay drawn in "linear" mode.
RECT_OVERLAY_ALPHA = 0.4


def build_bulge_maps(width, height, zoom):
    """Radial remap that magnifies the central circle (radius = height / 2)
    and smoothly compresses the region around it back down to identity past
    FALLOFF_RATIO * radius, so no source content is cropped or has a hard seam.
    """
    cx, cy = width / 2.0, height / 2.0
    radius = height / 2.0
    outer_radius = radius * FALLOFF_RATIO

    ys, xs = np.indices((height, width), dtype=np.float32)
    dx = xs - cx
    dy = ys - cy
    r = np.sqrt(dx * dx + dy * dy)
    r_safe = np.maximum(r, 1e-6)

    # Exponent is `zoom` inside the circle, smoothstep-eased to 1 (no-op) by
    # outer_radius, and exactly 1 beyond it.
    t = np.clip((r - radius) / (outer_radius - radius), 0.0, 1.0)
    smooth = t * t * (3 - 2 * t)
    exponent = zoom + (1.0 - zoom) * smooth

    scale = np.power(r_safe / radius, exponent - 1.0)
    map_x = (cx + dx * scale).astype(np.float32)
    map_y = (cy + dy * scale).astype(np.float32)
    return map_x, map_y


def _axis_scale(d, inner, outer, zoom):
    """Flat uniform `1/zoom` scale (plain magnification, like a crop & resize
    zoom) for |d| <= inner, smoothstep-eased up to identity scale 1 by
    |d| == outer, and exactly 1 (untouched) beyond outer.
    """
    d_abs = np.abs(d)
    flat_scale = 1.0 / zoom
    span = max(outer - inner, 1e-6)
    t = np.clip((d_abs - inner) / span, 0.0, 1.0)
    smooth = t * t * (3 - 2 * t)
    return flat_scale + (1.0 - flat_scale) * smooth


def build_rect_zoom_maps(width, height, zoom, inner_ratio, padding=RECT_ZOOM_PADDING):
    """Flat uniform zoom (like the old crop & resize linear zoom) over a
    centered inner rectangle, with the band between that rectangle and
    `padding` pixels from each edge smoothly squeezed down to absorb the
    rest of the source frame, so nothing is cropped and the outer
    `padding`-pixel border stays completely untouched. Horizontal and
    vertical axes are handled independently, matching the frame's
    rectangular shape instead of a circle.
    """
    cx, cy = width / 2.0, height / 2.0
    outer_x = max(1.0, cx - padding)
    outer_y = max(1.0, cy - padding)
    inner_x = outer_x * inner_ratio
    inner_y = outer_y * inner_ratio

    ys, xs = np.indices((height, width), dtype=np.float32)
    dx = xs - cx
    dy = ys - cy

    scale_x = _axis_scale(dx, inner_x, outer_x, zoom)
    scale_y = _axis_scale(dy, inner_y, outer_y, zoom)

    map_x = (cx + dx * scale_x).astype(np.float32)
    map_y = (cy + dy * scale_y).astype(np.float32)
    return map_x, map_y


ZOOM_MODES = ["off", "bulge", "linear"]


def main():
    parser = argparse.ArgumentParser(description="Play a video file or camera device.")
    parser.add_argument("source", help="Path to a video file or a camera device, e.g. /dev/video0")
    args = parser.parse_args()

    # A camera device path/index needs to go to OpenCV as an int index on some platforms.
    source = args.source
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: could not open source '{args.source}'", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay_ms = max(1, int(1000 / fps))

    window = "soft_zoom"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seekable = frame_count > 0
    last_synced_pos = [-1]

    def on_seek(pos):
        # setTrackbarPos() (used to keep the slider in sync below) re-fires this
        # callback, so ignore calls that just echo our own last sync.
        if pos != last_synced_pos[0]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

    if seekable:
        cv2.createTrackbar("Position", window, 0, frame_count - 1, on_seek)

    # "Zoom" trackbar value is zoom * 100 so the slider can use plain ints;
    # 100 == 1.0x (no effect).
    zoom_slider_max = 500
    cv2.createTrackbar("Zoom", window, 100, zoom_slider_max, lambda _pos: None)

    # "Inner Ratio" trackbar value is the ratio * 100 (plain ints); only used
    # by the "linear" mode.
    cv2.createTrackbar(
        "Inner Ratio", window, int(RECT_ZOOM_INNER_RATIO * 100), 100, lambda _pos: None
    )

    bulge_cache = {"zoom_value": None, "size": None, "maps": None}
    rect_cache = {"zoom_value": None, "inner_ratio_value": None, "size": None, "maps": None}
    mode_index = ZOOM_MODES.index("linear")
    paused = False
    raw_frame = None

    try:
        while True:
            if not paused:
                ret, raw_frame = cap.read()
                if not ret:
                    break

            frame = raw_frame.copy()
            mode = ZOOM_MODES[mode_index]
            zoom_value = max(100, cv2.getTrackbarPos("Zoom", window))

            if mode != "off" and zoom_value != 100:
                zoom = zoom_value / 100.0
                frame_size = (frame.shape[1], frame.shape[0])
                if mode == "bulge":
                    if bulge_cache["zoom_value"] != zoom_value or bulge_cache["size"] != frame_size:
                        bulge_cache["maps"] = build_bulge_maps(frame_size[0], frame_size[1], zoom)
                        bulge_cache["zoom_value"] = zoom_value
                        bulge_cache["size"] = frame_size
                    map_x, map_y = bulge_cache["maps"]
                    frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)
                elif mode == "linear":
                    inner_ratio_value = cv2.getTrackbarPos("Inner Ratio", window)
                    inner_ratio = inner_ratio_value / 100.0
                    if (
                        rect_cache["zoom_value"] != zoom_value
                        or rect_cache["inner_ratio_value"] != inner_ratio_value
                        or rect_cache["size"] != frame_size
                    ):
                        rect_cache["maps"] = build_rect_zoom_maps(
                            frame_size[0], frame_size[1], zoom, inner_ratio
                        )
                        rect_cache["zoom_value"] = zoom_value
                        rect_cache["inner_ratio_value"] = inner_ratio_value
                        rect_cache["size"] = frame_size
                    map_x, map_y = rect_cache["maps"]
                    frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)

                    cx, cy = frame_size[0] / 2.0, frame_size[1] / 2.0
                    inner_x = (cx - RECT_ZOOM_PADDING) * inner_ratio
                    inner_y = (cy - RECT_ZOOM_PADDING) * inner_ratio
                    overlay = frame.copy()
                    cv2.rectangle(
                        overlay,
                        (int(cx - inner_x), int(cy - inner_y)),
                        (int(cx + inner_x), int(cy + inner_y)),
                        (0, 255, 255), 1,
                    )
                    frame = cv2.addWeighted(overlay, RECT_OVERLAY_ALPHA, frame, 1 - RECT_OVERLAY_ALPHA, 0)

            cv2.putText(
                frame, f"zoom: {mode}{' (paused)' if paused else ''}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
            )

            cv2.imshow(window, frame)
            if seekable:
                pos = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
                last_synced_pos[0] = pos
                cv2.setTrackbarPos("Position", window, pos)

            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord("q"):
                break
            if key == ord("z"):
                mode_index = (mode_index + 1) % len(ZOOM_MODES)
            if key == ord(" "):
                paused = not paused
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
