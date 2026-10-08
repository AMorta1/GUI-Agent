"""Portable read-only label checks extracted from the archived browser preflight.

No capture, OCR sampling, API, launcher or action entry. Existing grounding is
used per real OCR element only; a successful check never authorizes execution.
"""

from gui_agent.grounding import DEFAULT_MIN_OCR_CONFIDENCE, resolve_target


def normalized_label(text):
    return " ".join(text.split()).casefold()


def aligned_lines(lines, visible):
    boxes = [line["bbox"] for line in lines]
    heights = [box[3] - box[1] for box in boxes]
    centers = [(box[0] + box[2]) / 2 for box in boxes]
    if max(heights) > 1.5 * min(heights) or max(centers) - min(centers) > 0.25 * min(heights):
        return False
    if max(box[0] for box in boxes) >= min(box[2] for box in boxes):
        return False
    for previous, following in zip(boxes, boxes[1:]):
        gap = following[1] - previous[3]
        if gap < 0 or gap > 0.5 * min(previous[3] - previous[1], following[3] - following[1]):
            return False
        left, right = max(previous[0], following[0]), min(previous[2], following[2])
        if right - left < 0.8 * min(previous[2] - previous[0], following[2] - following[0]):
            return False
        # Do not skip intervening OCR in the same column.
        if any(element not in lines and element["bbox"][0] < right and element["bbox"][2] > left
               and element["bbox"][1] < following[1] and element["bbox"][3] > previous[3]
               for element in visible):
            return False
    return True


def label_groups(visible, label):
    required = normalized_label(label).split()
    groups = []

    def extend(lines, offset):
        if offset == len(required):
            groups.append(lines)
            return
        for element in visible:
            words = normalized_label(element["text"]).split()
            if not words or words != required[offset:offset + len(words)]:
                continue
            if element in lines:
                continue
            candidate = lines + [element]
            if len(candidate) > 1 and not aligned_lines(candidate, visible):
                continue
            extend(candidate, offset + len(words))

    extend([], 0)
    return groups


def check_target(observation, record, *, label="Week4 Browser Test"):
    visible = observation.context()["elements"]
    matches = label_groups(visible, label)
    record["shortcut_check"] = {
        "checked": True, "required_text": label,
        "min_confidence": DEFAULT_MIN_OCR_CONFIDENCE,
        "exact_matches": [{"label_text": " ".join(line["text"] for line in group),
                           "elements": group} for group in matches], "match_count": len(matches),
        "visible": bool(matches), "unique": len(matches) == 1,
        "confidence_passed": len(matches) == 1 and all(line["confidence"] >= DEFAULT_MIN_OCR_CONFIDENCE for line in matches[0]),
        "multiline_constraints": {"max_height_ratio": 1.5, "max_center_span_in_min_line_heights": 0.25,
                                  "max_gap_in_min_line_heights": 0.5, "min_horizontal_overlap_ratio": 0.8,
                                  "intervening_same_column_ocr_forbidden": True},
        "agent_shortcut_selection_or_launch_verified": False,
    }
    record["grounding_check"] = {"checked": False, "passed": False, "point_used_for_execution": False,
                                "scope": "Existing per-element grounding only; no merged target",
                                "resolved_elements": []}
    if len(matches) != 1:
        raise ValueError(f"Exact visible shortcut must be unique; found {len(matches)}")
    record["grounding_check"]["checked"] = True
    try:
        for target in matches[0]:
            point = resolve_target(observation, target["target_id"], target["text"])
            record["grounding_check"]["resolved_elements"].append(
                {"target_id": target["target_id"], "text": target["text"], "readonly_preview_point": point})
    except Exception as exc:
        record["grounding_check"]["reason"] = str(exc)
        raise
    record["grounding_check"]["passed"] = True
