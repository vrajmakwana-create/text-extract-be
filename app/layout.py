"""Reading-order helpers for multi-column pages."""


def reading_order(items: list, page_width: float) -> list:
    """Order (bbox, payload) items the way a person reads the page.

    Human order for a typical textbook page:
      1. Full-width bands (titles, tables) top to bottom as they appear
      2. Between those bands: entire left column, then entire right column
      3. Within a column: top to bottom

    Wide items (tables, captions) are treated as spanning bands. They must not
    be used when hunting for the column gutter, or they bridge the gap and the
    page falls back to a broken left-to-right weave.
    """
    if len(items) < 2:
        return list(items)

    page_left = min(item[0][0] for item in items)
    page_right = max(item[0][2] for item in items)
    region_width = max(page_right - page_left, 1.0)
    mid = (page_left + page_right) / 2

    spanning = [item for item in items if _is_spanning(item[0], mid, region_width)]
    column_items = [item for item in items if not _is_spanning(item[0], mid, region_width)]

    if not spanning:
        columns = _split_column_items(column_items, page_width, region_width)
        if columns:
            left, right = columns
            return _stack_column(left) + _stack_column(right)
        return _stack_column(items)

    # Spanning bands split the page into vertical slices. Read each slice in
    # order: spanning band, then the two-column body under/between them.
    spanning_ids = {id(item) for item in spanning}
    parts: list[list] = []
    run: list = []
    for item in sorted(items, key=lambda entry: (entry[0][1], entry[0][0])):
        if id(item) in spanning_ids:
            if run:
                parts.append(run)
                run = []
            parts.append([item])
        else:
            run.append(item)
    if run:
        parts.append(run)

    ordered: list = []
    for part in parts:
        if len(part) == 1 and id(part[0]) in spanning_ids:
            ordered.append(part[0])
            continue
        columns = _split_column_items(part, page_width, region_width)
        if columns:
            left, right = columns
            ordered.extend(_stack_column(left))
            ordered.extend(_stack_column(right))
        else:
            ordered.extend(_stack_column(part))
    return ordered


def _is_spanning(bbox, mid: float, region_width: float) -> bool:
    """True when a box is a full-width band, not a single column cell."""
    width = bbox[2] - bbox[0]
    if width >= region_width * 0.55:
        return True
    # Crosses the page midline with real content on both sides.
    left_part = mid - bbox[0]
    right_part = bbox[2] - mid
    return left_part > region_width * 0.15 and right_part > region_width * 0.15


def _split_column_items(items: list, page_width: float, region_width: float):
    """Split non-spanning items into left and right columns."""
    if len(items) < 2:
        return None

    # Prefer a gap between x-centres: robust when boxes nearly touch.
    by_center = sorted(
        (((item[0][0] + item[0][2]) / 2, item) for item in items),
        key=lambda entry: entry[0],
    )
    min_center_gap = max(36.0, page_width * 0.06, region_width * 0.08)
    best = None
    for index in range(len(by_center) - 1):
        gap = by_center[index + 1][0] - by_center[index][0]
        if gap < min_center_gap:
            continue
        left = [entry[1] for entry in by_center[: index + 1]]
        right = [entry[1] for entry in by_center[index + 1 :]]
        if not left or not right:
            continue
        # Both sides should own a meaningful share of the page width.
        if _width(left) < region_width * 0.12 or _width(right) < region_width * 0.12:
            continue
        score = gap
        if best is None or score > best[0]:
            best = (score, left, right)
    if best is not None:
        return best[1], best[2]

    # Fallback: classic empty-gutter scan on box edges.
    return split_columns(
        items,
        min_gap=max(12.0, page_width * 0.02),
        min_extent=max(24.0, page_width * 0.04),
    )


def _stack_column(items: list) -> list:
    """Top-to-bottom within one column (stable left tie-break)."""
    return sorted(items, key=lambda item: (round(item[0][1], 1), item[0][0]))


def split_columns(items: list, min_gap: float, min_extent: float):
    """Split items into a left and a right group at the first usable gutter.

    Returns (left, right), or None when there is no gap of at least min_gap
    that leaves at least min_extent of content on both sides.
    """
    if len(items) < 2:
        return None

    ordered = sorted(items, key=lambda item: item[0][0])
    groups: list[list] = [[ordered[0]]]
    reach = ordered[0][0][2]
    for item in ordered[1:]:
        if item[0][0] - reach >= min_gap:
            groups.append([item])
            reach = item[0][2]
        else:
            groups[-1].append(item)
            reach = max(reach, item[0][2])

    for cut in range(1, len(groups)):
        before = [item for group in groups[:cut] for item in group]
        after = [item for group in groups[cut:] for item in group]
        if _width(before) >= min_extent and _width(after) >= min_extent:
            return before, after
    return None


def _width(items: list) -> float:
    return max(item[0][2] for item in items) - min(item[0][0] for item in items)


def join_lines(items: list, row_tolerance: float = 12.0) -> str:
    """Join already reading-ordered (bbox, text) items into lines of text."""
    if not items:
        return ""

    rows: list[tuple[float, list[tuple[float, str]]]] = []
    for bbox, text in items:
        y = float(bbox[1])
        x = float(bbox[0])
        if rows and abs(y - rows[-1][0]) <= row_tolerance:
            rows[-1][1].append((x, text))
        else:
            rows.append((y, [(x, text)]))

    lines = []
    for _, parts in rows:
        parts.sort()
        lines.append(" ".join(part for _, part in parts))
    return "\n".join(lines)


def word_line_chunks(
    words: list,
    page_width: float,
    row_tolerance: float = 4.0,
) -> list[tuple[tuple[float, float, float, float], str]]:
    """Build column-aware text chunks from PyMuPDF word tuples.

    Each word is (x0, y0, x1, y1, word, block, line, word_no). Words on the
    same baseline that sit on opposite sides of a gutter become separate
    chunks, even when the PDF stored them as one visual line.
    """
    if not words:
        return []

    rows: list[tuple[float, list]] = []
    for word in sorted(words, key=lambda entry: (entry[1], entry[0])):
        y0 = float(word[1])
        if rows and abs(y0 - rows[-1][0]) <= row_tolerance:
            rows[-1][1].append(word)
        else:
            rows.append((y0, [word]))

    min_gap = max(18.0, page_width * 0.04)
    chunks: list[tuple[tuple[float, float, float, float], str]] = []
    for _, row in rows:
        row = sorted(row, key=lambda entry: entry[0])
        groups: list[list] = [[row[0]]]
        reach = float(row[0][2])
        for word in row[1:]:
            if float(word[0]) - reach >= min_gap:
                groups.append([word])
            else:
                groups[-1].append(word)
            reach = max(reach, float(word[2]))

        for group in groups:
            text = " ".join(str(word[4]) for word in group).strip()
            if not text:
                continue
            x0 = min(float(word[0]) for word in group)
            y0 = min(float(word[1]) for word in group)
            x1 = max(float(word[2]) for word in group)
            y1 = max(float(word[3]) for word in group)
            chunks.append(((x0, y0, x1, y1), text))
    return chunks


def find_image_gutter(image, min_gap_ratio: float = 0.03) -> int | None:
    """Find a vertical whitespace gutter that splits a two-column page image.

    Returns the x pixel at the centre of the gutter, or None when the page
    does not look like two columns.
    """
    gray = image.convert("L")
    width, height = gray.size
    if width < 80 or height < 80:
        return None

    pixels = gray.load()
    # Ignore thin top/bottom bands so headers and footers do not hide the body
    # gutter, and ignore outer margins where page edges are empty.
    y0 = int(height * 0.08)
    y1 = int(height * 0.92)
    x0 = int(width * 0.15)
    x1 = int(width * 0.85)
    if y1 <= y0 or x1 <= x0:
        return None

    ink = [0] * width
    for y in range(y0, y1):
        for x in range(x0, x1):
            # Text and drawings are darker than the paper.
            if pixels[x, y] < 200:
                ink[x] += 1

    band = ink[x0:x1]
    if not band:
        return None

    # A gutter is a run of near-empty columns near the page centre.
    threshold = max(2, int((y1 - y0) * 0.01))
    min_gap = max(8, int(width * min_gap_ratio))
    best = None
    start = None
    for index, value in enumerate(band):
        x = x0 + index
        if value <= threshold:
            if start is None:
                start = x
        elif start is not None:
            gap = x - start
            if gap >= min_gap:
                center = (start + x - 1) // 2
                score = gap
                # Prefer gutters close to the horizontal centre.
                score -= abs(center - width / 2) * 0.25
                if best is None or score > best[0]:
                    best = (score, center, start, x - 1)
            start = None
    if start is not None:
        gap = x1 - start
        if gap >= min_gap:
            center = (start + x1 - 1) // 2
            score = gap - abs(center - width / 2) * 0.25
            if best is None or score > best[0]:
                best = (score, center, start, x1 - 1)

    if best is None:
        return None

    _, center, gap_start, gap_end = best
    # Require real content on both sides of the gutter.
    left_ink = sum(ink[x0:gap_start])
    right_ink = sum(ink[gap_end + 1 : x1])
    total_ink = left_ink + right_ink
    if total_ink <= 0:
        return None
    if left_ink < total_ink * 0.2 or right_ink < total_ink * 0.2:
        return None
    return center
