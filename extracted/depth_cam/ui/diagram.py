# ui/diagram.py
# FSM 다이어그램 패널 + 상단 우측 미니 진행바(yaw 오차, FWD 잔여시간)
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ===== 스타일 상수 =====
PANEL_BG = (25, 25, 25)
GROUP_BG = (35, 35, 35)
GROUP_BORDER = (70, 70, 70)
GROUP_ACTIVE = (60, 200, 255)

TEXT_DIM   = (165, 165, 165)
TEXT_NORM  = (230, 230, 230)
TEXT_TITLE = (210, 210, 210)

NODE_W, NODE_H = 148, 30
LEVEL_GAP      = 24
ROW_GAP        = 18
COL_GAP        = 18

PANEL_MARGIN_LR = 16
PANEL_TITLE_H   = 40
GROUP_PAD       = 14
EDGE_COLOR      = (110, 110, 110)

# ===== 진행바 유틸 =====
def _bar(img, x, y, w, h, ratio, bg=(60,60,60), fg=(40,180,255), edge=(95,95,95)):
    cv2.rectangle(img, (x, y), (x+w, y+h), bg, -1, lineType=cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x+w, y+h), edge, 1, lineType=cv2.LINE_AA)
    fill = max(0, min(w, int(round(w * ratio))))
    if fill > 0:
        cv2.rectangle(img, (x, y), (x+fill, y+h), fg, -1, lineType=cv2.LINE_AA)

def _progress_from_cmdstatus(cmd_status):
    """
    (ratio, caption) 또는 (None, None)
    - UNTIL: yaw 오차 등 (metric_name 사용)
    - TIMED: FWD 잔여시간
    """
    if cmd_status is None:
        return None, None
    # 1) progress_ratio() 우선
    try:
        if hasattr(cmd_status, "progress_ratio"):
            r = cmd_status.progress_ratio()
            if r is not None:
                r = float(max(0.0, min(1.0, r)))
                mode = getattr(cmd_status, "mode", "")
                if mode == "until":
                    metric = getattr(cmd_status, "metric_name", "") or ""
                    tgt = getattr(cmd_status, "target_value", None)
                    if tgt:
                        return r, f"{metric} {r*100:.0f}%"
                    return r, f"{metric} 진행 {r*100:.0f}%"
                elif mode == "timed":
                    return r, f"FWD 진행 {r*100:.0f}%"
                else:
                    return r, f"진행 {r*100:.0f}%"
    except Exception:
        pass
    # 2) 속성으로 복원
    try:
        mode = getattr(cmd_status, "mode", "")
        if mode == "until":
            cur = getattr(cmd_status, "current_value", None)
            tgt = getattr(cmd_status, "target_value", None)
            metric = getattr(cmd_status, "metric_name", "") or ""
            if cur is not None and tgt and tgt > 0:
                r = float(max(0.0, min(1.0, cur / tgt)))
                return r, f"{metric} {cur:.1f}/{tgt:.1f} ({r*100:.0f}%)"
        elif mode == "timed":
            t_total  = getattr(cmd_status, "t_total", None)
            t_remain = getattr(cmd_status, "t_remain", None)
            t_elapsed = getattr(cmd_status, "t_elapsed", None)
            if t_total and t_total > 0:
                if t_elapsed is not None:
                    r = float(max(0.0, min(1.0, t_elapsed / t_total)))
                    return r, f"FWD {t_total - t_elapsed:.1f}s 남음 ({r*100:.0f}%)"
                if t_remain is not None:
                    r = float(max(0.0, min(1.0, (t_total - max(0.0, t_remain)) / t_total)))
                    return r, f"FWD {t_remain:.1f}s 남음 ({r*100:.0f}%)"
            if t_remain is not None:
                return None, f"FWD {t_remain:.1f}s 남음"
    except Exception:
        pass
    return None, None

def _put_center(img, text, cx, cy, color, scale=0.48, thick=1):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    cv2.putText(img, text, (int(cx - tw/2), int(cy + th/2 - 4)),
                FONT, scale, color, thick, cv2.LINE_AA)

def _draw_box(img, x, y, w, h, color, fill=None, thick=2):
    if fill is not None:
        cv2.rectangle(img, (x, y), (x+w, y+h), fill, -1, lineType=cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x+w, y+h), color, thick, lineType=cv2.LINE_AA)

def _draw_node(img, name, x, y, active=False, dim=False):
    fill = (60, 60, 60) if dim else (42, 42, 42)
    border = GROUP_ACTIVE if active else (95, 95, 95)
    _draw_box(img, x, y, NODE_W, NODE_H, border, fill=fill, thick=2)
    _put_center(img, name, x + NODE_W//2, y + NODE_H//2,
                TEXT_NORM if not dim else TEXT_DIM, scale=0.50, thick=1)

def _draw_group(img, x, y, w, h, title, active=False):
    _draw_box(img, x, y, w, h, GROUP_ACTIVE if active else GROUP_BORDER,
              fill=GROUP_BG, thick=2)
    _put_center(img, title, x + w//2, y + 18, TEXT_TITLE, scale=0.62, thick=2)

def _arrow(img, p1, p2):
    cv2.arrowedLine(img, p1, p2, EDGE_COLOR, 1, tipLength=0.03, line_type=cv2.LINE_AA)

# ===== 계층형 레이아웃 유틸 =====
def _induced_edges(edges, allowed):
    allowed = set(allowed)
    return [(u, v) for (u, v) in edges if u in allowed and v in allowed]

def _entry_candidates(nodes, edges):
    nodes = set(nodes)
    parents = {v for (u, v) in edges if u in nodes and v in nodes}
    roots = list(nodes - parents)
    if roots:
        return roots
    indeg = {n:0 for n in nodes}
    for u, v in edges:
        if v in nodes and u in nodes:
            indeg[v] += 1
    return [min(nodes, key=lambda n: indeg.get(n, 0))]

def _levels_by_bfs(nodes, edges, entries):
    nodes = list(nodes)
    E = {n: [] for n in nodes}
    for u, v in edges: E.setdefault(u, []).append(v)

    from collections import deque
    level_of = {n: None for n in nodes}
    q = deque()
    for s in entries:
        if s in level_of and level_of[s] is None:
            level_of[s] = 0; q.append(s)
    if not q:
        n = nodes[0]; level_of[n] = 0; q.append(n)

    while q:
        u = q.popleft()
        for v in E.get(u, []):
            if v in level_of and level_of[v] is None:
                level_of[v] = level_of[u] + 1
                q.append(v)

    max_lv = max([lv for lv in level_of.values() if lv is not None] or [0])
    for n in nodes:
        if level_of[n] is None:
            max_lv += 1
            level_of[n] = max_lv

    by_level = {}
    for n, lv in level_of.items():
        by_level.setdefault(lv, []).append(n)
    for lv in by_level:
        by_level[lv].sort()
    return by_level, level_of

def _wrap_row(items, inner_w):
    max_cols = max(1, (inner_w + COL_GAP) // (NODE_W + COL_GAP))
    rows = []
    if max_cols <= 0:
        max_cols = 1
    for i in range(0, len(items), max_cols):
        rows.append(items[i:i+max_cols])
    return rows

def _row_start_x(inner_x, inner_w, cols):
    total_w = cols*NODE_W + (cols-1)*COL_GAP
    return inner_x + max(0, (inner_w - total_w)//2)

def _layout_hier(inner_x, start_y, inner_w, nodes, edges, entries):
    by_level, _ = _levels_by_bfs(nodes, edges, entries)
    pos = {}
    y = start_y

    for lv in sorted(by_level.keys()):
        names = by_level[lv]
        rows = _wrap_row(names, inner_w)

        for r, row in enumerate(rows):
            cols = len(row)
            sx = _row_start_x(inner_x, inner_w, cols)
            for i, name in enumerate(row):
                x = sx + i*(NODE_W + COL_GAP)
                pos[name] = (x, y)
            if r < len(rows)-1:
                y += NODE_H + ROW_GAP
        y += NODE_H + LEVEL_GAP

    return pos, y

def _layout_by_levels(inner_x, start_y, inner_w, levels):
    pos = {}
    y = start_y
    for row in levels:
        if not row:
            continue
        cols = len(row)
        sx = _row_start_x(inner_x, inner_w, cols)
        for i, name in enumerate(row):
            x = sx + i*(NODE_W + COL_GAP)
            pos[name] = (x, y)
        y += NODE_H + LEVEL_GAP
    return pos, y

def draw_fsm_diagram_panel(fsm, panel_size=(820, 1200), col_weights=(0.26, 0.50, 0.24)):
    """
    대주제: OUTER / ALIGN / RECOVER
    - OUTER/RECOVER: 계층형(BFS) 배치
    - ALIGN: 실제 align.py의 서브상태 이름을 그대로 사용한 명시 순서 배치
    패널 상단 우측에 yaw 오차 또는 FWD 잔여시간 진행률을 표시.

    ALIGN은 정지 추론 체크포인트와 추론 없는 개방루프 동작을 구분해 표시한다.
    """
    H, W = panel_size
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = PANEL_BG

    # 상단 타이틀
    _put_center(img, "Calibration FSM (Hierarchical + ALIGN States Synced)", W//2, 22,
                (210,210,210), scale=0.70, thick=2)

    # ===== 상단 우측: 미니 진행바 =====
    cmd_status = getattr(fsm, "cmd_status", None)
    ratio, cap = _progress_from_cmdstatus(cmd_status)
    mini_w = 320
    mini_h = 36
    mini_x = W - PANEL_MARGIN_LR - mini_w
    mini_y = 8
    cap_text = cap or "진행 정보 없음"
    cv2.putText(img, cap_text, (mini_x, mini_y + 12), FONT, 0.45, (230,230,230), 1, cv2.LINE_AA)
    _bar(img, mini_x, mini_y + 18, mini_w, 10, 0.0 if ratio is None else float(max(0.0, min(1.0, ratio))))

    # 칼럼 배치
    total_w = W - 2 * PANEL_MARGIN_LR - 2 * COL_GAP
    w_outer   = int(total_w * col_weights[0])
    w_align   = int(total_w * col_weights[1])
    w_recover = total_w - w_outer - w_align
    x_outer   = PANEL_MARGIN_LR
    x_align   = x_outer + w_outer + COL_GAP
    x_recover = x_align + w_align + COL_GAP
    top_y = PANEL_TITLE_H

    # ===== 노드 정의 =====
    OUTER_CHECKS = [
        "DETECTED", "INITIAL_FACE_STOP", "INITIAL_FACE_ROTATE",
        "INITIAL_FACE_SETTLE", "CHECK", "DONE",
    ]
    outer_nodes  = ["SEARCH"] + OUTER_CHECKS

    # align.py의 실제 상태명과 동기화
    ALIGN_CHECKS = [
        "DIST_CHECK", "POST_DISTANCE_INFER", "POST_LATERAL_INFER", "READY_TO_DONE",
    ]
    ALIGN_MOTIONS = [
        "DIST_MOVE", "SNAPSHOT_YAW_ROTATE", "LATERAL_PRE_ROTATE",
        "LATERAL_FORWARD", "LATERAL_FACE_ROTATE", "INSERT_FORWARD",
    ]
    align_nodes = ALIGN_CHECKS + ALIGN_MOTIONS

    # RECOVER (기존 정의 유지)
    RECOVER_CHECKS      = ["DECIDE_TURN", "HOLD"]
    RECOVER_ROT_INPLACE = ["RECOVER_ROTATE_LEFT", "RECOVER_ROTATE_RIGHT"]
    recover_nodes = RECOVER_CHECKS + RECOVER_ROT_INPLACE

    # 활성 플래그
    top = getattr(fsm, "state", "SEARCH")
    align_sub   = getattr(fsm, "align_sub", None) or getattr(getattr(fsm, "align", None), "sub", None)
    recover_sub = getattr(fsm, "recover_sub", None) or getattr(getattr(fsm, "recover", None), "sub", None)
    active_outer   = top in (
        "SEARCH", "DETECTED", "INITIAL_FACE_STOP", "INITIAL_FACE_ROTATE",
        "INITIAL_FACE_SETTLE", "CHECK", "DONE",
    )
    active_align   = (top == "ALIGN")
    active_recover = (top == "RECOVER")

    # ===== 엣지 정의 =====
    EDGES = [
        # OUTER
        ("SEARCH", "DETECTED"),
        ("DETECTED", "INITIAL_FACE_STOP"),
        ("INITIAL_FACE_STOP", "INITIAL_FACE_ROTATE"),
        ("INITIAL_FACE_STOP", "INITIAL_FACE_SETTLE"),
        ("INITIAL_FACE_ROTATE", "INITIAL_FACE_SETTLE"),
        ("INITIAL_FACE_SETTLE", "DIST_CHECK"),

        ("HOLD", "CHECK"),
        ("CHECK", "DONE"),
        ("CHECK", "POST_LATERAL_INFER"),

        # ALIGN: 정지 추론 → 저장값 기반 이동 → 정지 추론
        ("DIST_CHECK", "DIST_MOVE"),
        ("DIST_CHECK", "POST_DISTANCE_INFER"),
        ("DIST_MOVE", "POST_DISTANCE_INFER"),

        # 거리 이동 후 단일 스냅샷으로 yaw/lateral 체인 결정
        ("POST_DISTANCE_INFER", "SNAPSHOT_YAW_ROTATE"),
        ("POST_DISTANCE_INFER", "LATERAL_PRE_ROTATE"),
        ("POST_DISTANCE_INFER", "INSERT_FORWARD"),
        ("SNAPSHOT_YAW_ROTATE", "POST_LATERAL_INFER"),
        ("LATERAL_PRE_ROTATE", "LATERAL_FORWARD"),
        ("LATERAL_FORWARD", "LATERAL_FACE_ROTATE"),
        ("LATERAL_FACE_ROTATE", "POST_LATERAL_INFER"),

        # 전면 복귀 후 1회 검증; 오차가 남으면 새 스냅샷으로 반복
        ("POST_LATERAL_INFER", "SNAPSHOT_YAW_ROTATE"),
        ("POST_LATERAL_INFER", "LATERAL_PRE_ROTATE"),
        ("POST_LATERAL_INFER", "INSERT_FORWARD"),
        ("INSERT_FORWARD", "READY_TO_DONE"),

        # DONE
        ("READY_TO_DONE", "DONE"),

        # RECOVER
        ("DECIDE_TURN", "RECOVER_ROTATE_LEFT"),
        ("DECIDE_TURN", "RECOVER_ROTATE_RIGHT"),
        ("DECIDE_TURN", "HOLD"),
    ]

    # ===== 그룹별 레이아웃 =====
    # OUTER: BFS
    inner_x_outer  = x_outer + GROUP_PAD
    inner_w_outer  = w_outer - 2 * GROUP_PAD
    start_y_outer  = top_y + GROUP_PAD + 18

    outer_edges_in = _induced_edges(EDGES, outer_nodes)
    outer_entries  = _entry_candidates(outer_nodes, outer_edges_in)
    pos_outer, y_out_last = _layout_hier(inner_x_outer, start_y_outer, inner_w_outer,
                                         outer_nodes, outer_edges_in, outer_entries)
    gy_start_outer = top_y
    gy_end_outer   = y_out_last + GROUP_PAD

    # ALIGN: 명시 순서
    inner_x_align  = x_align + GROUP_PAD
    inner_w_align  = w_align - 2 * GROUP_PAD
    start_y_align  = top_y + GROUP_PAD + 18

    align_levels_ordered = [
        ["DIST_CHECK"],
        ["DIST_MOVE"],
        ["POST_DISTANCE_INFER"],
        ["SNAPSHOT_YAW_ROTATE", "LATERAL_PRE_ROTATE"],
        ["LATERAL_FORWARD"],
        ["LATERAL_FACE_ROTATE"],
        ["POST_LATERAL_INFER"],
        ["INSERT_FORWARD"],
        ["READY_TO_DONE"],
    ]
    # 누락 보호
    flat = [n for row in align_levels_ordered for n in row]
    remain = [n for n in align_nodes if n not in flat]
    if remain:
        align_levels_ordered.append(remain)

    pos_align, y_align_last = _layout_by_levels(inner_x_align, start_y_align, inner_w_align,
                                                align_levels_ordered)
    gy_start_align = top_y
    gy_end_align   = y_align_last + GROUP_PAD

    # RECOVER: BFS
    inner_x_recover  = x_recover + GROUP_PAD
    inner_w_recover  = w_recover - 2 * GROUP_PAD
    start_y_recover  = top_y + GROUP_PAD + 18

    recover_edges_in = _induced_edges(EDGES, recover_nodes)
    recover_entries  = ["DECIDE_TURN"] if "DECIDE_TURN" in recover_nodes else _entry_candidates(recover_nodes, recover_edges_in)
    pos_recover, y_recover_last = _layout_hier(inner_x_recover, start_y_recover, inner_w_recover,
                                               recover_nodes, recover_edges_in, recover_entries)
    gy_start_recover = top_y
    gy_end_recover   = y_recover_last + GROUP_PAD

    # 패널 세로 크기 자동 확장
    needed_H = max(gy_end_outer, gy_end_align, gy_end_recover) + GROUP_PAD
    if needed_H > H:
        pad = needed_H - H
        pad_img = np.zeros((pad, W, 3), dtype=np.uint8); pad_img[:] = PANEL_BG
        img = np.vstack([img, pad_img])
        H = needed_H

    # 그룹 박스
    _draw_group(img, x_outer,   gy_start_outer,   w_outer,   gy_end_outer   - gy_start_outer,   "OUTER",   active=active_outer)
    _draw_group(img, x_align,   gy_start_align,   w_align,   gy_end_align   - gy_start_align,   "ALIGN",   active=active_align)
    _draw_group(img, x_recover, gy_start_recover, w_recover, gy_end_recover - gy_start_recover, "RECOVER", active=active_recover)

    # 포지션 병합
    POS = {}
    POS.update(pos_outer)
    POS.update(pos_align)
    POS.update(pos_recover)

    # 엣지 그리기(센터-센터)
    def C(name):
        x, y = POS[name]
        return (x + NODE_W//2, y + NODE_H//2)

    for u, v in EDGES:
        if u in POS and v in POS:
            _arrow(img, C(u), C(v))

    # 노드 그리기(활성/디밍)
    for name, (x, y) in POS.items():
        active = False
        dim = False

        if name in outer_nodes:
            active = (top == name)
        elif name in align_nodes:
            if active_align and (align_sub == name):
                active = True
            elif not active_align:
                dim = True
        elif name in recover_nodes:
            if active_recover and (recover_sub == name):
                active = True
            elif not active_recover:
                dim = True

        _draw_node(img, name, x, y, active=active, dim=dim)

    # 하단 상태 표시
    status = f"{top}"
    if top == "ALIGN" and align_sub:
        status += f" • {align_sub}"
    if top == "RECOVER" and recover_sub:
        status += f" • {recover_sub}"
    cv2.putText(img, status, (16, H - 14), FONT, 0.50, (225, 225, 225), 1, cv2.LINE_AA)

    return img
