#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 cube_scanner.py 的扫描结果（cube_*.json）转换成 cube_solver 的 54 位 facelet 状态串。

约定（与 web/index.html、app.py 完全一致）：
  facelet 顺序 U L F R B D，每面 3x3 行优先；配色 上白W 下黄Y 前绿G 后蓝B 右红R 左橙O。

难点：扫描时每个面都是随手拿着拍的，照片里该面的朝向与标准朝向可能相差 0/90/180/270°，
单张图无法判断。做法：
  1. 用各面「中心块颜色」确定该面是 U/D/F/B/L/R 中的哪一面；
  2. 6 个面各试 4 种旋转（至多 4^6 = 4096 种组合），逐一做完整合法性校验
     （每种颜色恰 9 格、角朝向和 % 3、棱翻转数 % 2、角/棱置换奇偶一致，最后把解析出的
     块状态重新渲染回 facelet 串逐格核对），第一个通过校验的组合即为真实状态；
  3. 全部不通过（识别有误）时，按「像真魔方的程度」取最接近的组合返回，供人工修正。
另外：JSON 里存有原始采样值（median_bgr），若 cube_scanner.py 的识别阈值调整过，
会用最新阈值重新识别一遍再使用（新结果更差时才保留存档颜色）。

命令行自测：
    python scan_to_facelet.py [cube_xxx.json]     # 缺省取同目录下最新的 cube_*.json
"""

import itertools
import json
import sys
from pathlib import Path

try:  # 同目录的扫描脚本：用其当前阈值对存档采样值重新识别（阈值调整后旧数据也能受益）
    from cube_scanner import classify_color as _classify_color
except Exception:  # noqa: BLE001  （缺 cv2/numpy 时退回 JSON 里存好的颜色）
    _classify_color = None

COLORS = "WYGRBO"
CENTER_TO_FACE = {"W": "U", "Y": "D", "G": "F", "B": "B", "R": "R", "O": "L"}
FACE_COLOR = {v: k for k, v in CENTER_TO_FACE.items()}
FACE_ORDER = ["U", "L", "F", "R", "B", "D"]          # facelet 拼接顺序
FACE_BASE = {"U": 0, "L": 9, "F": 18, "R": 27, "B": 36, "D": 45}
FACE_CN = {"U": "上", "D": "下", "F": "前", "B": "后", "R": "右", "L": "左"}


def _fi(face, rc):
    return FACE_BASE[face] + rc[0] * 3 + rc[1]


# ---- 角块/棱块位置表：与 Kociemba 标准表逐项对应（U9,R1,F3…），只换成本项目的 facelet 编号 ----
# 每项 = 名字 + 各贴纸的 (面, (行, 列))；行 0 为该面网格顶部（U 面=后沿、D 面=前沿），列 0 为观察者左手边。
# 角块的第一个贴纸必须位于 U/D 面（标准朝向算法的约定）。
_CORNERS = [
    ("URF", ("U", (2, 2)), ("R", (0, 0)), ("F", (0, 2))),
    ("UFL", ("U", (2, 0)), ("F", (0, 0)), ("L", (0, 2))),
    ("ULB", ("U", (0, 0)), ("L", (0, 0)), ("B", (0, 2))),
    ("UBR", ("U", (0, 2)), ("B", (0, 0)), ("R", (0, 2))),
    ("DFR", ("D", (0, 2)), ("F", (2, 2)), ("R", (2, 0))),
    ("DLF", ("D", (0, 0)), ("L", (2, 2)), ("F", (2, 0))),
    ("DBL", ("D", (2, 0)), ("B", (2, 2)), ("L", (2, 0))),
    ("DRB", ("D", (2, 2)), ("R", (2, 2)), ("B", (2, 0))),
]
_EDGES = [
    ("UR", ("U", (1, 2)), ("R", (0, 1))),
    ("UF", ("U", (2, 1)), ("F", (0, 1))),
    ("UL", ("U", (1, 0)), ("L", (0, 1))),
    ("UB", ("U", (0, 1)), ("B", (0, 1))),
    ("DR", ("D", (1, 2)), ("R", (2, 1))),
    ("DF", ("D", (0, 1)), ("F", (2, 1))),
    ("DL", ("D", (1, 0)), ("L", (2, 1))),
    ("DB", ("D", (2, 1)), ("B", (2, 1))),
    ("FR", ("F", (1, 2)), ("R", (1, 0))),
    ("FL", ("F", (1, 0)), ("L", (1, 2))),
    ("BL", ("B", (1, 2)), ("L", (1, 0))),
    ("BR", ("B", (1, 0)), ("R", (1, 2))),
]
CORNER_FACELET = [[_fi(f, rc) for f, rc in c[1:]] for c in _CORNERS]
EDGE_FACELET = [[_fi(f, rc) for f, rc in e[1:]] for e in _EDGES]
CORNER_COLOR = [[FACE_COLOR[f] for f, _ in c[1:]] for c in _CORNERS]
EDGE_COLOR = [[FACE_COLOR[f] for f, _ in e[1:]] for e in _EDGES]


def rot90(g):
    """把 3x3 网格顺时针旋转 90°（g 为长度 9 的行优先列表）。"""
    return [g[(2 - j) * 3 + i] for i in range(3) for j in range(3)]


def _all_rots(g):
    out, cur = [], list(g)
    for _ in range(4):
        out.append(cur)
        cur = rot90(cur)
    return out


def _score(s):
    """「位置上的贴纸恰好等于某真实块的合法摆放」的位置数（满分 8 + 12 = 20）。
    任何合法状态都得 20 分，分数越高代表该朝向组合越接近真实魔方；'?' 格不算分。"""
    n = 0
    for cf in CORNER_FACELET:
        cols = [s[k] for k in cf]
        for cc in CORNER_COLOR:
            if sorted(cols) == sorted(cc) and any(
                    all(cols[(ori + t) % 3] == cc[t] for t in range(3))
                    for ori in range(3)):
                n += 1
                break
    for ef in EDGE_FACELET:
        pair = (s[ef[0]], s[ef[1]])
        for ec in EDGE_COLOR:
            if pair in ((ec[0], ec[1]), (ec[1], ec[0])):
                n += 1
                break
    return n


def _render(cp, co, ep, eo):
    """把角/棱块排列（含朝向）重新渲染成 54 位串，用于与输入逐格核对。"""
    out = [""] * 54
    for fc in FACE_ORDER:
        out[FACE_BASE[fc] + 4] = FACE_COLOR[fc]
    for i in range(8):
        cc = CORNER_COLOR[cp[i]]
        for t in range(3):
            out[CORNER_FACELET[i][(co[i] + t) % 3]] = cc[t]
    for i in range(12):
        ec = EDGE_COLOR[ep[i]]
        pair = (ec[0], ec[1]) if eo[i] == 0 else (ec[1], ec[0])
        out[EDGE_FACELET[i][0]], out[EDGE_FACELET[i][1]] = pair
    return "".join(out)


def _check_state(s):
    """解析 54 位串为块排列并严格校验可达性。合法返回 None，否则返回原因。"""
    cp, co = [0] * 8, [0] * 8
    for i, idx in enumerate(CORNER_FACELET):
        cols = [s[k] for k in idx]
        ori = next((k for k in range(3)
                    if cols[k] in (FACE_COLOR["U"], FACE_COLOR["D"])), None)
        if ori is None:
            return f"角位 {_CORNERS[i][0]} 缺少上/下色（{''.join(cols)}）"
        c1, c2 = cols[(ori + 1) % 3], cols[(ori + 2) % 3]
        j = next((t for t, cc in enumerate(CORNER_COLOR)
                  if c1 == cc[1] and c2 == cc[2]), None)
        if j is None:
            return f"角位 {_CORNERS[i][0]} 的颜色组合非法（{''.join(cols)}）"
        cp[i], co[i] = j, ori
    if len(set(cp)) != 8:
        return "角块重复或缺失"
    ep, eo = [0] * 12, [0] * 12
    for i, idx in enumerate(EDGE_FACELET):
        c0, c1 = s[idx[0]], s[idx[1]]
        j = next((t for t, ec in enumerate(EDGE_COLOR)
                  if (c0, c1) in ((ec[0], ec[1]), (ec[1], ec[0]))), None)
        if j is None:
            return f"棱位 {_EDGES[i][0]} 的颜色组合非法（{c0}{c1}）"
        ep[i] = j
        eo[i] = 0 if c0 == EDGE_COLOR[j][0] else 1
    if len(set(ep)) != 12:
        return "棱块重复或缺失"
    if sum(co) % 3:
        return f"角块朝向和不对（{sum(co)}，应为 3 的倍数）"
    if sum(eo) % 2:
        return f"棱块翻转数不对（{sum(eo)}，应为偶数）"
    if sum(1 for i in range(8) for j in range(i) if cp[j] > cp[i]) % 2 != \
       sum(1 for i in range(12) for j in range(i) if ep[j] > ep[i]) % 2:
        return "角块与棱块置换奇偶不一致"
    if _render(cp, co, ep, eo) != s:
        return "存在无法解释的贴纸（个别格子的颜色识别可能有误）"
    return None


def _face_grid(f):
    """取一面的 3x3 颜色矩阵；优先用当前阈值对采样值重新识别。"""
    grid = [[str(c) for c in row] for row in f["colors"]]
    if len(grid) != 3 or any(len(r) != 3 for r in grid):
        raise ValueError("颜色矩阵不是 3x3")
    if _classify_color is not None and f.get("median_bgr"):
        try:
            g2 = [[str(_classify_color(cell)) for cell in row] for row in f["median_bgr"]]
            n_old = sum(r.count("?") for r in grid)
            n_new = sum(r.count("?") for r in g2)
            if n_new <= n_old:      # 新阈值没识别得更差才采用
                grid = g2
        except Exception:  # noqa: BLE001
            pass
    return grid


def scan_to_repr(json_path):
    """读取扫描 JSON 并转换为 facelet。返回结果 dict，失败时 ok=False + error。"""
    path = Path(json_path)
    try:
        scan = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"读取 {path.name} 失败：{e}"}

    warnings = []

    # ---- 1. 读取各面（可重新识别） ----
    grids = []
    for f in (scan.get("faces") or []):
        try:
            grids.append((f, _face_grid(f)))
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"第 {f.get('index', '?')} 面数据不正确：{e}"}
    if not grids:
        return {"ok": False, "error": f"{path.name} 里没有任何面数据"}

    # ---- 2. 按中心块颜色确定面归属 ----
    face_of_center = {}
    for f, g in grids:
        c = g[1][1]
        if c not in CENTER_TO_FACE:
            return {"ok": False, "error":
                    f"第 {f.get('index', '?')} 面的中心块识别为「{c}」，无法定位该面；请重新扫描或手动录入"}
        if c in face_of_center:
            return {"ok": False, "error":
                    f"两个面的中心块都识别为「{c}」，无法区分各面；请重新扫描或手动录入"}
        face_of_center[c] = CENTER_TO_FACE[c]

    missing = [fc for fc in FACE_ORDER if fc not in face_of_center.values()]
    if missing:
        warnings.append("扫描不完整，缺失 " + "/".join(missing) + " 面（已留空待手动补填）")

    # ---- 3. 每面 4 种旋转，枚举组合取最优 ----
    face_rots = {}
    for f, g in grids:
        face = face_of_center[g[1][1]]
        flat = [v if v in COLORS else "?" for row in g for v in row]
        face_rots[face] = _all_rots(flat)

    present = [fc for fc in FACE_ORDER if fc in face_rots]
    pos_of = {fc: i for i, fc in enumerate(present)}

    def build(ks):
        return "".join("".join(face_rots[fc][ks[pos_of[fc]]]) if fc in face_rots
                       else "?" * 9 for fc in FACE_ORDER)

    strings = sorted({build(ks) for ks in itertools.product(range(4), repeat=len(present))})

    # ---- 4. 先找完全合法的组合；找不到再按「像真魔方的程度」取最接近的 ----
    def full_check(s):
        counts = {c: s.count(c) for c in COLORS}
        unknown = s.count("?")
        if missing:
            return False, "扫描不完整", counts
        if unknown:
            return False, f"有 {unknown} 格未识别出颜色", counts
        bad = {c: n for c, n in counts.items() if n != 9}
        if bad:
            return False, ("颜色数量与标准不符（"
                           + "、".join(f"{c}={n}" for c, n in bad.items())
                           + "，每种应为 9 格）"), counts
        reason = _check_state(s) or ""
        return (not reason), reason, counts

    repr_s, valid, reason, counts = None, False, "", {c: 0 for c in COLORS}
    for s in strings:
        v, r, c = full_check(s)
        if v:
            repr_s, valid, reason, counts = s, True, "", c
            break
    if repr_s is None:
        score_of = {s: _score(s) for s in strings}
        top = sorted(s for s, sc in score_of.items() if sc == max(score_of.values()))
        repr_s = top[0]
        valid, reason, counts = full_check(repr_s)
        if len(top) > 1:
            warnings.append(f"最接近的朝向组合有 {len(top)} 种，已任选其一")
    if not valid and not missing:
        warnings.append("状态未通过合法性校验：" + reason + "；请对照魔方修正后再求解")

    faces_out = {fc: [ch if ch != "?" else "" for ch in repr_s[FACE_BASE[fc]:FACE_BASE[fc] + 9]]
                 for fc in FACE_ORDER}

    return {
        "ok": True,
        "source": path.name,
        "captured_at": str(scan.get("captured_at", "")),
        "repr": repr_s,
        "faces": faces_out,
        "valid": valid,
        "warnings": warnings,
        "score": _score(repr_s),
        "counts": counts,
    }


def latest_scan_file(scan_dir):
    """取目录下最新的 cube_*.json（文件名带时间戳，按名称排序即可）。"""
    files = list(Path(scan_dir).glob("cube_*.json"))
    return max(files, key=lambda p: p.name) if files else None


def latest_scan_result(scan_dir):
    """web 后端用：转换最新一次扫描结果。"""
    p = latest_scan_file(scan_dir)
    if p is None:
        return {"ok": False, "error":
                "cube_scans 目录下没有扫描文件（cube_*.json），请先运行 cube_scanner.py 采集"}
    return scan_to_repr(p)


def main():
    here = Path(__file__).resolve().parent
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = latest_scan_file(here)
        if path is None:
            print(f"目录 {here} 下没有 cube_*.json，请先运行 cube_scanner.py 采集")
            return 1
    res = scan_to_repr(path)
    if not res.get("ok"):
        print("转换失败:", res.get("error"))
        return 1
    print(f"来源: {res['source']}（{res['captured_at']}）")
    for fc in FACE_ORDER:
        g = res["faces"][fc]
        rows = ["".join(ch if ch else "·" for ch in g[i * 3:i * 3 + 3]) for i in range(3)]
        print(f"  {fc}({FACE_CN[fc]})  " + " / ".join(rows))
    print(f"朝向组合得分: {res['score']}/20")
    print("颜色统计:", res["counts"])
    print("合法状态:", "是 ✅" if res["valid"] else "否 ❌")
    for w in res["warnings"]:
        print("  ⚠", w)
    print("facelet:", res["repr"])
    return 0 if res["valid"] else 2


if __name__ == "__main__":
    sys.exit(main())
