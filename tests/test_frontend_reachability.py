# -*- coding: utf-8 -*-
"""#10 前端录入「可达性预校验」与后端 _check_state 的一致性回归测试。

前端 web/index.html 里的 checkReachability 是后端 cube_scans/scan_to_facelet.py
的 _check_state 的 JS 移植，两者必须逐字一致。本测试从 index.html 中抽取真实代码，
用 JS 引擎执行后与后端逐例比对；并验证录入界面 buildReprFromEntry 的提示文案。

依赖可选：需要 JS 引擎 quickjs（pip install quickjs）；未安装时本文件整体跳过。
运行：pip install quickjs && python -m unittest discover -s tests
"""
import json
import os
import random
import re
import sys
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, TESTS_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "cube_scans"))

try:
    import quickjs
except ImportError:  # 未安装则整体跳过（不引入硬依赖）
    quickjs = None

from facelet_sim import SimCube, SOLVED  # noqa: E402
from scan_to_facelet import _check_state, CORNER_FACELET, EDGE_FACELET  # noqa: E402

_STUB = """
const FACE_BASE = { U: 0, L: 9, F: 18, R: 27, B: 36, D: 45 };
const PAINT_FACES = ['U', 'L', 'F', 'R', 'B', 'D'];
const FACELET = { U: 'W', D: 'Y', F: 'G', B: 'B', R: 'R', L: 'O' };
const entryState = {};
function fillSolved() { for (const f of PAINT_FACES) entryState[f] = new Array(9).fill(FACELET[f]); }
function setState(map) { fillSolved(); for (const k in map) { const p = k.split(':'); entryState[p[0]][+p[1]] = map[k]; } }
"""


def _extract_js():
    with open(os.path.join(ROOT, "web", "index.html"), encoding="utf-8") as f:
        html = f.read()
    blk = re.search(r"/\* ===== #10 可达性预校验.*?/\* ===== #10 end ===== \*/", html, re.S)
    fn = re.search(r"function buildReprFromEntry\(\) \{.*?\n\}", html, re.S)
    if not blk or not fn:
        raise AssertionError("未能从 index.html 抽取 #10 前端代码（是否被改动/删除？）")
    return blk.group(0), fn.group(0)


def _cases():
    """构造覆盖全部判定分支的用例：合法 + 定向破坏 + 随机全排列。"""
    rng = random.Random(20261012)
    out = [SOLVED]
    for _ in range(120):                      # 合法打乱
        seq = [rng.choice("RLUDFB") + rng.choice(["", "'", "2"])
               for _ in range(rng.randint(1, 25))]
        out.append(SimCube().apply(seq).repr())
    base = list(SOLVED)
    for _ in range(120):                      # 全排列（连中心也乱）：触发各类非法分支
        p = base[:]
        rng.shuffle(p)
        out.append("".join(p))
    # 定向：单棱翻转
    s = list(SOLVED)
    s[EDGE_FACELET[0][0]], s[EDGE_FACELET[0][1]] = s[EDGE_FACELET[0][1]], s[EDGE_FACELET[0][0]]
    out.append("".join(s))
    # 定向：单角扭转
    s = list(SOLVED)
    a, b, c = CORNER_FACELET[0]
    s[a], s[b], s[c] = s[c], s[a], s[b]
    out.append("".join(s))
    # 定向：棱位组合非法
    s = list(SOLVED)
    s[EDGE_FACELET[0][0]] = s[EDGE_FACELET[0][1]] = "G"
    out.append("".join(s))
    # 定向：角块重复
    s = list(SOLVED)
    for dst, src in zip(CORNER_FACELET[1], CORNER_FACELET[0]):
        s[dst] = SOLVED[src]
    out.append("".join(s))
    # 定向：中心不符（渲染核对分支）
    s = list(SOLVED)
    s[4] = "G"
    out.append("".join(s))
    return out


@unittest.skipUnless(quickjs, "未安装 quickjs，跳过前端可达性一致性测试")
class TestFrontendReachability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        blk, fn = _extract_js()
        cls.ctx = quickjs.Context()
        cls.ctx.eval(_STUB + "\n" + blk + "\n" + fn)

    def _js_check(self, s):
        return self.ctx.eval("checkReachability(" + json.dumps(s) + ")")

    def test_js_matches_backend_every_case(self):
        for s in _cases():
            self.assertEqual(self._js_check(s), _check_state(s),
                             f"前后端判定不一致，状态：{s}")

    def test_build_repr_accepts_solved(self):
        self.ctx.eval("fillSolved();")
        res = json.loads(self.ctx.eval("JSON.stringify(buildReprFromEntry())"))
        self.assertNotIn("err", res)
        self.assertEqual(res["repr"], SOLVED)

    def test_build_repr_reports_missing_and_count(self):
        self.ctx.eval("fillSolved(); entryState.D[0]='';")
        self.assertIn("还有未填的格子",
                      json.loads(self.ctx.eval("JSON.stringify(buildReprFromEntry())"))["err"])
        self.ctx.eval("fillSolved(); entryState.U[0]='R';")
        self.assertIn("颜色数量错误",
                      json.loads(self.ctx.eval("JSON.stringify(buildReprFromEntry())"))["err"])

    def test_build_repr_blocks_unreachable(self):
        # 单棱翻转（颜色数量仍各 9）
        self.ctx.eval("setState({'U:5':'R','R:1':'W'});")
        err = json.loads(self.ctx.eval("JSON.stringify(buildReprFromEntry())"))["err"]
        self.assertIn("该状态不可达", err)
        self.assertIn("棱块翻转数不对", err)
        # 两棱整体互换（置换奇偶不一致）
        self.ctx.eval("setState({'R:1':'G','F:1':'R'});")
        err = json.loads(self.ctx.eval("JSON.stringify(buildReprFromEntry())"))["err"]
        self.assertIn("该状态不可达", err)
        self.assertIn("置换奇偶", err)


if __name__ == "__main__":
    unittest.main(verbosity=2)
