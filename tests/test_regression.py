# -*- coding: utf-8 -*-
"""可复现回归测试（直接运行：python tests/test_regression.py，或 python -m unittest discover -s tests）。

覆盖：
  1. 面片模拟器自身正确性（还原态、X^4=恒等、公式+逆公式回到还原态）
  2. 教程《分层七步法》所有公式作用在合法魔方上、且结果仍为合法可达状态
     （用生产代码 scan_to_facelet._check_state 做严格校验）
  3. 随机打乱 200 个状态全部通过合法性校验
  4. app.parse_solution 正常解析与非法记号显式报错（不静默截断）
  5. 最新一次摄像头扫描结果可被转换为 54 位 facelet（若存在扫描文件）

说明：README 中“第 5 步覆盖 148 个入口局面、第 6+7 步覆盖 287 个局面”的
判定规则穷举验证脚本规模较大，本测试先固化「公式级」回归基线；
后续若把规则引擎脚本找回，可直接在 tests/ 下扩展。
"""
import os
import sys
import random
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, TESTS_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "cube_scans"))

from facelet_sim import SimCube, invert_tokens, SOLVED  # noqa: E402

import app  # noqa: E402
from scan_to_facelet import _check_state, scan_to_repr, latest_scan_file  # noqa: E402

ALL_TOKENS = [f + m for f in "RLUDFB" for m in ("", "'", "2")]


def tutorial_formulas():
    """从 app.STEPS 取出全部公式 token（去重保序）。"""
    seen, out = set(), []
    for step in app.STEPS:
        for f in step["formulas"]:
            for t in f["moves"].split():
                if t not in seen:
                    seen.add(t)
                    out.append(t)
    return [f["moves"].split() for step in app.STEPS for f in step["formulas"]]


class TestSimulator(unittest.TestCase):
    def test_solved_repr(self):
        self.assertEqual(SimCube().repr(), SOLVED)

    def test_four_quarter_turns_identity(self):
        for face in "RLUDFB":
            c = SimCube()
            for _ in range(4):
                c.move(face)
            self.assertEqual(c.repr(), SOLVED, f"{face}^4 应为恒等")
            c = SimCube()
            for _ in range(4):
                c.move(face + "'")
            self.assertEqual(c.repr(), SOLVED, f"{face}'^4 应为恒等")

    def test_half_turn_equals_two_quarters(self):
        for face in "RLUDFB":
            a = SimCube().move(face + "2").repr()
            b = SimCube().apply([face, face]).repr()
            self.assertEqual(a, b, f"{face}2 应等于 {face}{face}")


class TestTutorialFormulas(unittest.TestCase):
    def test_formulas_keep_state_valid_and_invertible(self):
        bad_tokens = [t for seq in tutorial_formulas() for t in seq
                      if t not in ALL_TOKENS]
        self.assertEqual(bad_tokens, [], "教程中存在非法记号")
        for seq in tutorial_formulas():
            # 作用于还原态后必须仍是合法魔方
            c = SimCube().apply(seq)
            self.assertIsNone(_check_state(c.repr()),
                              f"公式 {' '.join(seq)} 产生了非法状态：{c.repr()}")
            # 公式后紧跟逆公式必须回到还原态
            c.apply(invert_tokens(seq))
            self.assertEqual(c.repr(), SOLVED,
                             f"公式 {' '.join(seq)} 的逆未能复原")


class TestRandomStates(unittest.TestCase):
    def test_200_random_scrambles_valid(self):
        rng = random.Random(20261012)
        for i in range(200):
            n = rng.randint(1, 25)
            seq = [rng.choice(ALL_TOKENS) for _ in range(n)]
            c = SimCube().apply(seq)
            self.assertIsNone(_check_state(c.repr()),
                              f"第 {i} 个随机打乱非法：{' '.join(seq)}")


class TestParseSolution(unittest.TestCase):
    def test_parse_normal(self):
        self.assertEqual(app.parse_solution("F2 B' U' R'"), ["F2", "B'", "U'", "R'"])
        self.assertEqual(app.parse_solution(""), [])

    def test_parse_rejects_unknown_notation(self):
        with self.assertRaises(ValueError):
            app.parse_solution("R M U'")      # M 不被支持，必须报错而非静默丢弃
        with self.assertRaises(ValueError):
            app.parse_solution("r u")         # 小写记号不支持


class TestScanConversion(unittest.TestCase):
    def test_latest_scan_convertible(self):
        p = latest_scan_file(os.path.join(ROOT, "cube_scans"))
        if p is None:
            self.skipTest("cube_scans/ 下没有 cube_*.json，跳过扫描转换测试")
        res = scan_to_repr(p)
        self.assertTrue(res.get("ok"), f"扫描转换失败：{res.get('error')}")
        self.assertEqual(len(res["repr"]), 54)


if __name__ == "__main__":
    unittest.main(verbosity=2)
