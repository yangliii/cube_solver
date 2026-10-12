# -*- coding: utf-8 -*-
"""面片级魔方模拟器（仅用于测试，零第三方依赖）。

与前端 web/index.html 完全一致的整数矩阵约定：
  每张贴纸用 (home 颜色, 当前所在块坐标 pos, 当前世界法线 normal) 描述，
  转动某一层时对该层所有贴纸的 pos/normal 同时做整数 90° 旋转，
  逻辑状态零浮点漂移。facelet 顺序与配色同 app.py：U L F R B D，上白W 下黄Y
  前绿G 后蓝B 右红R 左橙O。
"""

# 每面顺时针（从该面外侧观察）的整数旋转矩阵，取自前端 FACES 定义
FACE_MAT = {
    "R": [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
    "L": [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
    "U": [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
    "D": [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
    "F": [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
    "B": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
}
FACE_AXIS = {"R": (0, 1), "L": (0, -1), "U": (1, 1), "D": (1, -1),
             "F": (2, 1), "B": (2, -1)}
FACE_BASE = {"U": 0, "L": 9, "F": 18, "R": 27, "B": 36, "D": 45}

# home 法线 -> 颜色（与前端 HOME_LETTER 一致）
NORMAL_COLOR = {
    (1, 0, 0): "R", (-1, 0, 0): "O", (0, 1, 0): "W",
    (0, -1, 0): "Y", (0, 0, 1): "G", (0, 0, -1): "B",
}
# 当前世界法线 -> 面
NORMAL_FACE = {
    (1, 0, 0): "R", (-1, 0, 0): "L", (0, 1, 0): "U",
    (0, -1, 0): "D", (0, 0, 1): "F", (0, 0, -1): "B",
}
SOLVED = ("WWWWWWWWW" "OOOOOOOOO" "GGGGGGGGG" "RRRRRRRRR"
          "BBBBBBBBB" "YYYYYYYYY")


def _mul(M, v):
    return [M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2],
            M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2],
            M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2]]


def _transpose(M):
    return [[M[0][0], M[1][0], M[2][0]],
            [M[0][1], M[1][1], M[2][1]],
            [M[0][2], M[1][2], M[2][2]]]


def _mmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]


def _facelet_index(face, x, y, z):
    """块坐标 -> 该面 9 格下标，公式与前端 faceletIndex 一致。"""
    if face == "U":
        row, col = z + 1, x + 1
    elif face == "D":
        row, col = 1 - z, x + 1
    elif face == "F":
        row, col = 1 - y, x + 1
    elif face == "B":
        row, col = 1 - y, 1 - x
    elif face == "R":
        row, col = 1 - y, 1 - z
    else:  # L
        row, col = 1 - y, z + 1
    return FACE_BASE[face] + row * 3 + col


def invert_tokens(tokens):
    """把整段手法倒序取逆（R -> R'，R' -> R，R2 -> R2）。"""
    out = []
    for t in reversed(tokens):
        face, mod = t[0], t[1:]
        out.append(t if mod == "2" else face + ("'" if mod == "" else ""))
    return out


class SimCube:
    def __init__(self):
        # 54 张贴纸：枚举每个 home 法线上的 9 个块位置
        self.stickers = []
        for n, color in NORMAL_COLOR.items():
            ax = 0 if n[0] else (1 if n[1] else 2)
            others = [i for i in range(3) if i != ax]
            for a in (-1, 0, 1):
                for b in (-1, 0, 1):
                    p = [0, 0, 0]
                    p[ax] = n[ax]
                    p[others[0]], p[others[1]] = a, b
                    self.stickers.append([color, p, list(n)])

    def move(self, token):
        face = token[0]
        mod = token[1:]
        if face not in FACE_MAT:
            raise ValueError(f"未知转动面：{token}")
        mat = FACE_MAT[face]
        if mod == "'":
            mat = _transpose(mat)
        elif mod == "2":
            mat = _mmul(mat, mat)
        elif mod:
            raise ValueError(f"未知修饰符：{token}")
        ax, sign = FACE_AXIS[face]
        for st in self.stickers:
            if st[1][ax] == sign:
                st[1] = _mul(mat, st[1])
                st[2] = _mul(mat, st[2])
        return self

    def apply(self, tokens):
        for t in tokens:
            self.move(t)
        return self

    def repr(self):
        s = [""] * 54
        for color, p, n in self.stickers:
            face = NORMAL_FACE[tuple(n)]
            s[_facelet_index(face, *p)] = color
        if any(c == "" for c in s):
            raise RuntimeError("模拟后出现未覆盖的面格，模拟器有误")
        return "".join(s)
