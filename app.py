# -*- coding: utf-8 -*-
"""
三阶魔方新手还原教程 - Python 后端
职责：
  1. 提供静态页面服务（web/index.html，Three.js 3D 前端）
  2. 提供教程数据 API（/api/steps 分层法七步教程、/api/notation 符号说明）
  3. 提供魔方求解 API（/api/solve：接收当前状态，返回自动还原步骤）
运行：python app.py  然后浏览器访问 http://localhost:8000
求解引擎：cube_solver（纯 Python + C 加速的 Kociemba 两阶段算法）。
         配色约定与前端完全一致：上白(W) 下黄(Y) 前绿(G) 后蓝(B) 右红(R) 左橙(O)。
         facelet 字符串顺序为 U L F R B D，每面按行优先 9 格。
"""
import json
import os
import mimetypes
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

# ---------------- 求解引擎（可选依赖，缺失时后端仍可运行） ----------------
try:
    from cube_solver import Cube, Kociemba
    _SOLVER = Kociemba()
    SOLVER_OK = True
    SOLVER_ERR = ""
except Exception as _e:  # noqa: BLE001
    Cube = Kociemba = _SOLVER = None
    SOLVER_OK = False
    SOLVER_ERR = str(_e)


ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(ROOT, "web")
PORT = 8000

# ---------------- 转动符号说明 ----------------
NOTATION = {
    "faces": [
        {"face": "R", "cn": "右面", "desc": "Right - 右侧面顺时针转 90°"},
        {"face": "L", "cn": "左面", "desc": "Left - 左侧面顺时针转 90°"},
        {"face": "U", "cn": "上面", "desc": "Up - 顶面顺时针转 90°"},
        {"face": "D", "cn": "下面", "desc": "Down - 底面顺时针转 90°"},
        {"face": "F", "cn": "前面", "desc": "Front - 正对您的前面顺时针转 90°"},
        {"face": "B", "cn": "后面", "desc": "Back - 背面顺时针转 90°"},
    ],
    "modifiers": [
        {"symbol": "R",  "meaning": "该面顺时针转 90°"},
        {"symbol": "R'", "meaning": "该面逆时针转 90°"},
        {"symbol": "R2", "meaning": "该面转 180°"},
    ],
    "tips": "提示：字母代表要转动哪个面。R U R' U' 表示依次执行 R、U、R逆、U逆 四个动作。",
}

# ---------------- 分层法七步教程 ----------------
STEPS = [
    {
        "id": 1,
        "title": "底面十字",
        "goal": "以白色为底，把 4 个带白色的棱块转到底面，拼出白色十字，且棱块侧面颜色与中心对齐。",
        "detail": "这一步没有固定公式，靠理解与练习。先在底层找到目标棱块，想清楚它要到的位置和方向。"
                  "棱块在顶层且白色朝侧面时，可将白色面转到侧面前方，再用一次该面转动送入底层。",
        "formulas": [
            {"name": "白顶直接下落", "moves": "F2", "note": "棱块在前上位置且白色朝上"},
            {"name": "侧面送入", "moves": "R F R'", "note": "棱块在白朝向需转向时用类似手法"},
        ],
    },
    {
        "id": 2,
        "title": "底面角块归位（还原第一层）",
        "goal": "把 4 个带白色的角块放回底层角位，完成整个白色底面，第一层侧面颜色也要对齐。",
        "detail": "反复使用右手公式，直到角块就位。角块在顶层、白色朝右用右手公式；白色朝前用左手公式。",
        "formulas": [
            {"name": "右手公式", "moves": "R U R' U'", "note": "白色角块在右上前位置时重复使用"},
            {"name": "左手公式", "moves": "L' U' L U", "note": "右手公式的镜像，白色朝左时使用"},
        ],
    },
    {
        "id": 3,
        "title": "第二层棱块归位",
        "goal": "顶层 4 个不带黄色的棱块，分别嵌入中层四个位置，完成第二层。",
        "detail": "观察棱块侧面颜色：先转动顶层使棱块侧面颜色对齐中心列，形成倒 T 形，再用右手或左手公式送入。",
        "formulas": [
            {"name": "右手式入位", "moves": "U R U' R' U' F' U F", "note": "棱块要进入右侧中层"},
            {"name": "左手式入位", "moves": "U' L' U L U F U' F'", "note": "棱块要进入左侧中层"},
        ],
    },
    {
        "id": 4,
        "title": "顶面十字",
        "goal": "在顶层黄色面拼出黄色十字（此时角块颜色暂不考虑）。",
        "detail": "先看顶面黄色棱的分布：只有中心一个黄点用公式一次；出现直角（小拐弯）朝向左后方做一次；"
                  "出现一字横着放再做一次即可变十字。",
        "formulas": [
            {"name": "顶十字公式", "moves": "F R U R' U' F'", "note": "点→拐→线→十字，按需重复"},
        ],
    },
    {
        "id": 5,
        "title": "顶面还原（黄色全朝上）",
        "goal": "让顶面 4 个角块也翻成黄色朝上，顶面全部变黄，侧面暂时乱没关系。",
        "detail": "左手固定在前上右角块位置：该角块黄色朝前用顺时针公式，黄色朝左用逆时针公式，"
                  "反复转动 U 面把未还原的角块送到固定位置逐个翻黄。",
        "formulas": [
            {"name": "顺时针翻角", "moves": "R U R' U R U2 R'", "note": "固定角黄色朝前时用"},
            {"name": "逆时针翻角", "moves": "R U2 R' U' R U' R'", "note": "固定角黄色朝左时用"},
        ],
    },
    {
        "id": 6,
        "title": "顶层角块归位",
        "goal": "把顶层 4 个角块移到各自正确的颜色位置（朝向先不管）。",
        "detail": "转顶层寻找两个侧面颜色完全相同的角块对，放到左手边，观察右手边是否有已归位的角块，"
                  "没有则执行一次公式，有则按公式方向调整。",
        "formulas": [
            {"name": "角块三换公式", "moves": "U R U' L' U R' U' L", "note": "让 3 个角块轮换，需找对基准面"},
        ],
    },
    {
        "id": 7,
        "title": "顶层棱块归位（最后一步）",
        "goal": "把顶层 4 个棱块轮换到位，整个魔方还原完成！",
        "detail": "若有一面顶层棱块已全部归位，把它作为底面（B 面）朝向左手；若都未归位，任选方向执行一次公式即会出现归位面。",
        "formulas": [
            {"name": "棱块顺时针三换", "moves": "R U' R U R U R U' R' U' R2", "note": "配合 U 转动调整归位面"},
            {"name": "棱块逆时针三换", "moves": "R2 U R U R' U' R' U' R'", "note": "与上式互为逆操作"},
        ],
    },
]


def parse_solution(maneuver_str):
    """把 cube_solver 的解法字符串(如 "F2 B' U' R'")解析成前端可播放的 token 列表。"""
    tokens = []
    buf = ""
    for ch in maneuver_str:
        if ch in "RLUDFB":
            if buf:
                tokens.append(buf)
            buf = ch
        elif ch in "'2":
            buf += ch
        else:  # 空格或其它分隔符
            if buf:
                tokens.append(buf)
                buf = ""
    if buf:
        tokens.append(buf)
    return tokens


def solve_facelet(facelet):
    """校验并求解：facelet 为 54 位 U L F R B D 顺序、字符集 WGRYBO 的状态串。
    返回 (moves 列表, None) 或 (None, 错误信息字符串)。"""
    if not SOLVER_OK:
        return None, "后端未安装求解引擎 cube_solver，请先执行：pip install cube-solver"
    if not isinstance(facelet, str) or len(facelet) != 54:
        return None, "状态串长度必须为 54"
    if set(facelet) - set("WGRYBO"):
        return None, "状态串只能包含字符 W G R Y B O"
    try:
        cube = Cube(repr=facelet)
        if cube.is_solved:
            return [], None
        maneuver = _SOLVER.solve(cube)
        if maneuver is None:
            return None, "无法求解，该状态可能不可达（魔方被错误拆装）"
        return parse_solution(str(maneuver)), None
    except Exception as e:  # noqa: BLE001
        return None, "求解失败：" + str(e)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # 静默日志，保持控制台清爽

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, filepath):
        if not os.path.isfile(filepath):
            self.send_error(404, "Not Found")
            return
        ctype, _ = mimetypes.guess_type(filepath)
        with open(filepath, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/steps":
            self._send_json({"steps": STEPS})
        elif path == "/api/notation":
            self._send_json(NOTATION)
        elif path in ("/", "/index.html"):
            self._send_file(os.path.join(WEB_DIR, "index.html"))
        elif path.startswith("/web/"):
            # 防目录穿越
            rel = os.path.normpath(path[5:])
            if rel.startswith("..") or os.path.isabs(rel):
                self.send_error(403, "Forbidden")
                return
            self._send_file(os.path.join(WEB_DIR, rel))
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/api/solve":
            try:
                length = int(self.headers.get("Content-Length", 0) or 0)
                raw = self.rfile.read(length) if length else b"{}"
                payload = json.loads(raw.decode("utf-8") or "{}")
            except Exception:  # noqa: BLE001
                self._send_json({"error": "请求体必须是合法 JSON"}, 400)
                return
            facelet = payload.get("repr") or payload.get("facelet")
            moves, err = solve_facelet(facelet)
            if err:
                self._send_json({"error": err}, 400)
                return
            self._send_json({"moves": moves, "solution": " ".join(moves), "count": len(moves)})
        else:
            self.send_error(404, "Not Found")


def main():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print("=" * 46)
    print("  三阶魔方新手还原教程 已启动")
    print(f"  请用浏览器访问: http://localhost:{PORT}")
    if SOLVER_OK:
        try:  # 预热：首次求解会构建搜索表，避免用户第一次点击时卡顿
            _SOLVER.solve(Cube("R U R' U'"))
            print("  求解引擎: cube_solver 就绪")
        except Exception as e:  # noqa: BLE001
            print("  求解引擎警告:", e)
    else:
        print("  求解引擎不可用(自动还原将失败):", SOLVER_ERR)
        print("  安装命令: pip install cube-solver")

    print("  按 Ctrl+C 停止服务")
    print("=" * 46)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")


if __name__ == "__main__":
    main()
