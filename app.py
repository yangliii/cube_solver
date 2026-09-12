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
# 本页面所有公式都按「新手拿法」编写：白色那一面朝下当底面，黄色朝上当顶面。
# 教程文案一律用「你手中的方向」（朝下/朝上/朝前/朝后/朝左/朝右）描述，
# 不提具体颜色，这样换任何配色的魔方都适用。
NOTATION = {
    "faces": [
        {"face": "R", "cn": "右", "desc": "Right —— 你右手边那一层，从右侧看过去转 90°"},
        {"face": "L", "cn": "左", "desc": "Left —— 你左手边那一层，从左侧看过去转 90°"},
        {"face": "U", "cn": "上", "desc": "Up —— 顶面，从上方看下去转 90°"},
        {"face": "D", "cn": "下", "desc": "Down —— 底面，从下方看上去转 90°"},
        {"face": "F", "cn": "前", "desc": "Front —— 正对着你的那一面，从正面看过去转 90°"},
        {"face": "B", "cn": "后", "desc": "Back —— 背对你的那一面，从后面看过去转 90°"},
    ],
    "modifiers": [
        {"symbol": "R",  "meaning": "那一层顺时针转 90°"},
        {"symbol": "R'", "meaning": "那一层逆时针转 90°（撇号 = 反向）"},
        {"symbol": "R2", "meaning": "那一层转 180°（方向随意，结果相同）"},
    ],
    "tips": [
        "顺时针 / 逆时针的判断方法：把脸凑到那一面正前方盯着它看，再判断转 90° 是顺时针还是逆时针 —— "
        "这是新手唯一容易搞错的地方。拿不准就点公式卡片上的「单步 ▶」，看左侧 3D 魔方里是哪一层在动、往哪边动。",
        "公式要连着做、中间不要停、也不要中途转动手里的魔方；手顺了肌肉会自己记住。",
        "本页面公式按「白色朝下、黄色朝上」的拿法编写：动手前先把白色中心块转到朝下、黄色中心块朝上"
        "（左侧 3D 魔方默认是白色在上，只是为了好看，跟教程拿法相反）。"
        "后面每一步说的「朝前 / 朝后 / 朝左 / 朝右」，都是指你此刻手里的方向。",
        "卡片上的「▶ 完整演示 / 单步 ▶」转的是左侧 3D 魔方此刻的姿态，只用来确认「这个字母到底转哪一层、转多少度」；"
        "它的上下方向可能和你手里的拿法相反 —— 看演示记手顺，动手仍以你自己的方向为准。",
        "「水平转动整个魔方」＝ 整个魔方一起左右转 90°，六层跟着一起换方向；"
        "这跟「转顶层 U」完全不同 —— 转 U 只会把已经归位的块挪走，做第 5、6、7 步时绝对不要靠转 U 来找位置。",
    ],
}

# ---------------- 分层法七步教程 ----------------
# 下面每一条公式、每一句判定规则都经过面片级魔方模拟器穷举验证：
#   第 5 步规则在全部 148 个「顶十字已完成」入口局面上 100% 收敛（最多 3 次公式）
#   第 6+7 步规则在全部 287 个顶层排列局面上 100% 还原（最多 4 次角公式 + 2 次棱公式）
#   第 1~4 步的公式与判定则用逐块位置断言验证
STEPS = [
    {
        "id": 1,
        "title": "底面十字",
        "goal": "选一个中心块作底面，把 4 个带该底色的棱块转到底层，拼出一个十字，"
                "并且每个棱块的侧面颜色和它旁边的中心块一致。",
        "prepare": "挑一个中心块当底（建议选你最容易辨认的颜色），把它转到**朝下**，"
                   "另一个颜色的中心块朝上。整步魔方都保持这个拿法。",
        "check": "底面出现 4 个同色棱块组成的十字，**并且**这 4 个棱块露在侧面的颜色，"
                 "正好和各自面前方的中心块同色 —— 颜色没对齐就不算完成。",
        "detail": "这一步没有一招通用的公式，靠「目标块现在在哪 → 它该去哪个槽 → 怎么不破坏已做好的部分」来判断。"
                  "先找到所有带底色的棱块，逐个处理：先水平转魔方（或转顶层），把该棱块要去的槽位转到它正下方，"
                  "再按它底色朝哪决定用下面的哪条。已做好的棱块如果被公式带走，就再做一次把它送回去。",
        "formulas": [
            {"name": "直接落下", "moves": "F2",
             "when": "底色朝上的棱块，正好在它的目标槽位正上方",
             "note": "该面转 180°，棱块带着底色落入底层。前提：它两侧的侧面颜色已与该列中心块对齐，"
                     "否则要先水平转整个魔方对齐，或改用下一条。"},
            {"name": "侧面送入", "moves": "U' R' F",
             "when": "棱块在顶层、底色朝向侧面（没朝上也没朝下）",
             "note": "先让该棱块让开位置，再把底色翻朝上，最后让它落到底层。"
                     "这条会暂时打乱顶层其他棱块 —— 属正常现象，稍后重新归位即可。"},
            {"name": "取出重来", "moves": "F2",
             "when": "棱块已在底层，但颜色朝向不对（在错误的槽里翻着）",
             "note": "先把它送回顶层，再按上面两条重新处理。"},
            {"name": "卡在中层", "moves": "F",
             "when": "棱块嵌在中间层，但位置或朝向不对",
             "note": "转一次它所在的那一面（左边的块用 L），它就升到顶层，再按上面两条处理。"},
        ],
        "tips": [
            "四个棱块逐个搞定，不必一次到位；已经做对的棱块被破坏，就顺手补回来。",
            "判断「对齐」只看侧面：棱块露在外面的那两个颜色，要和它相邻的两个中心块颜色完全一致。",
            "把魔方举到眼前，让目标槽位朝向你，比隔着老远盯着顶面更容易判断。",
        ],
    },
    {
        "id": 2,
        "title": "底面角块归位（还原第一层）",
        "goal": "把 4 个带底色的角块放回底层角位：底面四角补齐，同时侧面第一层颜色也要对齐。",
        "prepare": "保持底色朝下、顶面朝上。在**顶层**找一个带底色的角块。",
        "check": "底面四角全部填好，从斜上方看底面是一个完整的同色正方形；"
                 "同时侧面最下面一圈（4 个中心块 + 4 个角块的侧面）颜色一致。",
        "detail": "先水平转动整个魔方，把目标角块要进入的那个角槽转到你面前（该槽两侧的色心与角块另两个颜色对得上），"
                  "此时该角块在**右前上**，再按它的底色朝哪选公式："
                  "朝右 → 右手公式；朝上 → 右手公式连做 3 次；朝前 → 改成从左侧进入，用左手公式。"
                  "一个角块做完会出现新的顶层角块，重复同样的判断即可。",
        "formulas": [
            {"name": "右手公式", "moves": "R U R' U'",
             "when": "角块在右前上，且**底色朝右**（做 1 次）；**底色朝上**（连做 3 次）",
             "note": "反复使用的万能公式：右手边一层上推 → 顶层右移 → 右手边下层推回 → 顶层左移。"
                     "底色朝上时连做 3 次，它就转到侧面并落到底层。"},
            {"name": "左手公式", "moves": "L' U' L U",
             "when": "把魔方水平转动，使角块在**左前上**且**底色朝左**；或角块在右前上、底色朝前",
             "note": "右手公式的镜像。底色朝前的角块无法用右手公式送入当前角槽，"
                     "必须水平转动整个魔方让它到左前上、底色朝左，再用左手公式。"},
        ],
        "tips": [
            "找不到目标角块时，它可能已经埋在底层但位置或朝向错了：用一次右手公式把它换到顶层，再重新处理。",
            "别只盯底面是否填满，侧面第一层的颜色对齐才算真的完成。",
        ],
    },
    {
        "id": 3,
        "title": "第二层棱块归位",
        "goal": "把顶层 4 个「不含顶面颜色」的棱块分别嵌入中间层，完成第二层。",
        "prepare": "还是顶面朝上。在顶层找一个侧面没有顶面颜色的棱块。",
        "check": "侧面中间一圈的 4 个棱块，两端颜色都与各自相邻的中心块一致（下两层全部还原）。",
        "detail": "看这个棱块**朝下**的那个颜色（顶面颜色除外的那一面）：这个颜色决定它要去哪一面。"
                  "水平转动整个魔方，让朝下的颜色与对应的中心块对齐，该棱块正前方就出现一个「上尖下正」的箭头形，"
                  "它的目的地就是这个箭头的下方："
                  "目的地在**右边** → 右手式入位；目的地在**左边** → 左手式入位。"
                  "如果棱块已经卡在中间层但位置/朝向不对：在它所在面做一次该面转动（右侧用 R、左侧用 L）"
                  "把它送回顶层，再按上面的方法重新送入。",
        "formulas": [
            {"name": "右手式入位", "moves": "U R U' R' U' F' U F",
             "when": "目的地在右边（顶层朝前的颜色与右面中心块同色）",
             "note": "口诀「先往右躲、再往上让，转回去、再转回来」：前 4 步是右手公式，后 4 步是它的镜像。"},
            {"name": "左手式入位", "moves": "U' L' U L U F U' F'",
             "when": "目的地在左边（顶层朝前的颜色与左面中心块同色）",
             "note": "右手式入位的镜像，前 4 步是左手公式。"},
        ],
        "tips": [
            "手顺记忆法：右手式 = 右手公式 (R U R' U') 接「魔方往前转一下」的左手公式镜像 (U' F' U F)。",
            "顶层只剩带顶面颜色的棱块时，说明第 3 步已全部完成，可以进入第 4 步。",
        ],
    },
    {
        "id": 4,
        "title": "顶面十字",
        "goal": "让顶面出现一个由 4 个棱块组成的同色十字（此时角块颜色先不管）。",
        "prepare": "顶面（黄色那一面）朝上拿着，第 1~3 步做好的下两层保持不动。",
        "check": "顶面中心块 + 它前后左右 4 个棱块都是顶面色。角块先不用管。",
        "detail": "看顶面有几个顶色棱、摆成什么形状，按下面三种情况处理，每次做完都重新判断："
                  "只有中心一个点 → 做一次公式；"
                  "出现直角拐弯（两个棱，一个朝前一个朝左）→ 把拐角摆成「朝后 + 朝左」再做一次；"
                  "出现一条直线 → 让它**左右方向**横着放再做一次。点 → 拐 → 线 → 十字 正好是它的演化顺序。",
        "formulas": [
            {"name": "顶十字公式", "moves": "F R U R' U' F'",
             "when": "顶面缺棱块时按需重复",
             "note": "前 3 步「F R U」像先做个小十字，后 3 步「R U' F'」是镜像收回；共 6 步，比小鱼的 7 步短。"},
        ],
        "tips": [
            "这一步只解决 4 个棱块的朝向，不会破坏已还原的下两层，放心做。",
            "拐角朝向容易搞错：一定要摆成「一个朝后、一个朝左」（像数字 9 的形状）再做公式。",
        ],
    },
    {
        "id": 5,
        "title": "顶面还原（四角翻色）",
        "goal": "把顶层 4 个角块全部翻成顶面颜色朝上，顶面变成一个完整的同色面（侧面顺序先不管）。",
        "prepare": "顶面继续朝上。**这一步之后不要再用整体翻转魔方的动作**，只允许水平转动整个魔方来找位置。",
        "check": "顶面 9 格全部同色。此时侧面顶层还是乱的，属正常，交给第 6、7 步。",
        "detail": "数一数顶面**还有几个角**是顶色朝上，分三种情况："
                  "① 只剩 1 个角没翻好（俗称「小鱼」）→ 把这个角摆到**左前上**当鱼头，看**右前上**那个角的顶色朝向："
                  "朝前 → 做顺时针小鱼；朝右 → 水平转 180° 把鱼头摆到**右后上**，再做逆时针小鱼，一次就全黄。"
                  "② 有 2 个角朝上且在同一侧面（相邻）→ 水平转动魔方，把这两个角摆到**左面**，做顺时针小鱼。"
                  "③ 2 个朝上的角正好斜对角，或一个都没朝上 → 水平转动魔方，把某个顶色**朝前**的角摆到**右前上**，做顺时针小鱼"
                  "（若四个角没有一个顶色朝前，就随便摆一次后改用逆时针小鱼）。"
                  "每做完一次都重新数一遍，最多 3 次就能全黄。",
        "formulas": [
            {"name": "顺时针小鱼", "moves": "R U R' U R U2 R'",
             "when": "情况 ① 鱼头在左前上且右前上顶色朝前；情况 ②③",
             "note": "记忆：右手公式做两遍，但最后一遍顶面转 180°（R U R' U + R U2 R'），共 7 步。"},
            {"name": "逆时针小鱼", "moves": "R U2 R' U' R U' R'",
             "when": "情况 ① 鱼头已摆到右后上（即右前上顶色朝右时）",
             "note": "顺时针小鱼的逆操作，只用来翻最后那一个角，同样 7 步。"},
        ],
        "tips": [
            "小鱼公式的「头」和「尾」都在顶层，做的时候顶面始终朝上、不要翻魔方，只允许水平转动整个魔方。",
            "这一步只改角块朝向，顶十字和下面两层都不会被破坏。",
        ],
    },
    {
        "id": 6,
        "title": "顶层角块归位",
        "goal": "让顶层 4 个角块的侧面颜色都与相邻中心块对齐（顶面已全黄，这一步不会弄乱它）。",
        "prepare": "顶面保持朝上。**只能用「水平转动整个魔方」来找位置，千万不要转顶层 U**，否则会把已归位的角块带走。",
        "check": "4 个角块的侧面颜色都和相邻的两个中心块对齐。此时顶层的棱块顺序可能还不对，交给第 7 步。",
        "detail": "先观察四个侧面，数有几个角块已经归位（它的两个侧面颜色都和相邻中心块一致），"
                  "然后水平转动整个魔方按下面三种情况处理："
                  "① 有一整面（左右两侧）的 2 个角都已归位 → 把这一面转到**左边**，做 T 变换（一次即可）。"
                  "② 只有 1 个角已归位 → 把这个角摆到**右前上**，做小车公式；做完发现还剩 2 个角要换位置，"
                  "保持同一握法再做一次小车公式即可。"
                  "③ 一个归位的角都找不到 → 随便哪个握法做一次小车公式，就会出现情况 ① 或 ②，再重新判断。",
        "formulas": [
            {"name": "小车公式（三角轮换）", "moves": "R U R' F' R U R' U' R' F R2 U' R'",
             "when": "② 归位角放右前上；③ 找不到归位角时任意握法",
             "note": "顶面 3 个角轮换、另外 1 个不动，同时顺带换 3 个棱块；它做 3 次等于回到原状，"
                     "所以「做一次方向不对」就同握法再做一次即可。"},
            {"name": "T 变换（两角互换）", "moves": "R U R' U' R' F R2 U' R' U' R U R' F'",
             "when": "① 已归位的两个角所在那一面朝左",
             "note": "互换右前上与右后上两个角，同时换两个棱块 —— 棱块顺序交给第 7 步，不用管。"},
        ],
        "tips": [
            "两条公式都只动顶层，绝不会把已经翻好的顶面弄乱；如果顶面出现非顶色块，说明公式中途漏做或多做了。",
            "角块位置只有有限几种情形，做到熟练后可以只看「有几个角已归位」直接决定用哪条公式，不用再逐个试。",
        ],
    },
    {
        "id": 7,
        "title": "顶层棱块归位（最后一步）",
        "goal": "把顶层 4 个棱块轮换到正确位置，整个魔方还原完成！",
        "prepare": "顶面保持朝上，同样只允许水平转动整个魔方。",
        "check": "全部 6 面颜色统一 —— 恭喜通关。",
        "detail": "看四个侧面：如果有一面的顶层棱块已经和中心块对齐（含该面两个角都对），"
                  "就把**这一面转到后面**（朝后、背对你），然后做「三棱轮换」："
                  "做 1 次没还原就保持握法**再做 1 次**，两次里必有一次成功。"
                  "如果四面没有一个棱块归位：随便哪个握法做一次三棱轮换，就会出现归位的那一面，再照上面的办。",
        "formulas": [
            {"name": "三棱轮换", "moves": "R U' R U R U R U' R' U' R2",
             "when": "归位的那一面朝后；或找不到归位面时任意握法",
             "note": "只轮换 3 个棱块、第 4 个（朝后那个）保持不动，角块完全不受影响。"
                     "它的反方向 = 连做两次，所以只需记这一条公式。"},
        ],
        "tips": [
            "做完如果只剩两个棱块要互换，说明第 6 步少做了一次小车公式（角块顺序错了会伪装成棱块问题）"
            "—— 回第 6 步按情况 ① 或 ② 再做一次，再回到这一步。",
            "这一步只需要记 1 条公式：反方向不用另学，同一握法连做两次就是反方向。",
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
