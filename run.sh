#!/usr/bin/env bash
# 前台启动魔方网页应用（使用项目内虚拟环境）
# 关闭本窗口（或按 Ctrl+C）即停止服务
# 注意：先 cd 到脚本所在目录，求解引擎的 tables/ 是按当前目录查找的
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# 端口与 app.py 同一来源：默认 8000，可用 CUBE_PORT 覆盖（也可直接 ./run.sh --port 8123）
PORT="${CUBE_PORT:-8000}"

echo "  访问: http://localhost:$PORT"
echo "  停止: 关闭本窗口 / Ctrl+C"

# python -u 关闭输出缓冲，日志实时可见；前台运行，窗口关闭即结束服务
# "$@" 透传命令行参数（--port / --no-browser / --prewarm）
exec ./.venv/bin/python -u app.py "$@"