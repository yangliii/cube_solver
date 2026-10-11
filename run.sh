#!/usr/bin/env bash
# 启动魔方网页应用（使用项目内虚拟环境）
# 注意：先 cd 到脚本所在目录，求解引擎的 tables/ 是按当前目录查找的
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec ./.venv/bin/python app.py "$@"
