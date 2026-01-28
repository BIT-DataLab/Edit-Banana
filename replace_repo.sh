#!/bin/bash
# 用当前工作区内容完全替换远程 Edit-Banana 仓库，清空历史，保留 Star。
# 使用前请阅读 REPLACE_REPO.md，确认 .gitignore 已排除敏感文件。
set -e
cd "$(dirname "$0")"

echo "=== 1. 检查敏感文件是否被忽略 ==="
if git check-ignore -q config/config.yaml 2>/dev/null; then
  echo "  [OK] config/config.yaml 已被忽略"
else
  echo "  [WARN] config/config.yaml 未被忽略，请检查 .gitignore"
  exit 1
fi

echo ""
echo "=== 2. 设置远程为 Edit-Banana ==="
if ! git remote get-url origin &>/dev/null; then
  git remote add origin https://github.com/BIT-DataLab/Edit-Banana.git
else
  git remote set-url origin https://github.com/BIT-DataLab/Edit-Banana.git
fi
git remote -v

echo ""
echo "=== 3. 创建无历史分支并提交当前内容 ==="
git checkout --orphan new_main 2>/dev/null || true
git rm -rf --cached . 2>/dev/null || true
git add .
echo "  即将提交的文件："
git status --short
read -p "  确认无敏感文件后按 Enter 继续，Ctrl+C 取消..."
git commit -m "Initial commit: algorithm pipeline only (Image to DrawIO)"

echo ""
echo "=== 4. 覆盖远程 main ==="
git branch -D main 2>/dev/null || true
git branch -m main
echo "  即将执行: git push -f origin main"
read -p "  确认后按 Enter 执行，Ctrl+C 取消..."
git push -f origin main

echo ""
echo "=== 完成 ==="
echo "  仓库已替换，历史已清空，Star 保留。"
echo "  https://github.com/BIT-DataLab/Edit-Banana"
