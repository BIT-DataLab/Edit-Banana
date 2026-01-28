# 将当前工作区完全替换为 Edit-Banana 仓库（保留 Star、清空历史）

**目标**：用当前文件夹内容完全覆盖 [BIT-DataLab/Edit-Banana](https://github.com/BIT-DataLab/Edit-Banana)，删除原仓库所有提交历史，**Star 数量会保留**（Star 挂在仓库上，与提交历史无关）。

**注意**：以下操作需要你对 `BIT-DataLab/Edit-Banana` 有 **push 权限**（最好是 maintainer/admin）。

---

## 一、不能暴露的文件（已通过 .gitignore 排除）

以下内容**不会**被提交，请确认无误：

- `config/config.yaml`（本地配置、密钥路径等）
- `.env`、`*.env`（环境变量、API Key）
- `models/`、`*.pt`、`*.pth` 等模型文件
- `users.db`、`*.db`
- `output/`、`input/`、`temp/`、`logs/`
- `__pycache__/`、`venv/`、`.venv/`

**操作前请再次确认**：在项目根目录执行 `git status`，确保上面这些路径没有出现在 “Changes to be committed” 里。

---

## 二、操作步骤（在本地终端执行）

在**当前工作区根目录**（`sam3_workflow_prod_fuben`）打开终端，按顺序执行：

### 1. 确认远程仓库地址

```bash
git remote -v
```

若 `origin` 不是 Edit-Banana，请改为：

```bash
git remote set-url origin https://github.com/BIT-DataLab/Edit-Banana.git
```

或首次添加：

```bash
git remote add origin https://github.com/BIT-DataLab/Edit-Banana.git
```

### 2. 用当前内容新建“无历史”分支并提交

```bash
# 新建一个无历史分支（orphan）
git checkout --orphan new_main

# 清空暂存区（避免带上旧历史的文件）
git rm -rf --cached . 2>/dev/null || true

# 按 .gitignore 添加当前工作区文件
git add .

# 再次确认：不要出现 config/config.yaml、.env、models 等
git status

# 若没问题，提交
git commit -m "Initial commit: algorithm pipeline only (Image to DrawIO)"
```

### 3. 用新历史覆盖远程 main 并删除旧历史

```bash
# 删除本地旧的 main（若存在）
git branch -D main 2>/dev/null || true

# 将当前分支改名为 main
git branch -m main

# 强制推送到远程，覆盖远程 main 并清空远程历史
git push -f origin main
```

### 4. （可选）清理远程其他分支与标签

若希望远程只保留一个 `main`、无其他分支/标签：

```bash
# 查看远程分支
git ls-remote --heads origin

# 删除远程其他分支（把 <other-branch> 换成实际分支名）
git push origin --delete <other-branch>

# 删除远程所有标签（慎用）
git ls-remote --tags origin
git push origin --delete $(git ls-remote --tags origin | cut -f2)
```

---

## 三、完成后

- 打开 https://github.com/BIT-DataLab/Edit-Banana 会看到只有 1 个 commit，内容为当前工作区（算法部分）。
- **Star 数不变**，因为仍是同一个仓库。
- 若之前有 Issues/PRs，它们会保留；若希望仓库“像全新项目”，可在 GitHub 上再决定是否关闭或归档。

---

## 四、若执行出错

- **推送被拒绝**：检查是否有 push 权限、是否被分支保护规则拦截（需在 GitHub 设置里暂时允许 force push 或找管理员操作）。
- **误把 config.yaml 加入提交**：在 `git add .` 之后用 `git status` 检查；若已误提交，执行 `git reset HEAD config/config.yaml` 再 `git commit --amend`，然后重新 `git push -f origin main`。
