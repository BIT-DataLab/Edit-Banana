# 🍌 Edit Banana - Next.js 改造完成

## ✅ 已完成内容

### 1. Next.js 15 前端 (apps/web/)

```
apps/web/
├── app/
│   ├── page.tsx           # 首页 (Landing Page)
│   ├── layout.tsx         # 根布局
│   ├── upload/page.tsx    # 上传页面
│   ├── processing/[id]/   # 处理进度页面
│   └── globals.css
├── components/
│   ├── ui/                # 基础UI组件
│   │   ├── button.tsx
│   │   └── card.tsx
│   ├── upload/
│   │   └── file-upload.tsx    # 文件上传组件
│   ├── progress/
│   │   └── progress-bar.tsx   # 进度条组件
│   └── navbar.tsx         # 导航栏
├── lib/
│   ├── utils.ts           # 工具函数
│   ├── config.ts          # 配置
│   ├── types.ts           # TypeScript类型
│   ├── api.ts             # API客户端
│   └── websocket.ts       # WebSocket hook
├── package.json           # 依赖
├── next.config.ts         # Next.js配置
└── vercel.json            # Vercel部署配置
```

**技术栈:**
- Next.js 15 + React 19 + TypeScript
- Tailwind CSS 4
- Framer Motion (动画)
- react-dropzone (文件上传)
- Lucide React (图标)

### 2. Python FastAPI 后端 (apps/api/)

```
apps/api/
├── main_api.py            # FastAPI主入口
├── main.py -> ../../main.py  (符号链接)
├── modules -> ../../modules  (符号链接)
├── config -> ../../config    (符号链接)
├── requirements.txt       # Python依赖
├── railway.toml           # Railway部署配置
└── fly.toml               # Fly.io部署配置
```

**API端点:**
- `POST /api/v1/convert` - 创建转换任务
- `GET /api/v1/jobs/{id}` - 查询任务状态
- `GET /api/v1/jobs/{id}/result` - 下载结果
- `DELETE /api/v1/jobs/{id}` - 取消任务
- `WS /ws/jobs/{id}/progress` - 实时进度推送

### 3. CI/CD 配置 (.github/workflows/)

- `deploy-web.yml` - Vercel自动部署
- `deploy-api.yml` - Railway/Fly.io自动部署

### 4. 文档

- `PROJECT-NEXTJS.md` - 项目说明
- `.env.local.example` - 环境变量示例

---

## 🚀 部署指南

### 前端 → Vercel

```bash
# 安装Vercel CLI
npm i -g vercel

# 部署
cd apps/web
vercel --prod
```

**环境变量:**
```
NEXT_PUBLIC_API_URL=https://api.yoursite.com
NEXT_PUBLIC_WS_URL=wss://api.yoursite.com
```

### 后端 → Railway

```bash
# 安装Railway CLI
npm i -g @railway/cli

# 部署
cd apps/api
railway login
railway init
railway up
```

### 后端 → Fly.io

```bash
# 安装Fly CLI
brew install flyctl

# 部署
cd apps/api
flyctl launch
flyctl deploy
```

---

## 🏃 本地开发

```bash
# 1. 启动Python后端
cd apps/api
pip install -r requirements.txt
python main_api.py

# 2. 启动Next.js前端 (新终端)
cd apps/web
npm install
npm run dev

# 3. 访问 http://localhost:3000
```

---

## 📁 文件清单

| 路径 | 说明 |
|------|------|
| `apps/web/app/page.tsx` | 首页Landing Page |
| `apps/web/app/upload/page.tsx` | 文件上传页面 |
| `apps/web/app/processing/[id]/page.tsx` | 处理进度页面 |
| `apps/web/lib/websocket.ts` | WebSocket hook |
| `apps/web/lib/api.ts` | API客户端 |
| `apps/web/components/upload/file-upload.tsx` | 上传组件 |
| `apps/web/components/progress/progress-bar.tsx` | 进度条 |
| `apps/api/main_api.py` | FastAPI后端 |
| `apps/api/railway.toml` | Railway配置 |
| `apps/api/fly.toml` | Fly.io配置 |
| `.github/workflows/` | CI/CD工作流 |
| `PROJECT-NEXTJS.md` | 完整说明文档 |

---

## ⚠️ 注意事项

1. **Python后端无法部署到Vercel**，因为Vercel只支持Node.js运行时
2. 推荐部署方案:
   - 前端: **Vercel** (免费)
   - 后端: **Railway** 或 **Fly.io** (有免费额度)
3. SAM3模型需要GPU，确保后端部署在支持GPU的平台上
4. 需要配置环境变量 `NEXT_PUBLIC_API_URL` 指向后端地址

---

## 🎯 特性

- ✅ 拖拽上传图片/PDF
- ✅ WebSocket实时进度推送
- ✅ 多阶段处理显示 (预处理 → OCR → 分割 → 处理 → XML生成)
- ✅ 可配置转换选项
- ✅ 响应式设计
- ✅ 现代化UI
- ✅ 自动CI/CD部署
