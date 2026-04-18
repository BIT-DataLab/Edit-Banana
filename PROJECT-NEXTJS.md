# Edit Banana Next.js 版本

## 项目结构

```
edit-banana/
├── apps/
│   ├── web/                 # Next.js 15 前端 (部署到 Vercel)
│   └── api/                 # Python FastAPI 后端 (部署到 Railway/Fly.io)
│       ├── main_api.py      # 改造后的API入口
│       ├── railway.toml     # Railway部署配置
│       └── fly.toml         # Fly.io部署配置
├── requirements.txt         # Python依赖
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
# 前端
cd apps/web
npm install

# 后端 (需要Python 3.10+)
cd apps/api
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 前端
cp apps/web/.env.local.example apps/web/.env.local

# 后端
cp apps/api/.env.example apps/api/.env
```

### 3. 启动开发服务器

```bash
# 启动后端
cd apps/api
python main_api.py

# 启动前端 (新终端)
cd apps/web
npm run dev
```

访问 http://localhost:3000

## 部署

### 前端 → Vercel

1. 在Vercel导入Git仓库
2. 设置根目录: `apps/web`
3. 添加环境变量:
   - `NEXT_PUBLIC_API_URL`: 你的后端API地址
4. 部署

```bash
# 或使用Vercel CLI
cd apps/web
vercel --prod
```

### 后端 → Railway

1. 在Railway创建项目
2. 导入Git仓库
3. 设置启动命令: `uvicorn main_api:app --host 0.0.0.0 --port $PORT`
4. 部署

```bash
# 或使用Railway CLI
cd apps/api
railway login
railway init
railway up
```

### 后端 → Fly.io

```bash
cd apps/api
flyctl launch
flyctl deploy
```

## API端点

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | /api/v1/convert | 上传图片创建转换任务 |
| GET | /api/v1/jobs/{id} | 查询任务状态 |
| GET | /api/v1/jobs/{id}/result | 下载结果文件 |
| DELETE | /api/v1/jobs/{id} | 取消任务 |
| WS | /ws/jobs/{id}/progress | WebSocket实时进度 |

## 特性

- ✅ 拖拽上传图片/PDF
- ✅ WebSocket实时进度推送
- ✅ 多阶段处理显示
- ✅ 可配置的转换选项
- ✅ 响应式设计
- ✅ 现代化UI (Tailwind CSS + Framer Motion)
