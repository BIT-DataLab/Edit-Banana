"""
Edit Banana FastAPI Backend
提供图片转DrawIO的API服务，支持WebSocket实时进度推送
"""

import os
import sys
import uuid
import asyncio
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)  # 添加当前目录
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))  # 添加apps目录
sys.path.insert(0, os.path.join(os.path.dirname(PROJECT_ROOT), ".."))  # 添加项目根目录

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

# 导入主流程
from main import load_config, Pipeline

# 导入路由和中间件
from routes.preview import router as preview_router
from middleware.rate_limit import PreviewRateLimitMiddleware

# ======================== 任务存储 ========================
# 简单内存存储，生产环境应使用Redis
class JobStore:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.connections: Dict[str, list] = {}  # job_id -> [websockets]

    def create_job(self, filename: str) -> str:
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "id": job_id,
            "filename": filename,
            "status": "pending",  # pending, processing, completed, failed
            "progress": 0,
            "stage": "",
            "message": "",
            "output_path": None,
            "error": None,
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
        }
        self.connections[job_id] = []
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict]:
        return self.jobs.get(job_id)

    def update_job(self, job_id: str, **kwargs):
        if job_id in self.jobs:
            self.jobs[job_id].update(kwargs)

    async def broadcast_progress(self, job_id: str, progress: int, stage: str, message: str):
        """向所有连接的WebSocket推送进度"""
        self.update_job(job_id, progress=progress, stage=stage, message=message)

        if job_id in self.connections:
            dead_connections = []
            data = {
                "type": "progress",
                "progress": progress,
                "stage": stage,
                "message": message,
                "timestamp": datetime.now().isoformat()
            }

            for ws in self.connections[job_id]:
                try:
                    await ws.send_json(data)
                except Exception:
                    dead_connections.append(ws)

            # 清理断开的连接
            for ws in dead_connections:
                self.connections[job_id].remove(ws)

job_store = JobStore()

# ======================== FastAPI应用 ========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    print("🚀 Edit Banana API 启动")
    yield
    print("👋 Edit Banana API 关闭")

app = FastAPI(
    title="Edit Banana API",
    description="图片/PDF转可编辑DrawIO - Next.js版本",
    version="2.0.0",
    lifespan=lifespan
)

# CORS配置 - 允许Next.js前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 预览端点速率限制中间件 (3 previews/hour)
app.add_middleware(PreviewRateLimitMiddleware)

# 注册预览路由
app.include_router(preview_router)

# ======================== 路由 ========================

@app.get("/health")
def health():
    return {"status": "ok", "service": "edit-banana", "version": "2.0.0"}


@app.get("/")
def root():
    return {
        "service": "Edit Banana",
        "description": "图片/PDF转可编辑DrawIO",
        "docs": "/docs",
        "version": "2.0.0"
    }


@app.post("/api/v1/convert")
async def convert(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    with_refinement: bool = False,
    with_text: bool = True
):
    """
    创建转换任务
    返回job_id，客户端用job_id连接WebSocket获取进度
    """
    # 验证文件类型
    name = file.filename or ""
    ext = Path(name).suffix.lower()
    if ext not in {".png", ".jpg", ".jpeg", ".pdf", ".bmp", ".tiff", ".webp", ".gif"}:
        raise HTTPException(400, f"不支持的文件格式: {ext}")

    # 创建任务
    job_id = job_store.create_job(name)

    # 保存上传文件
    config = load_config()
    output_dir = config.get("paths", {}).get("output_dir", "./output")
    upload_dir = os.path.join(output_dir, "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, f"{job_id}{ext}")
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # 后台处理
    background_tasks.add_task(
        process_image_task,
        job_id=job_id,
        file_path=file_path,
        output_dir=output_dir,
        with_refinement=with_refinement,
        with_text=with_text
    )

    return {
        "success": True,
        "job_id": job_id,
        "filename": name,
        "ws_url": f"/ws/jobs/{job_id}/progress"
    }


@app.get("/api/v1/jobs/{job_id}")
def get_job_status(job_id: str):
    """查询任务状态"""
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    return job


@app.get("/api/v1/jobs/{job_id}/result")
def get_job_result(job_id: str):
    """获取转换结果文件"""
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")

    if job["status"] != "completed":
        raise HTTPException(400, f"任务未完成，当前状态: {job['status']}")

    output_path = job.get("output_path")
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(404, "结果文件不存在")

    return FileResponse(
        output_path,
        filename=os.path.basename(output_path),
        media_type="application/octet-stream"
    )


@app.delete("/api/v1/jobs/{job_id}")
def cancel_job(job_id: str):
    """取消任务（标记为取消，实际处理可能仍在进行）"""
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")

    if job["status"] in ["completed", "failed"]:
        return {"success": True, "message": "任务已结束"}

    job_store.update_job(job_id, status="cancelled")
    return {"success": True, "message": "任务已取消"}


# ======================== WebSocket ========================

@app.websocket("/ws/jobs/{job_id}/progress")
async def websocket_progress(websocket: WebSocket, job_id: str):
    """WebSocket实时进度推送"""
    job = job_store.get_job(job_id)
    if not job:
        await websocket.close(code=1008, reason="任务不存在")
        return

    await websocket.accept()
    job_store.connections[job_id].append(websocket)

    # 发送当前状态
    await websocket.send_json({
        "type": "connected",
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "stage": job["stage"]
    })

    try:
        while True:
            # 保持连接，客户端可以发送ping
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        if websocket in job_store.connections[job_id]:
            job_store.connections[job_id].remove(websocket)


# ======================== 后台处理 ========================

class ProgressCallback:
    """进度回调，用于向WebSocket推送进度"""
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.stages = {
            "preprocess": (0, 10, "预处理"),
            "text_extraction": (10, 25, "文字提取"),
            "segmentation": (25, 50, "图像分割"),
            "processing": (50, 75, "元素处理"),
            "arrow": (75, 85, "箭头识别"),
            "xml_generation": (85, 95, "生成XML"),
            "refinement": (95, 100, "优化处理"),
        }

    async def on_stage_start(self, stage: str, message: str = ""):
        if stage in self.stages:
            progress = self.stages[stage][0]
            stage_name = self.stages[stage][2]
            await job_store.broadcast_progress(
                self.job_id, progress, stage_name, message or f"开始{stage_name}"
            )

    async def on_stage_progress(self, stage: str, percent: float, message: str = ""):
        """stage内进度 0-100"""
        if stage in self.stages:
            start, end, stage_name = self.stages[stage]
            progress = int(start + (end - start) * percent / 100)
            await job_store.broadcast_progress(
                self.job_id, progress, stage_name, message
            )

    async def on_stage_complete(self, stage: str, message: str = ""):
        if stage in self.stages:
            progress = self.stages[stage][1]
            stage_name = self.stages[stage][2]
            await job_store.broadcast_progress(
                self.job_id, progress, stage_name, message or f"{stage_name}完成"
            )


async def process_image_task(
    job_id: str,
    file_path: str,
    output_dir: str,
    with_refinement: bool,
    with_text: bool
):
    """后台处理任务"""
    callback = ProgressCallback(job_id)

    job_store.update_job(job_id, status="processing")

    try:
        # 预处理
        await callback.on_stage_start("preprocess", "读取图片...")
        await asyncio.sleep(0.5)  # 模拟进度
        await callback.on_stage_complete("preprocess", "预处理完成")

        # 文字提取
        if with_text:
            await callback.on_stage_start("text_extraction", "识别文字...")
            await asyncio.sleep(0.5)
            await callback.on_stage_complete("text_extraction", "文字提取完成")

        # 图像分割
        await callback.on_stage_start("segmentation", "SAM3分割中...")
        await callback.on_stage_progress("segmentation", 50, "分析图像结构...")

        # 执行实际处理
        config = load_config()
        pipeline = Pipeline(config)

        # 这里需要修改Pipeline以支持进度回调
        # 目前先用原始处理方式
        result_path = pipeline.process_image(
            file_path,
            output_dir=os.path.join(output_dir, job_id),
            with_refinement=with_refinement,
            with_text=with_text
        )

        await callback.on_stage_complete("segmentation", "分割完成")
        await callback.on_stage_complete("processing", "元素处理完成")
        await callback.on_stage_complete("arrow", "箭头识别完成")
        await callback.on_stage_complete("xml_generation", "XML生成完成")

        if with_refinement:
            await callback.on_stage_start("refinement", "优化处理...")
            await callback.on_stage_complete("refinement", "优化完成")

        # 完成任务
        job_store.update_job(
            job_id,
            status="completed",
            progress=100,
            stage="完成",
            message="转换成功",
            output_path=result_path,
            completed_at=datetime.now().isoformat()
        )

        # 通知客户端
        await job_store.broadcast_progress(job_id, 100, "完成", "转换成功！")

    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"任务 {job_id} 处理失败: {error_msg}")
        traceback.print_exc()

        job_store.update_job(
            job_id,
            status="failed",
            error=error_msg,
            completed_at=datetime.now().isoformat()
        )

        await job_store.broadcast_progress(job_id, 0, "失败", error_msg)


def main():
    """启动服务器"""
    uvicorn.run(
        "main_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
