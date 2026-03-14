// 任务状态类型
export interface Job {
  id: string
  filename: string
  status: "pending" | "processing" | "completed" | "failed" | "cancelled"
  progress: number
  stage: string
  message: string
  output_path?: string
  error?: string
  created_at: string
  completed_at?: string
}

// WebSocket消息类型
export interface ProgressMessage {
  type: "connected" | "progress" | "pong" | "error"
  job_id?: string
  status?: string
  progress?: number
  stage?: string
  message?: string
  timestamp?: string
}

// 上传响应
export interface UploadResponse {
  success: boolean
  job_id: string
  filename: string
  ws_url: string
}
