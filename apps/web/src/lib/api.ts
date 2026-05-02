import { API_BASE_URL } from "./config"
import { UploadResponse, Job } from "./types"
import { historyStorage, HistoryItem } from "./storage"

// API错误类
export class APIError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public response?: Response
  ) {
    super(message)
    this.name = "APIError"
  }
}

// 统一处理响应
async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let message = "请求失败"
    let data: any = {}
    try {
      data = await response.json()
      message = data.detail || data.message || message
    } catch {
      message = response.statusText || message
    }
    throw new APIError(message, response.status, response)
  }
  return response.json()
}

// 上传文件
export async function uploadFile(
  file: File,
  withText: boolean = true,
  withRefinement: boolean = false
): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append("file", file)
  formData.append("with_text", String(withText))
  formData.append("with_refinement", String(withRefinement))

  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/convert`, {
      method: "POST",
      body: formData,
    })

    const data = await handleResponse<UploadResponse>(response)

    // 添加到历史记录
    historyStorage.add({
      id: data.job_id,
      filename: file.name,
      status: "pending",
      created_at: new Date().toISOString(),
    })

    return data
  } catch (error) {
    if (error instanceof APIError) {
      throw error
    }
    throw new APIError(error instanceof Error ? error.message : "上传失败")
  }
}

// 获取任务状态
export async function getJobStatus(jobId: string): Promise<Job> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/jobs/${jobId}`)
    const data = await handleResponse<Job>(response)

    // 更新历史记录
    historyStorage.update(data.id, {
      status: data.status,
      completed_at: data.completed_at,
      error: data.error,
    })

    return data
  } catch (error) {
    if (error instanceof APIError) {
      throw error
    }
    throw new APIError(error instanceof Error ? error.message : "获取任务状态失败")
  }
}

// 下载结果
export async function downloadResult(jobId: string): Promise<Blob> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/jobs/${jobId}/result`)

    if (!response.ok) {
      await handleResponse(response)
    }

    return response.blob()
  } catch (error) {
    if (error instanceof APIError) {
      throw error
    }
    throw new APIError(error instanceof Error ? error.message : "下载失败")
  }
}

// 取消任务
export async function cancelJob(jobId: string): Promise<void> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/jobs/${jobId}`, {
      method: "DELETE",
    })

    await handleResponse(response)

    // 更新历史记录
    historyStorage.update(jobId, { status: "cancelled" as Job["status"] })
  } catch (error) {
    if (error instanceof APIError) {
      throw error
    }
    throw new APIError(error instanceof Error ? error.message : "取消失败")
  }
}

// 批量获取任务状态（用于历史页面刷新）
export async function refreshJobsStatus(jobIds: string[]): Promise<Job[]> {
  const promises = jobIds.map((id) =>
    getJobStatus(id).catch((error) => {
      console.error(`获取任务 ${id} 状态失败:`, error)
      return null
    })
  )

  const results = await Promise.all(promises)
  return results.filter((job): job is Job => job !== null)
}
