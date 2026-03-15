"use client"

import { useState, useEffect, useRef, useCallback } from "react"
import { WS_BASE_URL } from "./config"
import { ProgressMessage, Job } from "./types"

interface UseWebSocketOptions {
  jobId: string | null
  onProgress?: (progress: number, stage: string, message: string) => void
  onComplete?: (job: Job) => void
  onError?: (error: string) => void
}

export function useWebSocket({
  jobId,
  onProgress,
  onComplete,
  onError,
}: UseWebSocketOptions) {
  const [isConnected, setIsConnected] = useState(false)
  const [progress, setProgress] = useState(0)
  const [stage, setStage] = useState("")
  const [message, setMessage] = useState("")
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  const connect = useCallback(() => {
    if (!jobId) return

    // 清理旧连接
    if (wsRef.current) {
      wsRef.current.close()
    }

    const ws = new WebSocket(`${WS_BASE_URL}/ws/jobs/${jobId}/progress`)
    wsRef.current = ws

    ws.onopen = () => {
      setIsConnected(true)
    }

    ws.onmessage = (event) => {
      try {
        const data: ProgressMessage = JSON.parse(event.data)

        if (data.type === "progress") {
          setProgress(data.progress || 0)
          setStage(data.stage || "")
          setMessage(data.message || "")
          onProgress?.(data.progress || 0, data.stage || "", data.message || "")

          // 检查是否完成
          if (data.progress === 100) {
            // 延迟获取最终状态
            setTimeout(async () => {
              const response = await fetch(`/api/jobs/${jobId}`)
              if (response.ok) {
                const job = await response.json()
                onComplete?.(job)
              }
            }, 500)
          }
        } else if (data.type === "connected") {
          setIsConnected(true)
          if (data.progress) setProgress(data.progress)
          if (data.stage) setStage(data.stage)
        }
      } catch (err) {
        console.error("WebSocket消息解析失败:", err)
      }
    }

    ws.onclose = () => {
      setIsConnected(false)
    }

    ws.onerror = (error) => {
      console.error("WebSocket错误:", error)
      onError?.("连接错误")
      setIsConnected(false)
    }
  }, [jobId, onProgress, onComplete, onError])

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }
    setIsConnected(false)
  }, [])

  useEffect(() => {
    if (jobId) {
      connect()
    }

    return () => {
      disconnect()
    }
  }, [jobId, connect, disconnect])

  // 定期发送ping保持连接
  useEffect(() => {
    if (!isConnected || !wsRef.current) return

    const pingInterval = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send("ping")
      }
    }, 30000)

    return () => clearInterval(pingInterval)
  }, [isConnected])

  return {
    isConnected,
    progress,
    stage,
    message,
    connect,
    disconnect,
  }
}
