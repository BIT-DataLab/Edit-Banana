"use client"

import { useEffect, useState } from "react"
import { AlertCircle, RefreshCw } from "lucide-react"
import { Button } from "@/components/ui/button"

interface ErrorBoundaryProps {
  children: React.ReactNode
}

interface ErrorState {
  hasError: boolean
  error?: Error
}

export function ErrorBoundary({ children }: ErrorBoundaryProps) {
  const [error, setError] = useState<ErrorState>({ hasError: false })

  useEffect(() => {
    const handleError = (event: ErrorEvent) => {
      console.error("Error caught by boundary:", event.error)
      setError({ hasError: true, error: event.error })
    }

    window.addEventListener("error", handleError)
    return () => window.removeEventListener("error", handleError)
  }, [])

  const handleReset = () => {
    setError({ hasError: false })
    window.location.reload()
  }

  if (error.hasError) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
        <div className="max-w-md w-full bg-white rounded-xl shadow-lg p-8 text-center">
          <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <AlertCircle className="w-8 h-8 text-red-500" />
          </div>
          <h2 className="text-xl font-semibold text-gray-900 mb-2">
            出错了
          </h2>
          <p className="text-gray-600 mb-6">
            {error.error?.message || "页面加载时发生错误"}
          </p>
          <Button onClick={handleReset} className="gap-2">
            <RefreshCw className="w-4 h-4" />
            刷新页面
          </Button>
        </div>
      </div>
    )
  }

  return <>{children}</>
}

// API错误处理
export class APIError extends Error {
  constructor(
    message: string,
    public statusCode: number,
    public response?: Response
  ) {
    super(message)
    this.name = "APIError"
  }
}

export async function handleAPIError(response: Response): Promise<void> {
  if (!response.ok) {
    let message = "请求失败"
    try {
      const data = await response.json()
      message = data.detail || data.message || message
    } catch {
      message = response.statusText || message
    }
    throw new APIError(message, response.status, response)
  }
}
