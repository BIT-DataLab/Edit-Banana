"use client"

import { useState, useEffect, useCallback } from "react"
import { X, Clock, Download, Lock } from "lucide-react"
import { Button } from "@/components/ui/button"

interface PreviewModalProps {
  isOpen: boolean
  onClose: () => void
  previewUrl: string | null
  expiresAt: string | null
  jobId: string | null
  onDownload: () => void
}

export function PreviewModal({
  isOpen,
  onClose,
  previewUrl,
  expiresAt,
  jobId,
  onDownload,
}: PreviewModalProps) {
  const [timeRemaining, setTimeRemaining] = useState<number>(0)
  const [isExpired, setIsExpired] = useState(false)

  // Calculate time remaining
  useEffect(() => {
    if (!expiresAt || !isOpen) return

    const calculateTimeRemaining = () => {
      const expiry = new Date(expiresAt).getTime()
      const now = Date.now()
      const remaining = Math.max(0, Math.floor((expiry - now) / 1000))
      setTimeRemaining(remaining)
      setIsExpired(remaining === 0)

      if (remaining === 0) {
        onClose()
      }
    }

    calculateTimeRemaining()
    const interval = setInterval(calculateTimeRemaining, 1000)

    return () => clearInterval(interval)
  }, [expiresAt, isOpen, onClose])

  // Disable right-click on preview image
  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    return false
  }, [])

  // Disable drag on preview image
  const handleDragStart = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    return false
  }, [])

  // Format seconds to MM:SS
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, "0")}`
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm">
      <div className="relative w-full max-w-4xl mx-4 bg-white rounded-2xl shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">
              Preview Your Diagram
            </h3>
            <p className="text-sm text-gray-500">
              Watermarked preview - expires in{" "}
              <span className="font-medium text-yellow-600">
                {formatTime(timeRemaining)}
              </span>
            </p>
          </div>
          <div className="flex items-center gap-3">
            {/* Countdown timer */}
            <div className="flex items-center gap-2 px-3 py-1.5 bg-yellow-50 rounded-full">
              <Clock className="w-4 h-4 text-yellow-600" />
              <span className="text-sm font-medium text-yellow-700">
                {formatTime(timeRemaining)}
              </span>
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-full hover:bg-gray-100 transition-colors"
            >
              <X className="w-5 h-5 text-gray-500" />
            </button>
          </div>
        </div>

        {/* Preview Image Container */}
        <div className="relative bg-gray-100">
          {previewUrl ? (
            <div className="relative">
              {/* Preview Image */}
              <img
                src={previewUrl}
                alt="Preview"
                className="w-full h-auto max-h-[60vh] object-contain"
                onContextMenu={handleContextMenu}
                onDragStart={handleDragStart}
                style={{
                  WebkitUserSelect: "none",
                  userSelect: "none",
                  WebkitTouchCallout: "none",
                }}
              />

              {/* Blur Overlay - Bottom 30% */}
              <div
                className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 via-black/60 to-transparent backdrop-blur-md flex flex-col items-center justify-center"
                style={{ height: "30%" }}
              >
                <Lock className="w-8 h-8 text-white/80 mb-3" />
                <p className="text-white text-lg font-semibold mb-2">
                  Pay $4.99 to unlock full quality
                </p>
                <p className="text-white/70 text-sm mb-4">
                  Get the high-resolution, watermark-free version
                </p>
                <Button
                  onClick={onDownload}
                  className="bg-yellow-500 hover:bg-yellow-600 text-white font-semibold px-6"
                >
                  <Download className="w-4 h-4 mr-2" />
                  Download Full Quality
                </Button>
              </div>

              {/* Watermark Notice */}
              <div className="absolute top-4 left-4 px-3 py-1.5 bg-black/50 backdrop-blur-sm rounded-full">
                <p className="text-xs text-white/90 font-medium">
                  EditBanana Preview - Watermarked
                </p>
              </div>

              {/* View-only notice */}
              <div className="absolute top-4 right-4 px-3 py-1.5 bg-red-500/80 backdrop-blur-sm rounded-full">
                <p className="text-xs text-white font-medium">
                  View Only
                </p>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-64">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-yellow-500" />
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 bg-gray-50 border-t border-gray-200">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="text-sm text-gray-500">
              <p>
                This preview will expire in{" "}
                <span className="font-medium text-gray-700">
                  {formatTime(timeRemaining)}
                </span>
              </p>
              <p className="text-xs text-gray-400 mt-1">
                Previews are limited to 3 per hour per IP address
              </p>
            </div>
            <div className="flex gap-3">
              <Button
                variant="outline"
                onClick={onClose}
                className="border-gray-300 text-gray-700 hover:bg-gray-100"
              >
                Close
              </Button>
              <Button
                onClick={onDownload}
                className="bg-yellow-500 hover:bg-yellow-600 text-white"
              >
                <Download className="w-4 h-4 mr-2" />
                Download Full Quality
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
