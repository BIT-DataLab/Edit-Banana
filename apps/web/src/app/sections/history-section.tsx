"use client"

import { motion } from "framer-motion"
import { History, Download, Trash2, Clock, CheckCircle, XCircle, AlertCircle, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import type { ConversionHistoryItem } from "@/types/history"

interface HistorySectionProps {
  history: ConversionHistoryItem[]
  onRemove: (id: string) => void
  onClear: () => void
}

function formatDate(dateString: string): string {
  const date = new Date(dateString)
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  })
}

function formatFileSize(bytes?: number): string {
  if (!bytes) return ""
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function StatusIcon({ status }: { status: ConversionHistoryItem["status"] }) {
  switch (status) {
    case "pending":
      return <Loader2 className="w-4 h-4 animate-spin text-yellow-500" />
    case "completed":
      return <CheckCircle className="w-4 h-4 text-green-500" />
    case "failed":
      return <XCircle className="w-4 h-4 text-red-500" />
    case "cancelled":
      return <AlertCircle className="w-4 h-4 text-gray-500" />
    default:
      return null
  }
}

export function HistorySection({ history, onRemove, onClear }: HistorySectionProps) {
  if (history.length === 0) {
    return null
  }

  return (
    <section id="history" className="w-full py-20 bg-white">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          <div className="flex items-center justify-between mb-8">
            <div className="flex items-center gap-3">
              <History className="w-6 h-6 text-yellow-500" />
              <h2 className="text-2xl font-bold text-gray-900">Conversion History</h2>
              <span className="px-2.5 py-0.5 bg-gray-100 text-gray-600 text-sm rounded-full">
                {history.length}
              </span>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={onClear}
              className="text-gray-500 hover:text-red-600"
            >
              <Trash2 className="w-4 h-4 mr-1" />
              Clear All
            </Button>
          </div>

          <div className="space-y-3">
            {history.map((item, index) => (
              <motion.div
                key={item.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
                className="group flex items-center gap-4 p-4 bg-gray-50 rounded-xl border border-gray-100 hover:border-yellow-200 hover:bg-yellow-50/50 transition-all"
              >
                {/* Status Icon */}
                <div className="flex-shrink-0">
                  <StatusIcon status={item.status} />
                </div>

                {/* File Info */}
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-gray-900 truncate">{item.filename}</p>
                  <div className="flex items-center gap-3 text-sm text-gray-500">
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {formatDate(item.createdAt)}
                    </span>
                    {item.fileSize && (
                      <span className="text-gray-400">{formatFileSize(item.fileSize)}</span>
                    )}
                  </div>
                  {item.error && (
                    <p className="text-sm text-red-600 mt-1">{item.error}</p>
                  )}
                </div>

                {/* Actions */}
                <div className="flex items-center gap-2">
                  {item.status === "completed" && item.resultUrl && (
                    <a
                      href={item.resultUrl}
                      download
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 px-3 py-1.5 text-sm font-medium text-yellow-600 hover:text-yellow-700 hover:bg-yellow-100 rounded-md transition-colors"
                    >
                      <Download className="w-4 h-4" />
                      Download
                    </a>
                  )}
                  <button
                    onClick={() => onRemove(item.id)}
                    className="p-2 text-gray-400 hover:text-red-600 opacity-0 group-hover:opacity-100 transition-opacity rounded-md hover:bg-gray-100"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  )
}
