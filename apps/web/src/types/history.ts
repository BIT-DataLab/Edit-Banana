export interface ConversionHistoryItem {
  id: string
  jobId: string
  filename: string
  originalUrl: string
  resultUrl: string | null
  status: "pending" | "completed" | "failed" | "cancelled"
  createdAt: string
  fileSize?: number
  error?: string
}

export interface HistoryStorage {
  items: ConversionHistoryItem[]
  lastUpdated: string
}

export const HISTORY_STORAGE_KEY = "editbanana_conversion_history"
export const MAX_HISTORY_ITEMS = 50
