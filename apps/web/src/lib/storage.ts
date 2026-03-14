import { Job } from "./types"

const STORAGE_KEY = "edit-banana-history"
const MAX_HISTORY_ITEMS = 50

export interface HistoryItem {
  id: string
  filename: string
  status: Job["status"]
  created_at: string
  completed_at?: string
  error?: string
}

export const historyStorage = {
  // 获取所有历史记录
  getAll(): HistoryItem[] {
    try {
      const stored = localStorage.getItem(STORAGE_KEY)
      if (stored) {
        return JSON.parse(stored)
      }
    } catch (e) {
      console.error("读取历史记录失败:", e)
    }
    return []
  },

  // 添加新记录
  add(item: HistoryItem): void {
    try {
      const history = this.getAll()
      // 检查是否已存在
      const existingIndex = history.findIndex((h) => h.id === item.id)
      if (existingIndex >= 0) {
        history[existingIndex] = item
      } else {
        history.unshift(item)
      }
      // 限制数量
      if (history.length > MAX_HISTORY_ITEMS) {
        history.pop()
      }
      localStorage.setItem(STORAGE_KEY, JSON.stringify(history))
    } catch (e) {
      console.error("保存历史记录失败:", e)
    }
  },

  // 删除记录
  remove(id: string): void {
    try {
      const history = this.getAll().filter((h) => h.id !== id)
      localStorage.setItem(STORAGE_KEY, JSON.stringify(history))
    } catch (e) {
      console.error("删除历史记录失败:", e)
    }
  },

  // 清空记录
  clear(): void {
    try {
      localStorage.removeItem(STORAGE_KEY)
    } catch (e) {
      console.error("清空历史记录失败:", e)
    }
  },

  // 更新记录状态
  update(id: string, updates: Partial<HistoryItem>): void {
    try {
      const history = this.getAll()
      const index = history.findIndex((h) => h.id === id)
      if (index >= 0) {
        history[index] = { ...history[index], ...updates }
        localStorage.setItem(STORAGE_KEY, JSON.stringify(history))
      }
    } catch (e) {
      console.error("更新历史记录失败:", e)
    }
  },
}
