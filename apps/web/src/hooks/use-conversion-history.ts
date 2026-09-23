"use client"

import { useState, useEffect, useCallback, useRef } from "react"
import type { ConversionHistoryItem, HistoryStorage } from "@/types/history"
import { HISTORY_STORAGE_KEY, MAX_HISTORY_ITEMS } from "@/types/history"

function loadHistoryFromStorage(): ConversionHistoryItem[] {
  try {
    const stored = localStorage.getItem(HISTORY_STORAGE_KEY)
    if (stored) {
      const parsed: HistoryStorage = JSON.parse(stored)
      return parsed.items || []
    }
  } catch (error) {
    console.error("Failed to load conversion history:", error)
  }
  return []
}

export function useConversionHistory() {
  const [history, setHistory] = useState<ConversionHistoryItem[]>(loadHistoryFromStorage)
  const isFirstRender = useRef(true)

  // Save history to localStorage whenever it changes
  useEffect(() => {
    if (isFirstRender.current) {
      isFirstRender.current = false
      return
    }

    try {
      const storage: HistoryStorage = {
        items: history,
        lastUpdated: new Date().toISOString(),
      }
      localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(storage))
    } catch (error) {
      console.error("Failed to save conversion history:", error)
    }
  }, [history])

  const addHistoryItem = useCallback((item: Omit<ConversionHistoryItem, "id" | "createdAt">) => {
    const newItem: ConversionHistoryItem = {
      ...item,
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      createdAt: new Date().toISOString(),
    }

    setHistory((prev) => {
      // Remove any existing item with the same jobId
      const filtered = prev.filter((i) => i.jobId !== item.jobId)
      // Add new item at the beginning
      const updated = [newItem, ...filtered]
      // Limit to max items
      return updated.slice(0, MAX_HISTORY_ITEMS)
    })

    return newItem
  }, [])

  const updateHistoryItem = useCallback((jobId: string, updates: Partial<ConversionHistoryItem>) => {
    setHistory((prev) =>
      prev.map((item) => (item.jobId === jobId ? { ...item, ...updates } : item))
    )
  }, [])

  const removeHistoryItem = useCallback((id: string) => {
    setHistory((prev) => prev.filter((item) => item.id !== id))
  }, [])

  const clearHistory = useCallback(() => {
    if (typeof window !== "undefined" && window.confirm("Are you sure you want to clear all history?")) {
      setHistory([])
    }
  }, [])

  const getHistoryItemByJobId = useCallback(
    (jobId: string) => {
      return history.find((item) => item.jobId === jobId)
    },
    [history]
  )

  return {
    history,
    isLoaded: true,
    addHistoryItem,
    updateHistoryItem,
    removeHistoryItem,
    clearHistory,
    getHistoryItemByJobId,
  }
}
