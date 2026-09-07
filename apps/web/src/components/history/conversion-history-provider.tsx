"use client"

import { createContext, useContext, ReactNode } from "react"
import { useConversionHistory } from "@/hooks/use-conversion-history"
import type { ConversionHistoryItem } from "@/types/history"

interface ConversionHistoryContextType {
  history: ConversionHistoryItem[]
  isLoaded: boolean
  addHistoryItem: (item: Omit<ConversionHistoryItem, "id" | "createdAt">) => ConversionHistoryItem
  updateHistoryItem: (jobId: string, updates: Partial<ConversionHistoryItem>) => void
  removeHistoryItem: (id: string) => void
  clearHistory: () => void
  getHistoryItemByJobId: (jobId: string) => ConversionHistoryItem | undefined
}

const ConversionHistoryContext = createContext<ConversionHistoryContextType | undefined>(undefined)

export function ConversionHistoryProvider({ children }: { children: ReactNode }) {
  const historyState = useConversionHistory()

  return (
    <ConversionHistoryContext.Provider value={historyState}>
      {children}
    </ConversionHistoryContext.Provider>
  )
}

export function useConversionHistoryContext() {
  const context = useContext(ConversionHistoryContext)
  if (context === undefined) {
    throw new Error("useConversionHistoryContext must be used within a ConversionHistoryProvider")
  }
  return context
}
