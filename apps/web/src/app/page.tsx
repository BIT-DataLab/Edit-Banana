"use client"

import { Navbar } from "./sections/navbar"
import { Hero } from "./sections/hero"
import { UploadSection } from "./sections/upload-section"
import { Features } from "./sections/features"
import { ExampleShowcase } from "./sections/example-showcase"
import { Footer } from "./sections/footer"
import { HistorySection } from "./sections/history-section"
import { ConversionHistoryProvider, useConversionHistoryContext } from "@/components/history/conversion-history-provider"

function HomeContent() {
  const { history, removeHistoryItem, clearHistory } = useConversionHistoryContext()

  return (
    <div className="min-h-screen bg-white">
      <Navbar />
      <main>
        <Hero />
        <UploadSection />
        <HistorySection
          history={history}
          onRemove={removeHistoryItem}
          onClear={clearHistory}
        />
        <Features />
        <ExampleShowcase />
      </main>
      <Footer />
    </div>
  )
}

export default function Home() {
  return (
    <ConversionHistoryProvider>
      <HomeContent />
    </ConversionHistoryProvider>
  )
}
