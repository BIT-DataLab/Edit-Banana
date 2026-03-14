<<<<<<< HEAD
"use client"

=======
>>>>>>> pr-35
import { Navbar } from "./sections/navbar"
import { Hero } from "./sections/hero"
import { UploadSection } from "./sections/upload-section"
import { Features } from "./sections/features"
import { ExampleShowcase } from "./sections/example-showcase"
import { Footer } from "./sections/footer"
<<<<<<< HEAD
import { HistorySection } from "./sections/history-section"
import { ConversionHistoryProvider, useConversionHistoryContext } from "@/components/history/conversion-history-provider"

function HomeContent() {
  const { history, removeHistoryItem, clearHistory } = useConversionHistoryContext()

=======

export default function Home() {
>>>>>>> pr-35
  return (
    <div className="min-h-screen bg-white">
      <Navbar />
      <main>
        <Hero />
        <UploadSection />
<<<<<<< HEAD
        <HistorySection
          history={history}
          onRemove={removeHistoryItem}
          onClear={clearHistory}
        />
=======
>>>>>>> pr-35
        <Features />
        <ExampleShowcase />
      </main>
      <Footer />
    </div>
  )
}
<<<<<<< HEAD

export default function Home() {
  return (
    <ConversionHistoryProvider>
      <HomeContent />
    </ConversionHistoryProvider>
  )
}
=======
>>>>>>> pr-35
