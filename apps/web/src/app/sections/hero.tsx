"use client"

import { ArrowRight, Sparkles } from "lucide-react"
import { Button } from "@/components/ui/button"

export function Hero() {
  const scrollToUpload = () => {
    document.getElementById("upload")?.scrollIntoView({ behavior: "smooth" })
  }

  return (
    <section className="relative w-full py-20 lg:py-32 overflow-hidden">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-yellow-50 via-white to-yellow-50/30" />
      <div className="absolute top-0 right-0 w-1/2 h-1/2 bg-gradient-to-bl from-yellow-200/20 to-transparent rounded-full blur-3xl" />
      <div className="absolute bottom-0 left-0 w-1/3 h-1/3 bg-gradient-to-tr from-yellow-300/10 to-transparent rounded-full blur-3xl" />

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-2 mb-8 bg-yellow-100 text-yellow-800 rounded-full text-sm font-medium">
            <Sparkles className="w-4 h-4" />
            <span>AI-Powered Diagram Conversion</span>
          </div>

          {/* Heading */}
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-gray-900 tracking-tight mb-6">
            Turn Images into
            <span className="block text-transparent bg-clip-text bg-gradient-to-r from-yellow-600 to-yellow-400">
              Editable Diagrams
            </span>
          </h1>

          {/* Subheading */}
          <p className="max-w-2xl mx-auto text-lg sm:text-xl text-gray-600 mb-10">
            Upload any image or PDF and convert it to editable Draw.io XML or PowerPoint format.
            Powered by SAM3 segmentation and OCR technology.
          </p>

          {/* CTA Button */}
          <Button
            onClick={scrollToUpload}
            size="lg"
            className="bg-gradient-to-r from-yellow-500 to-yellow-600 hover:from-yellow-600 hover:to-yellow-700 text-white font-semibold px-8 py-6 text-lg rounded-full shadow-lg shadow-yellow-500/25 transition-all hover:shadow-xl hover:shadow-yellow-500/30"
          >
            Start Converting
            <ArrowRight className="ml-2 w-5 h-5" />
          </Button>

          {/* Stats */}
          <div className="mt-16 grid grid-cols-3 gap-8 max-w-md mx-auto">
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-900">10s</div>
              <div className="text-sm text-gray-500">Avg. Processing</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-900">99%</div>
              <div className="text-sm text-gray-500">Accuracy</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-900">Free</div>
              <div className="text-sm text-gray-500">to Try</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
