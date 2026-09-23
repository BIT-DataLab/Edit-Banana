"use client"

import {
  Wand2,
  FileType,
  Zap,
  Download,
  Shield,
  Globe,
} from "lucide-react"

const features = [
  {
    name: "AI-Powered Recognition",
    description:
      "Advanced SAM3 segmentation and OCR technology accurately identify shapes, text, and arrows in your images.",
    icon: Wand2,
  },
  {
    name: "Multiple Formats",
    description:
      "Export to Draw.io XML, PowerPoint PPTX, or other editable formats. Compatible with all major diagram tools.",
    icon: FileType,
  },
  {
    name: "Fast Processing",
    description:
      "Get your editable diagrams in seconds. Our optimized pipeline ensures quick turnaround times.",
    icon: Zap,
  },
  {
    name: "Easy Export",
    description:
      "One-click download of your converted diagrams. No registration or account required.",
    icon: Download,
  },
  {
    name: "Privacy First",
    description:
      "Your files are processed securely and automatically deleted after conversion. We never store your data.",
    icon: Shield,
  },
  {
    name: "Works Everywhere",
    description:
      "Access from any device with a web browser. No software installation needed.",
    icon: Globe,
  },
]

export function Features() {
  return (
    <section id="features" className="w-full py-20 bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            Everything You Need
          </h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Powerful features to make diagram conversion simple and accurate
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature) => (
            <div
              key={feature.name}
              className="relative p-6 bg-white rounded-2xl border border-gray-100 shadow-sm hover:shadow-md transition-shadow"
            >
              <div className="w-12 h-12 bg-yellow-100 rounded-xl flex items-center justify-center mb-4">
                <feature.icon className="w-6 h-6 text-yellow-600" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                {feature.name}
              </h3>
              <p className="text-gray-600 text-sm leading-relaxed">
                {feature.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
