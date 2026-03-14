"use client"

import { useState } from "react"
import { ArrowRight, ZoomIn } from "lucide-react"
import { motion } from "framer-motion"

interface Example {
  id: number
  title: string
  description: string
  beforeImage: string
  afterImage: string
}

const examples: Example[] = [
  {
    id: 1,
    title: "Flowchart Conversion",
    description: "Transform static flowchart images into editable diagrams",
    beforeImage: "/examples/flowchart-before.jpg",
    afterImage: "/examples/flowchart-after.png",
  },
  {
    id: 2,
    title: "Architecture Diagram",
    description: "Convert complex architecture diagrams to editable format",
    beforeImage: "/examples/architecture-before.png",
    afterImage: "/examples/architecture-after.png",
  },
  {
    id: 3,
    title: "Network Topology",
    description: "Turn network diagrams into fully editable assets",
    beforeImage: "/examples/network-before.jpg",
    afterImage: "/examples/network-after.png",
  },
]

export function ExampleShowcase() {
  const [activeExample, setActiveExample] = useState<Example>(examples[0])
  const [isHovering, setIsHovering] = useState(false)

  return (
    <section id="examples" className="w-full py-20 bg-gray-50">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            See the Magic in Action
          </h2>
          <p className="text-gray-600 max-w-2xl mx-auto">
            Our AI-powered conversion transforms static images into fully editable diagrams.
            Here are some examples of what EditBanana can do.
          </p>
        </div>

        {/* Example Selector */}
        <div className="flex flex-wrap justify-center gap-3 mb-10">
          {examples.map((example) => (
            <button
              key={example.id}
              onClick={() => setActiveExample(example)}
              className={`px-5 py-2.5 rounded-full text-sm font-medium transition-all duration-200 ${
                activeExample.id === example.id
                  ? "bg-yellow-500 text-white shadow-lg shadow-yellow-500/30"
                  : "bg-white text-gray-600 hover:bg-gray-100 border border-gray-200"
              }`}
            >
              {example.title}
            </button>
          ))}
        </div>

        {/* Before/After Comparison */}
        <motion.div
          key={activeExample.id}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="bg-white rounded-2xl shadow-xl overflow-hidden"
        >
          <div className="p-6 border-b border-gray-100">
            <h3 className="text-xl font-semibold text-gray-900 mb-2">
              {activeExample.title}
            </h3>
            <p className="text-gray-600">{activeExample.description}</p>
          </div>

          <div className="grid md:grid-cols-2 gap-0">
            {/* Before */}
            <div className="relative group">
              <div className="absolute top-4 left-4 z-10">
                <span className="px-3 py-1.5 bg-gray-900/80 text-white text-xs font-medium rounded-full">
                  Before
                </span>
              </div>
              <div
                className="relative overflow-hidden bg-gray-100 aspect-[4/3]"
                onMouseEnter={() => setIsHovering(true)}
                onMouseLeave={() => setIsHovering(false)}
              >
                <img
                  src={activeExample.beforeImage}
                  alt={`${activeExample.title} - Before`}
                  className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
                />
                <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors duration-300 flex items-center justify-center">
                  <ZoomIn className="w-8 h-8 text-white opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                </div>
              </div>
              <div className="p-4 bg-gray-50 border-t border-gray-100">
                <p className="text-sm text-gray-500">
                  Static image that cannot be edited
                </p>
              </div>
            </div>

            {/* Arrow indicator on mobile */}
            <div className="flex md:hidden items-center justify-center py-4">
              <ArrowRight className="w-6 h-6 text-yellow-500 rotate-90" />
            </div>

            {/* After */}
            <div className="relative group">
              <div className="absolute top-4 left-4 z-10">
                <span className="px-3 py-1.5 bg-yellow-500 text-white text-xs font-medium rounded-full shadow-lg shadow-yellow-500/30">
                  After
                </span>
              </div>
              <div className="relative overflow-hidden bg-yellow-50 aspect-[4/3]">
                <img
                  src={activeExample.afterImage}
                  alt={`${activeExample.title} - After`}
                  className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
                />
                <div className="absolute inset-0 bg-yellow-500/0 group-hover:bg-yellow-500/10 transition-colors duration-300 flex items-center justify-center">
                  <ZoomIn className="w-8 h-8 text-yellow-600 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                </div>
              </div>
              <div className="p-4 bg-yellow-50 border-t border-yellow-100">
                <p className="text-sm text-yellow-700 font-medium">
                  Fully editable diagram - modify shapes, text, and connections
                </p>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Trust badges */}
        <div className="mt-12 grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
          <div className="p-4">
            <div className="text-3xl font-bold text-yellow-500 mb-1">99%</div>
            <div className="text-sm text-gray-600">Accuracy Rate</div>
          </div>
          <div className="p-4">
            <div className="text-3xl font-bold text-yellow-500 mb-1">10s</div>
            <div className="text-sm text-gray-600">Avg. Processing</div>
          </div>
          <div className="p-4">
            <div className="text-3xl font-bold text-yellow-500 mb-1">50K+</div>
            <div className="text-sm text-gray-600">Diagrams Converted</div>
          </div>
          <div className="p-4">
            <div className="text-3xl font-bold text-yellow-500 mb-1">4.9★</div>
            <div className="text-sm text-gray-600">User Rating</div>
          </div>
        </div>
      </div>
    </section>
  )
}
