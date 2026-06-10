"use client"

import { Banana } from "lucide-react"

export function Navbar() {
  return (
    <nav className="sticky top-0 z-50 w-full bg-white/80 backdrop-blur-md border-b border-gray-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center gap-2">
            <div className="p-2 bg-gradient-to-br from-yellow-400 to-yellow-600 rounded-lg">
              <Banana className="w-6 h-6 text-white" />
            </div>
            <span className="text-xl font-bold text-gray-900">Edit Banana</span>
          </div>
          <div className="flex items-center gap-4">
            <a
              href="https://github.com/yourusername/edit-banana"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-gray-600 hover:text-gray-900 transition-colors"
            >
              GitHub
            </a>
          </div>
        </div>
      </div>
    </nav>
  )
}
