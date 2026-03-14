"use client"

<<<<<<< HEAD
import { Banana, History } from "lucide-react"
=======
import { Banana } from "lucide-react"
>>>>>>> pr-35

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
<<<<<<< HEAD
              href="#history"
              className="flex items-center gap-1.5 text-sm text-gray-600 hover:text-gray-900 transition-colors"
            >
              <History className="w-4 h-4" />
              History
            </a>
            <a
=======
>>>>>>> pr-35
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
