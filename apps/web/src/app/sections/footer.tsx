"use client"

import { Banana, Github, Twitter } from "lucide-react"

export function Footer() {
  return (
    <footer className="w-full bg-white border-t border-gray-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          {/* Logo */}
          <div className="flex items-center gap-2">
            <div className="p-2 bg-gradient-to-br from-yellow-400 to-yellow-600 rounded-lg">
              <Banana className="w-5 h-5 text-white" />
            </div>
            <span className="text-lg font-bold text-gray-900">Edit Banana</span>
          </div>

          {/* Links */}
          <div className="flex items-center gap-6">
            <a
              href="https://github.com/yourusername/edit-banana"
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-500 hover:text-gray-900 transition-colors"
            >
              <Github className="w-5 h-5" />
            </a>
            <a
              href="#"
              className="text-gray-500 hover:text-gray-900 transition-colors"
            >
              <Twitter className="w-5 h-5" />
            </a>
          </div>

          {/* Copyright */}
          <p className="text-sm text-gray-500">
            &copy; {new Date().getFullYear()} Edit Banana. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  )
}
