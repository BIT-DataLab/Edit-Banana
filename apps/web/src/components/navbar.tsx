"use client"

import Link from "next/link"
import { useState } from "react"
import { Menu, X, Github, Banana } from "lucide-react"
import { Button } from "@/components/ui/button"

export function Navbar() {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <nav className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-200">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2">
            <div className="w-8 h-8 bg-yellow-500 rounded-lg flex items-center justify-center">
              <Banana className="w-5 h-5 text-white" />
            </div>
            <span className="font-bold text-xl text-gray-900">Edit Banana</span>
          </Link>

          {/* Desktop Nav */}
          <div className="hidden md:flex items-center gap-8">
            <Link href="/" className="text-gray-600 hover:text-gray-900 transition-colors">
              首页
            </Link>
            <Link href="/upload" className="text-gray-600 hover:text-gray-900 transition-colors">
              转换
            </Link>
            <Link href="/history" className="text-gray-600 hover:text-gray-900 transition-colors">
              历史
            </Link>
            <a
              href="https://github.com/BIT-DataLab/Edit-Banana"
              target="_blank"
              rel="noopener"
              className="text-gray-600 hover:text-gray-900 transition-colors"
            >
              <Github className="w-5 h-5" />
            </a>
            <Link href="/upload">
              <Button size="sm">开始转换</Button>
            </Link>
          </div>

          {/* Mobile Menu Button */}
          <button
            className="md:hidden p-2"
            onClick={() => setIsOpen(!isOpen)}
          >
            {isOpen ? (
              <X className="w-6 h-6 text-gray-600" />
            ) : (
              <Menu className="w-6 h-6 text-gray-600" />
            )}
          </button>
        </div>

        {/* Mobile Nav */}
        {isOpen && (
          <div className="md:hidden py-4 border-t border-gray-100">
            <div className="flex flex-col gap-4">
              <Link
                href="/"
                className="text-gray-600 hover:text-gray-900 transition-colors"
                onClick={() => setIsOpen(false)}
              >
                首页
              </Link>
              <Link
                href="/upload"
                className="text-gray-600 hover:text-gray-900 transition-colors"
                onClick={() => setIsOpen(false)}
              >
                转换
              </Link>
              <Link
                href="/history"
                className="text-gray-600 hover:text-gray-900 transition-colors"
                onClick={() => setIsOpen(false)}
              >
                历史
              </Link>
            </div>
          </div>
        )}
      </div>
    </nav>
  )
}
