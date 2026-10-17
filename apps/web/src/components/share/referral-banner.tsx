"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { Gift, Copy, Check, Users, Sparkles } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useReferral } from "@/hooks/use-referral"

export function ReferralBanner() {
  const { referralCode, invitedCount, bonusCredits, generateReferralLink, isLoaded } =
    useReferral()
  const [copied, setCopied] = useState(false)

  const handleCopyLink = async () => {
    const link = generateReferralLink()
    try {
      await navigator.clipboard.writeText(link)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // Fallback handled silently
    }
  }

  if (!isLoaded) {
    return null
  }

  return (
    <section className="w-full py-8 bg-gradient-to-r from-yellow-50 to-orange-50 border-y border-yellow-100">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="flex flex-col md:flex-row items-center gap-6"
        >
          {/* Left: Icon and Title */}
          <div className="flex items-center gap-4 flex-shrink-0">
            <div className="p-3 bg-yellow-500 rounded-xl shadow-lg shadow-yellow-200">
              <Gift className="w-6 h-6 text-white" />
            </div>
            <div>
              <h3 className="text-lg font-bold text-gray-900">Invite Friends, Get Free Credits</h3>
              <p className="text-sm text-gray-600">
                Each invite gives you <span className="font-semibold text-yellow-600">+5 conversions</span>
              </p>
            </div>
          </div>

          {/* Center: Stats */}
          <div className="flex items-center gap-6 px-6 py-2 bg-white/60 rounded-xl">
            <div className="flex items-center gap-2">
              <Users className="w-4 h-4 text-gray-500" />
              <span className="text-sm text-gray-600">
                <span className="font-bold text-gray-900">{invitedCount}</span> invited
              </span>
            </div>
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-yellow-500" />
              <span className="text-sm text-gray-600">
                <span className="font-bold text-yellow-600">{bonusCredits}</span> credits
              </span>
            </div>
          </div>

          {/* Right: Referral Code & Copy */}
          <div className="flex items-center gap-3 flex-shrink-0">
            <div className="flex flex-col">
              <span className="text-xs text-gray-500 uppercase tracking-wide">Your Code</span>
              <code className="px-3 py-1 bg-gray-900 text-white rounded-lg text-sm font-mono">
                {referralCode}
              </code>
            </div>
            <Button
              onClick={handleCopyLink}
              className={`${
                copied
                  ? "bg-green-600 hover:bg-green-700"
                  : "bg-yellow-500 hover:bg-yellow-600"
              } text-white transition-colors`}
            >
              {copied ? (
                <>
                  <Check className="w-4 h-4 mr-1" />
                  Copied!
                </>
              ) : (
                <>
                  <Copy className="w-4 h-4 mr-1" />
                  Copy Link
                </>
              )}
            </Button>
          </div>
        </motion.div>
      </div>
    </section>
  )
}
