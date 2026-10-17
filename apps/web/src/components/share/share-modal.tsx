"use client"

import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import {
  X,
  Twitter,
  Linkedin,
  Facebook,
  MessageCircle,
  Link2,
  Mail,
  Check,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { useShare } from "@/hooks/use-share"
import { ShareOptions, ShareChannel } from "@/types/share"

interface ShareModalProps {
  isOpen: boolean
  onClose: () => void
  options: ShareOptions
}

const shareChannels: { id: ShareChannel; label: string; icon: typeof Twitter; color: string }[] =
  [
    {
      id: "twitter",
      label: "Twitter",
      icon: Twitter,
      color: "bg-sky-500 hover:bg-sky-600",
    },
    {
      id: "linkedin",
      label: "LinkedIn",
      icon: Linkedin,
      color: "bg-blue-700 hover:bg-blue-800",
    },
    {
      id: "facebook",
      label: "Facebook",
      icon: Facebook,
      color: "bg-blue-600 hover:bg-blue-700",
    },
    {
      id: "wechat",
      label: "WeChat",
      icon: MessageCircle,
      color: "bg-green-500 hover:bg-green-600",
    },
    {
      id: "copy",
      label: "Copy Link",
      icon: Link2,
      color: "bg-gray-600 hover:bg-gray-700",
    },
    {
      id: "email",
      label: "Email",
      icon: Mail,
      color: "bg-yellow-600 hover:bg-yellow-700",
    },
  ]

export function ShareModal({ isOpen, onClose, options }: ShareModalProps) {
  const { share, copyLink } = useShare()
  const [copied, setCopied] = useState(false)
  const [sharing, setSharing] = useState<ShareChannel | null>(null)

  const handleShare = async (channel: ShareChannel) => {
    setSharing(channel)

    if (channel === "copy") {
      const result = await share(channel, options)
      if (result.success) {
        setCopied(true)
        setTimeout(() => setCopied(false), 2000)
      }
    } else {
      await share(channel, options)
    }

    setSharing(null)
  }

  const handleClose = () => {
    setCopied(false)
    setSharing(null)
    onClose()
  }

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={handleClose}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50"
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            transition={{ duration: 0.2 }}
            className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-md bg-white rounded-2xl shadow-2xl z-50 p-6"
          >
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-bold text-gray-900">Share Your Conversion</h3>
              <button
                onClick={handleClose}
                className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-full transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Preview */}
            <div className="bg-gray-50 rounded-xl p-4 mb-6">
              <p className="font-medium text-gray-900 mb-1">{options.title}</p>
              <p className="text-sm text-gray-600 line-clamp-2">{options.description}</p>
            </div>

            {/* Share Grid */}
            <div className="grid grid-cols-3 gap-3">
              {shareChannels.map((channel) => {
                const Icon = channel.icon
                const isCopy = channel.id === "copy"
                const showCopied = isCopy && copied

                return (
                  <button
                    key={channel.id}
                    onClick={() => handleShare(channel.id)}
                    disabled={sharing !== null}
                    className="flex flex-col items-center gap-2 p-4 rounded-xl hover:bg-gray-50 transition-colors group"
                  >
                    <div
                      className={`w-12 h-12 ${channel.color} rounded-xl flex items-center justify-center text-white shadow-lg shadow-gray-200 group-hover:scale-110 transition-transform`}
                    >
                      {showCopied ? (
                        <Check className="w-5 h-5" />
                      ) : (
                        <Icon className="w-5 h-5" />
                      )}
                    </div>
                    <span className="text-xs font-medium text-gray-700">
                      {showCopied ? "Copied!" : channel.label}
                    </span>
                  </button>
                )
              })}
            </div>

            {/* Link Copy Section */}
            <div className="mt-6 pt-6 border-t border-gray-100">
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  value={options.url}
                  readOnly
                  className="flex-1 px-3 py-2 bg-gray-50 border border-gray-200 rounded-lg text-sm text-gray-600 focus:outline-none focus:ring-2 focus:ring-yellow-500"
                />
                <Button
                  onClick={() => handleShare("copy")}
                  variant="outline"
                  className="shrink-0"
                >
                  {copied ? (
                    <>
                      <Check className="w-4 h-4 mr-1" />
                      Copied
                    </>
                  ) : (
                    <>
                      <Link2 className="w-4 h-4 mr-1" />
                      Copy
                    </>
                  )}
                </Button>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}
