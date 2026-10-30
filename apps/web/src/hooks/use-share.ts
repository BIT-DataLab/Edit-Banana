"use client"

import { useCallback } from "react"
import { ShareOptions, ShareChannel, ShareResult } from "@/types/share"

export function useShare() {
  const shareToTwitter = useCallback((options: ShareOptions): void => {
    const text = encodeURIComponent(`${options.title}\n\n${options.description}`)
    const url = encodeURIComponent(options.url)
    window.open(
      `https://twitter.com/intent/tweet?text=${text}&url=${url}`,
      "_blank",
      "width=550,height=420"
    )
  }, [])

  const shareToLinkedIn = useCallback((options: ShareOptions): void => {
    const url = encodeURIComponent(options.url)
    window.open(
      `https://www.linkedin.com/sharing/share-offsite/?url=${url}`,
      "_blank",
      "width=550,height=420"
    )
  }, [])

  const shareToFacebook = useCallback((options: ShareOptions): void => {
    const url = encodeURIComponent(options.url)
    window.open(
      `https://www.facebook.com/sharer/sharer.php?u=${url}`,
      "_blank",
      "width=550,height=420"
    )
  }, [])

  const shareToWeChat = useCallback((): void => {
    // WeChat requires QR code scanning on mobile
    // For web, we'll show a QR code or copy link
    alert('Please use WeChat\'s "Scan" feature to share this page')
  }, [])

  const copyLink = useCallback(async (url: string): Promise<boolean> => {
    try {
      await navigator.clipboard.writeText(url)
      return true
    } catch {
      // Fallback for older browsers
      const textArea = document.createElement("textarea")
      textArea.value = url
      textArea.style.position = "fixed"
      textArea.style.left = "-999999px"
      document.body.appendChild(textArea)
      textArea.focus()
      textArea.select()
      try {
        document.execCommand("copy")
        return true
      } catch {
        return false
      } finally {
        textArea.remove()
      }
    }
  }, [])

  const shareViaEmail = useCallback((options: ShareOptions): void => {
    const subject = encodeURIComponent(options.title)
    const body = encodeURIComponent(`${options.description}\n\n${options.url}`)
    window.location.href = `mailto:?subject=${subject}&body=${body}`
  }, [])

  const share = useCallback(
    async (channel: ShareChannel, options: ShareOptions): Promise<ShareResult> => {
      try {
        switch (channel) {
          case "twitter":
            shareToTwitter(options)
            return { success: true, channel }
          case "linkedin":
            shareToLinkedIn(options)
            return { success: true, channel }
          case "facebook":
            shareToFacebook(options)
            return { success: true, channel }
          case "wechat":
            shareToWeChat()
            return { success: true, channel }
          case "copy":
            const copied = await copyLink(options.url)
            return { success: copied, channel, error: copied ? undefined : "Failed to copy link" }
          case "email":
            shareViaEmail(options)
            return { success: true, channel }
          default:
            return { success: false, channel, error: "Unknown channel" }
        }
      } catch (error) {
        return { success: false, channel, error: String(error) }
      }
    },
    [shareToTwitter, shareToLinkedIn, shareToFacebook, shareToWeChat, copyLink, shareViaEmail]
  )

  return {
    share,
    shareToTwitter,
    shareToLinkedIn,
    shareToFacebook,
    shareToWeChat,
    copyLink,
    shareViaEmail,
  }
}
