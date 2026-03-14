"use client"

import { useCallback, useState, useEffect, useRef } from "react"
import { ReferralInfo } from "@/types/share"

const REFERRAL_STORAGE_KEY = "editbanana_referral"
const INVITED_COUNT_KEY = "editbanana_invited_count"

function generateReferralCode(): string {
  // Generate a 8-character alphanumeric code
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
  let code = ""
  for (let i = 0; i < 8; i++) {
    code += chars.charAt(Math.floor(Math.random() * chars.length))
  }
  return code
}

function loadReferralInfo(): ReferralInfo {
  try {
    const storedCode = localStorage.getItem(REFERRAL_STORAGE_KEY)
    const storedCount = localStorage.getItem(INVITED_COUNT_KEY)

    const code = storedCode || generateReferralCode()
    if (!storedCode) {
      localStorage.setItem(REFERRAL_STORAGE_KEY, code)
    }

    const count = storedCount ? parseInt(storedCount, 10) : 0
    const bonus = count * 5 // 5 credits per invite

    return {
      referralCode: code,
      invitedCount: count,
      bonusCredits: bonus,
    }
  } catch {
    // localStorage not available (private browsing)
    return {
      referralCode: generateReferralCode(),
      invitedCount: 0,
      bonusCredits: 0,
    }
  }
}

export function useReferral() {
  const [referralInfo, setReferralInfo] = useState<ReferralInfo>(loadReferralInfo)
  const hasCheckedReferral = useRef(false)

  const getReferralCode = useCallback((): string => {
    return referralInfo.referralCode
  }, [referralInfo.referralCode])

  const generateReferralLink = useCallback((): string => {
    const baseUrl = window.location.origin
    return `${baseUrl}?ref=${referralInfo.referralCode}`
  }, [referralInfo.referralCode])

  const trackReferral = useCallback((code: string): void => {
    // Simulate tracking a referral
    // In production, this would call an API endpoint
    try {
      const currentCount = parseInt(localStorage.getItem(INVITED_COUNT_KEY) || "0", 10)
      const newCount = currentCount + 1
      localStorage.setItem(INVITED_COUNT_KEY, String(newCount))

      setReferralInfo((prev) => ({
        ...prev,
        invitedCount: newCount,
        bonusCredits: newCount * 5,
      }))
    } catch {
      // localStorage not available
    }
  }, [])

  // Check for referral on mount - using setTimeout to avoid synchronous setState in effect
  useEffect(() => {
    if (hasCheckedReferral.current) return
    hasCheckedReferral.current = true

    const urlParams = new URLSearchParams(window.location.search)
    const refCode = urlParams.get("ref")
    if (refCode && refCode !== referralInfo.referralCode) {
      // Defer state update to avoid synchronous setState in effect body
      setTimeout(() => {
        trackReferral(refCode)
      }, 0)
      // Remove ref param from URL without reloading
      const newUrl = window.location.pathname
      window.history.replaceState({}, document.title, newUrl)
    }
  }, [referralInfo.referralCode, trackReferral])

  return {
    referralCode: referralInfo.referralCode,
    invitedCount: referralInfo.invitedCount,
    bonusCredits: referralInfo.bonusCredits,
    getReferralCode,
    generateReferralLink,
    trackReferral,
    isLoaded: true,
  }
}
