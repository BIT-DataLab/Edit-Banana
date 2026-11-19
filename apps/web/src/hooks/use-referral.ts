"use client"

import { useCallback, useState, useEffect } from "react"
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

export function useReferral() {
  const [referralInfo, setReferralInfo] = useState<ReferralInfo>({
    referralCode: "",
    invitedCount: 0,
    bonusCredits: 0,
  })
  const [isLoaded, setIsLoaded] = useState(false)

  // Load referral info from localStorage on mount
  useEffect(() => {
    try {
      const storedCode = localStorage.getItem(REFERRAL_STORAGE_KEY)
      const storedCount = localStorage.getItem(INVITED_COUNT_KEY)

      const code = storedCode || generateReferralCode()
      if (!storedCode) {
        localStorage.setItem(REFERRAL_STORAGE_KEY, code)
      }

      const count = storedCount ? parseInt(storedCount, 10) : 0
      const bonus = count * 5 // 5 credits per invite

      setReferralInfo({
        referralCode: code,
        invitedCount: count,
        bonusCredits: bonus,
      })
    } catch {
      // localStorage not available (private browsing)
      setReferralInfo({
        referralCode: generateReferralCode(),
        invitedCount: 0,
        bonusCredits: 0,
      })
    }
    setIsLoaded(true)
  }, [])

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

  const checkForReferral = useCallback((): void => {
    // Check if user came from a referral link
    const urlParams = new URLSearchParams(window.location.search)
    const refCode = urlParams.get("ref")
    if (refCode && refCode !== referralInfo.referralCode) {
      // Track this referral
      trackReferral(refCode)
      // Remove ref param from URL without reloading
      const newUrl = window.location.pathname
      window.history.replaceState({}, document.title, newUrl)
    }
  }, [referralInfo.referralCode, trackReferral])

  // Check for referral on mount
  useEffect(() => {
    if (isLoaded) {
      checkForReferral()
    }
  }, [isLoaded, checkForReferral])

  return {
    referralCode: referralInfo.referralCode,
    invitedCount: referralInfo.invitedCount,
    bonusCredits: referralInfo.bonusCredits,
    getReferralCode,
    generateReferralLink,
    trackReferral,
    checkForReferral,
    isLoaded,
  }
}
