export interface ShareOptions {
  title: string
  description: string
  url: string
  imageUrl?: string
}

export type ShareChannel = "twitter" | "linkedin" | "facebook" | "wechat" | "copy" | "email"

export interface ReferralInfo {
  referralCode: string
  invitedCount: number
  bonusCredits: number
}

export interface ShareResult {
  success: boolean
  channel: ShareChannel
  error?: string
}
