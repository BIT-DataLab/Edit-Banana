"use client"

import { motion } from "framer-motion"
import { cn } from "@/lib/utils"
import { Check, Loader2 } from "lucide-react"

interface ProgressBarProps {
  progress: number
  stage: string
  message: string
  isComplete?: boolean
}

const stages = [
  { key: "preprocess", label: "预处理", icon: "1" },
  { key: "text_extraction", label: "文字提取", icon: "2" },
  { key: "segmentation", label: "图像分割", icon: "3" },
  { key: "processing", label: "元素处理", icon: "4" },
  { key: "xml_generation", label: "生成XML", icon: "5" },
]

export function ProgressBar({
  progress,
  stage,
  message,
  isComplete,
}: ProgressBarProps) {
  const getStageStatus = (stageLabel: string) => {
    const stageIndex = stages.findIndex((s) => s.label === stage)
    const currentIndex = stages.findIndex((s) => s.label === stageLabel)

    if (isComplete) return "complete"
    if (currentIndex < stageIndex) return "complete"
    if (currentIndex === stageIndex) return "active"
    return "pending"
  }

  return (
    <div className="w-full max-w-2xl mx-auto">
      {/* 进度条 */}
      <div className="relative h-3 bg-gray-200 rounded-full overflow-hidden">
        <motion.div
          className="absolute top-0 left-0 h-full bg-gradient-to-r from-yellow-400 to-yellow-500 rounded-full"
          initial={{ width: 0 }}
          animate={{ width: `${progress}%` }}
          transition={{ duration: 0.5, ease: "easeOut" }}
        />
      </div>

      {/* 进度信息 */}
      <div className="mt-4 text-center">
        <motion.p
          key={stage}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-lg font-medium text-gray-900"
        >
          {stage}
        </motion.p>
        <motion.p
          key={message}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mt-1 text-sm text-gray-500"
        >
          {message}
        </motion.p>
      </div>

      {/* 阶段指示器 */}
      <div className="mt-8">
        <div className="flex justify-between">
          {stages.map((s, index) => {
            const status = getStageStatus(s.label)
            return (
              <div key={s.key} className="flex flex-col items-center">
                <motion.div
                  className={cn(
                    "w-10 h-10 rounded-full flex items-center justify-center text-sm font-medium transition-colors",
                    status === "complete" && "bg-green-500 text-white",
                    status === "active" && "bg-yellow-500 text-white ring-4 ring-yellow-200",
                    status === "pending" && "bg-gray-200 text-gray-500"
                  )}
                  animate={
                    status === "active"
                      ? { scale: [1, 1.1, 1] }
                      : {}
                  }
                  transition={{ repeat: Infinity, duration: 1.5 }}
                >
                  {status === "complete" ? (
                    <Check className="w-5 h-5" />
                  ) : status === "active" ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    s.icon
                  )}
                </motion.div>
                <span
                  className={cn(
                    "mt-2 text-xs font-medium",
                    status === "complete" && "text-green-600",
                    status === "active" && "text-yellow-600",
                    status === "pending" && "text-gray-400"
                  )}
                >
                  {s.label}
                </span>
              </div>
            )
          })}
        </div>
      </div>

      {/* 百分比 */}
      <div className="mt-6 text-center">
        <span className="text-3xl font-bold text-gray-900">{progress}%</span>
      </div>
    </div>
  )
}
