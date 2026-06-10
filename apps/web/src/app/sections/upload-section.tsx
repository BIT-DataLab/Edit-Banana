"use client"

import { useState } from "react"
import { Loader2, Download, CheckCircle, AlertCircle, Share2 } from "lucide-react"
import { track } from "@vercel/analytics"
import { Button } from "@/components/ui/button"
import { FileUpload } from "@/components/upload/file-upload"
import { ShareModal } from "@/components/share/share-modal"
import { uploadFile, getJobStatus, downloadResult, APIError } from "@/lib/api"
import { useConversionHistoryContext } from "@/components/history/conversion-history-provider"
import type { Job } from "@/lib/types"

export function UploadSection() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const [jobStatus, setJobStatus] = useState<Job["status"] | null>(null)
  const [progress, setProgress] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [resultUrl, setResultUrl] = useState<string | null>(null)
  const [showShareModal, setShowShareModal] = useState(false)
  const { addHistoryItem, updateHistoryItem } = useConversionHistoryContext()

  const handleFileSelect = (file: File) => {
    setSelectedFile(file)
    setError(null)
    setJobId(null)
    setJobStatus(null)
    setResultUrl(null)
    track("file_selected", { filename: file.name, size: file.size })
  }

  const handleConvert = async () => {
    if (!selectedFile) return

    setLoading(true)
    setError(null)

    try {
      // Upload file
      track("conversion_started", { filename: selectedFile.name })
      const response = await uploadFile(selectedFile, true, false)
      setJobId(response.job_id)
      setJobStatus("pending")

      // Create initial history item
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "https://editbanana.anxin6.cn"
      addHistoryItem({
        jobId: response.job_id,
        filename: selectedFile.name,
        originalUrl: `${apiUrl}/api/v1/jobs/${response.job_id}/input`,
        resultUrl: null,
        status: "pending",
        fileSize: selectedFile.size,
      })

      // Poll for status
      pollJobStatus(response.job_id, selectedFile.name, selectedFile.size)
    } catch (err) {
      setLoading(false)
      if (err instanceof APIError) {
        setError(err.message)
      } else {
        setError("Conversion failed. Please try again.")
      }
    }
  }

  const pollJobStatus = async (id: string, filename: string, fileSize: number) => {
    const interval = setInterval(async () => {
      try {
        const job = await getJobStatus(id)
        setJobStatus(job.status)

        // Calculate progress
        if (job.total_steps && job.total_steps > 0 && job.current_step !== undefined) {
          setProgress(Math.round((job.current_step / job.total_steps) * 100))
        }

        if (job.status === "completed") {
          clearInterval(interval)
          setLoading(false)
          setProgress(100)
          const apiUrl = process.env.NEXT_PUBLIC_API_URL || "https://editbanana.anxin6.cn"
          const newResultUrl = `${apiUrl}/api/v1/jobs/${id}/result`
          setResultUrl(newResultUrl)
          track("conversion_completed", { job_id: id })

          // Update history item with result URL
          updateHistoryItem(id, {
            status: "completed",
            resultUrl: newResultUrl,
          })
        } else if (job.status === "failed" || job.status === "cancelled") {
          clearInterval(interval)
          setLoading(false)
          setError(job.error || "Conversion failed")

          // Update history item with error status
          updateHistoryItem(id, {
            status: job.status as "failed" | "cancelled",
            error: job.error || "Conversion failed",
          })
        }
      } catch (err) {
        clearInterval(interval)
        setLoading(false)
        setError("Failed to get job status")

        // Update history item with error
        updateHistoryItem(id, {
          status: "failed",
          error: "Failed to get job status",
        })
      }
    }, 2000)
  }

  const handleDownload = async () => {
    if (!jobId) return

    try {
      track("download_clicked", { job_id: jobId, filename: selectedFile?.name })
      const blob = await downloadResult(jobId)
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `converted-${selectedFile?.name || "diagram.drawio"}`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (err) {
      setError("Download failed")
    }
  }

  return (
    <section id="upload" className="w-full py-20 bg-white">
      <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            Upload Your Image or PDF
          </h2>
          <p className="text-gray-600">
            We&apos;ll convert it to an editable diagram in seconds
          </p>
        </div>

        <div className="space-y-6">
          <FileUpload onFileSelect={handleFileSelect} />

          {selectedFile && !jobStatus && (
            <Button
              onClick={handleConvert}
              disabled={loading}
              size="lg"
              className="w-full bg-gradient-to-r from-yellow-500 to-yellow-600 hover:from-yellow-600 hover:to-yellow-700 text-white font-semibold py-6 rounded-xl"
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                  Uploading...
                </>
              ) : (
                "Convert to Editable Diagram"
              )}
            </Button>
          )}

          {loading && jobStatus && jobStatus !== "completed" && (
            <div className="p-6 bg-gray-50 rounded-xl border border-gray-200">
              <div className="flex items-center gap-3 mb-4">
                <Loader2 className="h-5 w-5 animate-spin text-yellow-600" />
                <span className="font-medium text-gray-900">
                  {jobStatus === "pending" && "Waiting to process..."}
                  {jobStatus === "processing" && "Converting your diagram..."}
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-gradient-to-r from-yellow-500 to-yellow-600 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="mt-2 text-sm text-gray-500">{progress}% complete</p>
            </div>
          )}

          {jobStatus === "completed" && (
            <div className="p-6 bg-green-50 rounded-xl border border-green-200">
              <div className="flex items-center gap-3 mb-4">
                <CheckCircle className="h-6 w-6 text-green-600" />
                <span className="font-semibold text-green-900">
                  Conversion Complete!
                </span>
              </div>
              <p className="text-sm text-green-700 mb-4">
                Your diagram has been converted and is ready to download.
              </p>
              <div className="flex flex-col gap-3">
                <Button
                  onClick={handleDownload}
                  className="w-full bg-green-600 hover:bg-green-700 text-white"
                >
                  <Download className="mr-2 h-5 w-5" />
                  Download Result
                </Button>
                <Button
                  onClick={() => setShowShareModal(true)}
                  variant="outline"
                  className="w-full border-green-300 text-green-700 hover:bg-green-100"
                >
                  <Share2 className="mr-2 h-5 w-5" />
                  Share Result
                </Button>
              </div>
            </div>
          )}

          {error && (
            <div className="p-4 bg-red-50 rounded-xl border border-red-200 flex items-center gap-3">
              <AlertCircle className="h-5 w-5 text-red-600 flex-shrink-0" />
              <p className="text-sm text-red-700">{error}</p>
            </div>
          )}
        </div>

        <ShareModal
          isOpen={showShareModal}
          onClose={() => setShowShareModal(false)}
          options={{
            title: "I just converted an image to an editable diagram!",
            description: `Check out how I converted ${selectedFile?.name || "my image"} to an editable diagram using EditBanana.`,
            url: typeof window !== "undefined" ? window.location.href : "https://editbanana.anxin6.cn",
          }}
        />
      </div>
    </section>
  )
}
