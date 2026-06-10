"use client"

import { useCallback, useState } from "react"
import { useDropzone } from "react-dropzone"
import { Upload, File, X, Image as ImageIcon } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"

interface FileUploadProps {
  onFileSelect: (file: File) => void
  accept?: Record<string, string[]>
  maxSize?: number
}

export function FileUpload({
  onFileSelect,
  accept = {
    "image/*": [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"],
    "application/pdf": [".pdf"],
  },
  maxSize = 10 * 1024 * 1024, // 10MB
}: FileUploadProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        const file = acceptedFiles[0]
        setSelectedFile(file)
        onFileSelect(file)

        // 创建预览
        if (file.type.startsWith("image/")) {
          const reader = new FileReader()
          reader.onload = (e) => {
            setPreview(e.target?.result as string)
          }
          reader.readAsDataURL(file)
        } else {
          setPreview(null)
        }
      }
    },
    [onFileSelect]
  )

  const { getRootProps, getInputProps, isDragActive, fileRejections } =
    useDropzone({
      onDrop,
      accept,
      maxSize,
      multiple: false,
    })

  const removeFile = () => {
    setSelectedFile(null)
    setPreview(null)
  }

  return (
    <div className="w-full">
      {!selectedFile ? (
        <div
          {...getRootProps()}
          className={cn(
            "relative flex flex-col items-center justify-center w-full h-64 border-2 border-dashed rounded-xl transition-colors cursor-pointer",
            "hover:border-yellow-500 hover:bg-yellow-50/50",
            isDragActive
              ? "border-yellow-500 bg-yellow-50"
              : "border-gray-300 bg-gray-50"
          )}
        >
          <input {...getInputProps()} />
          <div className="flex flex-col items-center gap-4">
            <div className="p-4 bg-white rounded-full shadow-sm">
              <Upload className="w-8 h-8 text-yellow-500" />
            </div>
            <div className="text-center">
              <p className="text-lg font-medium text-gray-900">
                {isDragActive ? "释放文件以上传" : "拖拽文件到此处"}
              </p>
              <p className="mt-1 text-sm text-gray-500">
                或点击选择文件，支持 PNG, JPG, PDF 等格式
              </p>
              <p className="mt-1 text-xs text-gray-400">
                最大支持 10MB
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="relative p-6 border border-gray-200 rounded-xl bg-gray-50">
          <button
            onClick={removeFile}
            className="absolute top-2 right-2 p-1 rounded-full bg-gray-200 hover:bg-gray-300 transition-colors"
          >
            <X className="w-4 h-4 text-gray-600" />
          </button>

          <div className="flex items-center gap-4">
            {preview ? (
              <div className="relative w-24 h-24 rounded-lg overflow-hidden border border-gray-200">
                <img
                  src={preview}
                  alt="Preview"
                  className="w-full h-full object-cover"
                />
              </div>
            ) : (
              <div className="flex items-center justify-center w-24 h-24 bg-gray-100 rounded-lg border border-gray-200">
                <File className="w-10 h-10 text-gray-400" />
              </div>
            )}
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-900 truncate">
                {selectedFile.name}
              </p>
              <p className="text-sm text-gray-500">
                {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>
          </div>
        </div>
      )}

      {fileRejections.length > 0 && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-sm text-red-600">
            文件格式不支持或超过大小限制
          </p>
        </div>
      )}
    </div>
  )
}
