import { useCallback, useState, useRef } from 'react'

interface ImageUploadProps {
  onUpload: (file: File) => void
}

export default function ImageUpload({ onUpload }: ImageUploadProps) {
  const [preview, setPreview] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = useCallback(
    (file: File) => {
      if (!file.type.match(/^image\/(jpeg|png)$/)) {
        alert('仅支持 JPEG / PNG 格式图片')
        return
      }
      setPreview(URL.createObjectURL(file))
      onUpload(file)
    },
    [onUpload]
  )

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragging(false)
      const file = e.dataTransfer.files[0]
      if (file) handleFile(file)
    },
    [handleFile]
  )

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const onDragLeave = useCallback(() => {
    setIsDragging(false)
  }, [])

  const onClick = useCallback(() => {
    inputRef.current?.click()
  }, [])

  const onFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) handleFile(file)
    },
    [handleFile]
  )

  return (
    <div className="flex flex-col gap-3">
      <div
        role="button"
        tabIndex={0}
        onClick={onClick}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onKeyDown={(e) => e.key === 'Enter' && onClick()}
        className={`
          relative flex flex-col items-center justify-center
          rounded-xl border-2 border-dashed p-8
          cursor-pointer transition-all duration-200
          ${
            isDragging
              ? 'border-emerald-500 bg-emerald-50'
              : 'border-gray-300 bg-gray-50 hover:border-emerald-400 hover:bg-emerald-50/50'
          }
        `}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png"
          onChange={onFileChange}
          className="hidden"
        />

        <div className="text-4xl mb-3">📷</div>
        <p className="text-sm font-medium text-gray-700">
          拖放作物病害图片到此处
        </p>
        <p className="text-xs text-gray-500 mt-1">
          或点击选择文件 · 支持 JPEG / PNG
        </p>
      </div>

      {preview && (
        <div className="relative rounded-lg overflow-hidden border border-gray-200 shadow-sm">
          <img
            src={preview}
            alt="上传预览"
            className="w-full h-48 object-cover"
          />
          <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/50 to-transparent p-2">
            <span className="text-xs text-white">已上传</span>
          </div>
        </div>
      )}
    </div>
  )
}
