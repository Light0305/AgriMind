import { useCallback, useState, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

interface ImageUploadProps {
  onUpload: (file: File) => void | Promise<void>
  images?: string[]
  disabled?: boolean
}

export default function ImageUpload({
  onUpload,
  images = [],
  disabled = false,
}: ImageUploadProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [enlargedIdx, setEnlargedIdx] = useState<number | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = useCallback(
    (file: File) => {
      if (disabled) return
      if (!file.type.match(/^image\/(jpeg|png)$/)) {
        alert('仅支持 JPEG / PNG 格式图片')
        return
      }
      onUpload(file)
    },
    [onUpload, disabled],
  )

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragging(false)
      const file = e.dataTransfer.files[0]
      if (file) handleFile(file)
    },
    [handleFile],
  )

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const onDragLeave = useCallback(() => {
    setIsDragging(false)
  }, [])

  const onClick = useCallback(() => {
    if (!disabled) inputRef.current?.click()
  }, [disabled])

  const onFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) handleFile(file)
      e.target.value = ''
    },
    [handleFile],
  )

  return (
    <div className="flex flex-col gap-3">
      {/* Drop zone */}
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
          rounded-xl border-2 border-dashed p-6
          cursor-pointer transition-all duration-200
          ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
          ${
            isDragging
              ? 'border-emerald-500 bg-emerald-50 scale-[1.01]'
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

        <div className="text-3xl mb-2">{'\u{1F4F7}'}</div>
        <p className="text-sm font-medium text-gray-700">
          拖放作物病害图片到此处
        </p>
        <p className="text-xs text-gray-500 mt-1">
          或点击选择文件 · 支持 JPEG / PNG
        </p>
      </div>

      {/* Thumbnails grid */}
      {images.length > 0 && (
        <div>
          <p className="text-xs text-gray-500 mb-2">
            已上传 {images.length} 张图片
          </p>
          <div className="grid grid-cols-3 gap-2">
            {images.map((src, i) => (
              <button
                key={i}
                type="button"
                onClick={() => setEnlargedIdx(i)}
                className="relative rounded-lg overflow-hidden border border-gray-200 shadow-sm hover:shadow-md transition-shadow group aspect-square"
              >
                <img
                  src={src}
                  alt={`上传 ${i + 1}`}
                  className="w-full h-full object-cover"
                />
                <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors flex items-center justify-center">
                  <span className="text-white text-lg opacity-0 group-hover:opacity-100 transition-opacity drop-shadow-lg">
                    {'\u{1F50D}'}
                  </span>
                </div>
                <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/40 to-transparent p-1">
                  <span className="text-[10px] text-white font-medium">
                    #{i + 1}
                  </span>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Lightbox */}
      <AnimatePresence>
        {enlargedIdx !== null && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 backdrop-blur-sm"
            onClick={() => setEnlargedIdx(null)}
          >
            <motion.img
              initial={{ scale: 0.85, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.85, opacity: 0 }}
              transition={{ type: 'spring', damping: 25, stiffness: 300 }}
              src={images[enlargedIdx]}
              alt="放大预览"
              className="max-w-[90vw] max-h-[85vh] rounded-xl shadow-2xl object-contain"
              onClick={(e) => e.stopPropagation()}
            />
            <button
              type="button"
              className="absolute top-6 right-6 w-10 h-10 rounded-full bg-white/20 hover:bg-white/40 text-white text-xl flex items-center justify-center transition-colors"
              onClick={() => setEnlargedIdx(null)}
            >
              {'\u{2715}'}
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
