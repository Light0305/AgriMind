import { useState, useRef, useCallback, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import type { GroundingBox } from '../types'

/* ── props ───────────────────────────────────────────────── */

interface ImageAnnotationProps {
  imageUrl: string
  boxes: GroundingBox[]
  showLabels?: boolean
}

/* ── color palette for boxes ─────────────────────────────── */

const BOX_COLORS = [
  { stroke: '#10b981', fill: 'rgba(16,185,129,0.12)', text: '#065f46' },
  { stroke: '#f59e0b', fill: 'rgba(245,158,11,0.12)', text: '#92400e' },
  { stroke: '#ef4444', fill: 'rgba(239,68,68,0.12)', text: '#991b1b' },
  { stroke: '#8b5cf6', fill: 'rgba(139,92,246,0.12)', text: '#5b21b6' },
  { stroke: '#06b6d4', fill: 'rgba(6,182,212,0.12)', text: '#164e63' },
  { stroke: '#ec4899', fill: 'rgba(236,72,153,0.12)', text: '#9d174d' },
]

type ViewMode = 'original' | 'annotated' | 'side-by-side'

/* ── main component ──────────────────────────────────────── */

export default function ImageAnnotation({
  imageUrl,
  boxes,
  showLabels = true,
}: ImageAnnotationProps) {
  const [mode, setMode] = useState<ViewMode>(boxes.length > 0 ? 'annotated' : 'original')
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 })
  const [isEnlarged, setIsEnlarged] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)

  // Measure rendered image size
  const measureImage = useCallback(() => {
    if (imgRef.current) {
      setImgSize({
        w: imgRef.current.clientWidth,
        h: imgRef.current.clientHeight,
      })
    }
  }, [])

  useEffect(() => {
    measureImage()
    window.addEventListener('resize', measureImage)
    return () => window.removeEventListener('resize', measureImage)
  }, [measureImage])

  const modes: { key: ViewMode; label: string }[] = [
    { key: 'original', label: '原图' },
    { key: 'annotated', label: '标注' },
    { key: 'side-by-side', label: '对比' },
  ]

  const renderAnnotatedImage = (className?: string) => (
    <div className={`relative inline-block ${className ?? ''}`}>
      <img
        ref={imgRef}
        src={imageUrl}
        alt="诊断图片"
        className="w-full h-auto rounded-lg"
        onLoad={measureImage}
      />
      {mode !== 'original' && boxes.length > 0 && imgSize.w > 0 && (
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none"
          viewBox={`0 0 ${imgSize.w} ${imgSize.h}`}
        >
          {boxes.map((box, i) => {
            const color = BOX_COLORS[i % BOX_COLORS.length]
            const bx = box.x * imgSize.w
            const by = box.y * imgSize.h
            const bw = box.width * imgSize.w
            const bh = box.height * imgSize.h

            return (
              <g key={i}>
                <rect
                  x={bx}
                  y={by}
                  width={bw}
                  height={bh}
                  fill={color.fill}
                  stroke={color.stroke}
                  strokeWidth={2.5}
                  rx={4}
                />
                {showLabels && box.label && (
                  <>
                    <rect
                      x={bx}
                      y={by - 20}
                      width={Math.max(box.label.length * 12 + 12, 40)}
                      height={20}
                      fill={color.stroke}
                      rx={3}
                    />
                    <text
                      x={bx + 6}
                      y={by - 5}
                      fill="white"
                      fontSize={12}
                      fontWeight="bold"
                      fontFamily="sans-serif"
                    >
                      {box.label}
                    </text>
                  </>
                )}
              </g>
            )
          })}
        </svg>
      )}
    </div>
  )

  return (
    <div ref={containerRef} className="flex flex-col gap-3">
      {/* Mode toggle */}
      {boxes.length > 0 && (
        <div className="flex items-center gap-1 p-1 bg-gray-100 rounded-lg w-fit">
          {modes.map((m) => (
            <button
              key={m.key}
              type="button"
              onClick={() => setMode(m.key)}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
                mode === m.key
                  ? 'bg-white text-emerald-700 shadow-sm'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
      )}

      {/* Image display */}
      <button
        type="button"
        onClick={() => setIsEnlarged(true)}
        className="relative rounded-xl overflow-hidden border border-gray-200 shadow-sm hover:shadow-md transition-shadow cursor-zoom-in"
      >
        {mode === 'side-by-side' ? (
          <div className="grid grid-cols-2 gap-px bg-gray-200">
            <div className="bg-white p-1">
              <img
                src={imageUrl}
                alt="原图"
                className="w-full h-auto rounded"
              />
              <p className="text-[10px] text-gray-400 text-center mt-1">原图</p>
            </div>
            <div className="bg-white p-1">
              {renderAnnotatedImage()}
              <p className="text-[10px] text-gray-400 text-center mt-1">
                标注
              </p>
            </div>
          </div>
        ) : (
          renderAnnotatedImage()
        )}
      </button>

      {/* Box legend */}
      {mode !== 'original' && boxes.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {boxes.map((box, i) => {
            const color = BOX_COLORS[i % BOX_COLORS.length]
            return (
              <span
                key={i}
                className="inline-flex items-center gap-1 text-[10px] font-medium px-2 py-1 rounded-full border"
                style={{
                  borderColor: color.stroke,
                  color: color.text,
                  backgroundColor: color.fill,
                }}
              >
                <span
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: color.stroke }}
                />
                {box.label}
              </span>
            )
          })}
        </div>
      )}

      {/* Enlarged lightbox */}
      <AnimatePresence>
        {isEnlarged && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 backdrop-blur-sm"
            onClick={() => setIsEnlarged(false)}
          >
            <motion.div
              initial={{ scale: 0.85, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.85, opacity: 0 }}
              transition={{ type: 'spring', damping: 25, stiffness: 300 }}
              className="max-w-[90vw] max-h-[85vh] overflow-auto"
              onClick={(e) => e.stopPropagation()}
            >
              {renderAnnotatedImage('max-h-[85vh] [&_img]:max-h-[85vh] [&_img]:w-auto')}
            </motion.div>
            <button
              type="button"
              className="absolute top-6 right-6 w-10 h-10 rounded-full bg-white/20 hover:bg-white/40 text-white text-xl flex items-center justify-center transition-colors"
              onClick={() => setIsEnlarged(false)}
            >
              {'\u{2715}'}
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
