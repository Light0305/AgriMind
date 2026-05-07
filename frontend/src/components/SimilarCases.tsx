import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import type { SimilarCase } from '../types'

/* ── props ───────────────────────────────────────────────── */

interface SimilarCasesProps {
  cases: SimilarCase[]
  currentImage?: string
}

/* ── similarity bar ──────────────────────────────────────── */

function SimilarityBar({ value }: { value: number }) {
  const pct = Math.round(value * 100)
  const color =
    pct >= 80 ? 'bg-emerald-500' : pct >= 60 ? 'bg-yellow-500' : 'bg-orange-500'

  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-gray-200 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
          className={`h-full rounded-full ${color}`}
        />
      </div>
      <span className="text-xs font-mono font-bold text-gray-600 w-10 text-right">
        {pct}%
      </span>
    </div>
  )
}

/* ── main component ──────────────────────────────────────── */

export default function SimilarCases({
  cases,
  currentImage,
}: SimilarCasesProps) {
  const [enlargedUrl, setEnlargedUrl] = useState<string | null>(null)

  if (cases.length === 0) return null

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <span className="text-base">{'\u{1F50E}'}</span>
        <h3 className="text-sm font-bold text-gray-700">相似病例参考</h3>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
        {/* Current image (if provided) */}
        {currentImage && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-xl border-2 border-emerald-300 bg-emerald-50/50 p-2 shadow-sm"
          >
            <button
              type="button"
              onClick={() => setEnlargedUrl(currentImage)}
              className="w-full aspect-square rounded-lg overflow-hidden mb-2 cursor-zoom-in"
            >
              <img
                src={currentImage}
                alt="当前图片"
                className="w-full h-full object-cover"
              />
            </button>
            <p className="text-xs font-semibold text-emerald-700 text-center">
              当前送诊图片
            </p>
          </motion.div>
        )}

        {/* Similar cases */}
        {cases.map((c, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.08 }}
            className="rounded-xl border border-gray-200 bg-white p-2 shadow-sm hover:shadow-md transition-shadow"
          >
            <button
              type="button"
              onClick={() => setEnlargedUrl(c.image_url)}
              className="w-full aspect-square rounded-lg overflow-hidden mb-2 cursor-zoom-in bg-gray-100"
            >
              <img
                src={c.image_url}
                alt={c.label}
                className="w-full h-full object-cover"
                onError={(e) => {
                  ;(e.target as HTMLImageElement).src = ''
                  ;(e.target as HTMLImageElement).alt = '图片加载失败'
                }}
              />
            </button>
            <div className="space-y-1.5 px-1">
              <p className="text-xs font-semibold text-gray-800 truncate">
                {c.label}
              </p>
              <SimilarityBar value={c.similarity} />
              <p className="text-[10px] text-gray-400 truncate">
                来源: {c.source}
              </p>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Lightbox */}
      <AnimatePresence>
        {enlargedUrl && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 backdrop-blur-sm"
            onClick={() => setEnlargedUrl(null)}
          >
            <motion.img
              initial={{ scale: 0.85, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.85, opacity: 0 }}
              transition={{ type: 'spring', damping: 25, stiffness: 300 }}
              src={enlargedUrl}
              alt="放大预览"
              className="max-w-[90vw] max-h-[85vh] rounded-xl shadow-2xl object-contain"
              onClick={(e) => e.stopPropagation()}
            />
            <button
              type="button"
              className="absolute top-6 right-6 w-10 h-10 rounded-full bg-white/20 hover:bg-white/40 text-white text-xl flex items-center justify-center transition-colors"
              onClick={() => setEnlargedUrl(null)}
            >
              {'\u{2715}'}
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
