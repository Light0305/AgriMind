import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import type { DebateResult } from '../types'

/* ── props ───────────────────────────────────────────────── */

interface DiagnosisReportProps {
  result: DebateResult
}

/* ── confidence badge config ─────────────────────────────── */

const confidenceConfig: Record<
  string,
  { text: string; color: string; icon: string }
> = {
  high: {
    text: '高置信度',
    color: 'bg-emerald-100 text-emerald-800 border-emerald-300',
    icon: '\u{1F7E2}',
  },
  medium: {
    text: '中置信度',
    color: 'bg-yellow-100 text-yellow-800 border-yellow-300',
    icon: '\u{1F7E1}',
  },
  low: {
    text: '低置信度',
    color: 'bg-red-100 text-red-800 border-red-300',
    icon: '\u{1F534}',
  },
}

/* ── section wrapper ─────────────────────────────────────── */

function Section({
  icon,
  title,
  children,
}: {
  icon: string
  title: string
  children: React.ReactNode
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <span className="text-base">{icon}</span>
        <h4 className="text-sm font-bold text-gray-700">{title}</h4>
      </div>
      {children}
    </div>
  )
}

/* ── main ────────────────────────────────────────────────── */

export default function DiagnosisReport({ result }: DiagnosisReportProps) {
  const [rejectedOpen, setRejectedOpen] = useState(false)
  const conf = confidenceConfig[result.confidence] ?? confidenceConfig.medium

  const handleExportPDF = () => {
    console.log('[DiagnosisReport] Export PDF requested', result)
    alert('PDF 导出功能即将上线')
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      className="rounded-2xl border border-emerald-200 bg-white shadow-xl overflow-hidden"
    >
      {/* Header banner */}
      <div className="bg-gradient-to-r from-emerald-600 via-emerald-500 to-teal-500 px-6 py-5 text-white">
        <div className="flex items-start gap-4">
          <div className="w-14 h-14 rounded-2xl bg-white/20 backdrop-blur-sm flex items-center justify-center text-3xl flex-shrink-0">
            {'\u{1F33F}'}
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="text-xl font-bold tracking-tight truncate">
              {result.final_diagnosis}
            </h3>
            <div className="flex items-center gap-2 mt-1.5">
              <span
                className={`inline-flex items-center gap-1 text-xs font-bold px-2.5 py-1 rounded-full border ${conf.color}`}
              >
                {conf.icon} {conf.text}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Body */}
      <div className="p-6 space-y-6">
        {/* Supporting evidence */}
        <Section icon={'\u{2705}'} title="支持证据">
          <div className="space-y-2">
            {result.supporting_evidence.map((ev, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -12 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.06 }}
                className="flex items-start gap-3 text-sm text-gray-700"
              >
                <span className="flex-shrink-0 w-6 h-6 rounded-full bg-emerald-100 text-emerald-700 flex items-center justify-center text-xs font-bold mt-0.5">
                  {i + 1}
                </span>
                <span className="leading-relaxed">{ev}</span>
              </motion.div>
            ))}
          </div>
        </Section>

        {/* Rejected diagnoses (collapsible) */}
        {result.rejected_diagnoses.length > 0 && (
          <Section icon={'\u{274C}'} title="已排除诊断">
            <button
              type="button"
              onClick={() => setRejectedOpen(!rejectedOpen)}
              className="text-xs text-gray-500 hover:text-gray-700 transition-colors flex items-center gap-1"
            >
              <motion.span
                animate={{ rotate: rejectedOpen ? 90 : 0 }}
                transition={{ duration: 0.2 }}
              >
                {'\u{25B6}'}
              </motion.span>
              {rejectedOpen ? '收起' : `展开 ${result.rejected_diagnoses.length} 项`}
            </button>
            <AnimatePresence>
              {rejectedOpen && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.3, ease: 'easeInOut' }}
                  className="overflow-hidden"
                >
                  <div className="space-y-2 pt-1">
                    {result.rejected_diagnoses.map((rd, i) => (
                      <div
                        key={i}
                        className="flex items-start gap-2 text-sm bg-red-50/50 rounded-lg px-3 py-2 border border-red-100"
                      >
                        <span className="text-red-400 mt-0.5">{'\u{2717}'}</span>
                        <div>
                          <span className="font-semibold text-gray-800">
                            {rd.name}
                          </span>
                          <span className="text-gray-500"> — {rd.reason}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </Section>
        )}

        {/* Uncertainty notes */}
        {result.uncertainty_notes.length > 0 && (
          <Section icon={'\u{26A0}\u{FE0F}'} title="不确定性说明">
            <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 space-y-1.5">
              {result.uncertainty_notes.map((note, i) => (
                <p key={i} className="text-sm text-amber-800 flex items-start gap-2">
                  <span className="mt-0.5 text-amber-500">{'\u{25CF}'}</span>
                  {note}
                </p>
              ))}
            </div>
          </Section>
        )}

        {/* Treatment recommendations */}
        {result.treatment && (
          <Section icon={'\u{1F48A}'} title="防治建议">
            <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
              <p className="text-sm text-blue-900 leading-relaxed whitespace-pre-wrap">
                {result.treatment.text}
              </p>
              {result.treatment.source && (
                <p className="text-[10px] text-blue-400 mt-2">
                  来源: {result.treatment.source}
                </p>
              )}
            </div>
          </Section>
        )}

        {/* Actions */}
        <div className="flex items-center gap-3 pt-2 border-t border-gray-100">
          <button
            type="button"
            onClick={handleExportPDF}
            className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm font-medium transition-colors"
          >
            {'\u{1F4C4}'} 导出报告
          </button>
          <span className="text-[10px] text-gray-400 ml-auto">
            AgriMind 智能诊断报告
          </span>
        </div>
      </div>
    </motion.div>
  )
}
