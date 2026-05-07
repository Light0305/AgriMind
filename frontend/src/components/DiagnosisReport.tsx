import type { DebateResult } from '../types'

interface DiagnosisReportProps {
  result: DebateResult
}

const confidenceLabel: Record<string, { text: string; color: string }> = {
  high: { text: '高置信度', color: 'bg-emerald-100 text-emerald-800' },
  medium: { text: '中置信度', color: 'bg-yellow-100 text-yellow-800' },
  low: { text: '低置信度', color: 'bg-red-100 text-red-800' },
}

export default function DiagnosisReport({ result }: DiagnosisReportProps) {
  const conf = confidenceLabel[result.confidence] ?? confidenceLabel.medium

  return (
    <div className="rounded-xl border border-emerald-200 bg-white shadow-lg p-6 space-y-5">
      {/* Header */}
      <div className="flex items-center gap-4">
        <div className="w-12 h-12 rounded-full bg-emerald-100 flex items-center justify-center text-2xl">
          🌿
        </div>
        <div>
          <h3 className="text-lg font-bold text-gray-900">
            {result.final_diagnosis}
          </h3>
          <span className={`inline-block mt-1 text-xs font-semibold px-2 py-0.5 rounded-full ${conf.color}`}>
            {conf.text}
          </span>
        </div>
      </div>

      {/* Supporting evidence */}
      <div>
        <h4 className="text-sm font-semibold text-gray-700 mb-2">支持证据</h4>
        <ul className="space-y-1">
          {result.supporting_evidence.map((ev, i) => (
            <li key={i} className="flex items-start gap-2 text-sm text-gray-600">
              <span className="text-emerald-500 mt-0.5">✓</span>
              {ev}
            </li>
          ))}
        </ul>
      </div>

      {/* Rejected diagnoses */}
      {result.rejected_diagnoses.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold text-gray-700 mb-2">已排除诊断</h4>
          <ul className="space-y-1">
            {result.rejected_diagnoses.map((rd, i) => (
              <li key={i} className="text-sm text-gray-500">
                <span className="text-red-400">✗</span>{' '}
                <span className="font-medium">{rd.name}</span> — {rd.reason}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Uncertainty notes */}
      {result.uncertainty_notes.length > 0 && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
          <h4 className="text-sm font-semibold text-yellow-800 mb-1">不确定性说明</h4>
          <ul className="space-y-1">
            {result.uncertainty_notes.map((note, i) => (
              <li key={i} className="text-sm text-yellow-700">⚠ {note}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
