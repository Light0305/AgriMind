import { useRef, useEffect, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import type { AgentMessage } from '../types'

/* ── role styling ────────────────────────────────────────── */

const roleMeta: Record<
  AgentMessage['role'],
  { label: string; labelFull: string; icon: string; border: string; bg: string; text: string; accent: string }
> = {
  proposer: {
    label: '初诊专家',
    labelFull: '初诊专家 · Proposer',
    icon: '\u{1F468}\u{200D}\u{2695}\u{FE0F}',
    border: 'border-l-blue-500',
    bg: 'bg-blue-50/70',
    text: 'text-blue-800',
    accent: 'bg-blue-500',
  },
  challenger: {
    label: '质疑专家',
    labelFull: '质疑专家 · Challenger',
    icon: '\u{1F50D}',
    border: 'border-l-orange-500',
    bg: 'bg-orange-50/70',
    text: 'text-orange-800',
    accent: 'bg-orange-500',
  },
  arbiter: {
    label: '仲裁专家',
    labelFull: '仲裁专家 · Arbiter',
    icon: '\u{1F3DB}\u{FE0F}',
    border: 'border-l-purple-600',
    bg: 'bg-purple-50/80',
    text: 'text-purple-900',
    accent: 'bg-purple-600',
  },
}

/* ── simple markdown‐ish renderer ────────────────────────── */

function renderContent(raw: string) {
  // Split by **bold** patterns; bold odd segments
  const parts = raw.split(/\*\*(.+?)\*\*/g)
  return (
    <span>
      {parts.map((part, i) =>
        i % 2 === 1 ? (
          <strong key={i} className="font-semibold">
            {part}
          </strong>
        ) : (
          <span key={i}>{part}</span>
        ),
      )}
    </span>
  )
}

/* ── loading dots ────────────────────────────────────────── */

function LoadingDots() {
  return (
    <div className="flex items-center gap-1 py-8 justify-center">
      {[0, 1, 2].map((i) => (
        <motion.span
          key={i}
          className="w-2.5 h-2.5 rounded-full bg-emerald-400"
          animate={{ opacity: [0.3, 1, 0.3], scale: [0.8, 1.1, 0.8] }}
          transition={{
            duration: 1.2,
            repeat: Infinity,
            delay: i * 0.2,
            ease: 'easeInOut',
          }}
        />
      ))}
      <span className="ml-3 text-sm text-gray-400 font-medium">
        正在分析...
      </span>
    </div>
  )
}

/* ── round divider ───────────────────────────────────────── */

function RoundDivider({ round }: { round: number }) {
  return (
    <div className="flex items-center gap-3 py-2">
      <div className="flex-1 h-px bg-gradient-to-r from-transparent via-gray-300 to-transparent" />
      <span className="text-xs font-bold text-gray-400 tracking-widest uppercase">
        Round {round}
      </span>
      <div className="flex-1 h-px bg-gradient-to-r from-transparent via-gray-300 to-transparent" />
    </div>
  )
}

/* ── evidence badges ─────────────────────────────────────── */

function EvidenceBadges({ content }: { content: string }) {
  const refs = content.match(/\[(\d+)\]/g)
  if (!refs || refs.length === 0) return null
  return (
    <div className="flex gap-1 mt-2 flex-wrap">
      {refs.map((ref, i) => (
        <span
          key={i}
          className="inline-flex items-center justify-center w-6 h-6 text-[10px] font-bold rounded-full bg-white/80 text-gray-600 border border-gray-200 shadow-sm"
        >
          {ref.replace(/[\[\]]/g, '')}
        </span>
      ))}
    </div>
  )
}

/* ── main component ──────────────────────────────────────── */

interface DebateViewerProps {
  messages: AgentMessage[]
  isLoading?: boolean
}

export default function DebateViewer({
  messages,
  isLoading = false,
}: DebateViewerProps) {
  const scrollRef = useRef<HTMLDivElement>(null)

  // Auto-scroll on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages.length])

  // Group messages to detect round boundaries
  const rounds = useMemo(() => {
    const seen = new Set<number>()
    const boundaries: number[] = []
    for (let i = 0; i < messages.length; i++) {
      const r = messages[i].round
      if (!seen.has(r)) {
        seen.add(r)
        boundaries.push(i)
      }
    }
    return boundaries
  }, [messages])

  // Current round for display
  const currentRound =
    messages.length > 0 ? messages[messages.length - 1].round : 0

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-1 pb-3 border-b border-gray-100">
        <div className="flex items-center gap-2">
          <span className="text-lg">{'\u{1F52C}'}</span>
          <span className="text-sm font-bold text-gray-700">专家辩论实况</span>
        </div>
        {currentRound > 0 && (
          <span className="text-xs font-mono text-gray-400 bg-gray-100 px-2.5 py-1 rounded-full">
            Round {currentRound}
          </span>
        )}
      </div>

      {/* Messages */}
      <div
        ref={scrollRef}
        className="flex-1 flex flex-col gap-3 overflow-y-auto pt-3 pr-1 scroll-smooth"
      >
        {messages.length === 0 && !isLoading && (
          <div className="flex flex-col items-center justify-center py-16 text-gray-400">
            <motion.div
              className="text-5xl mb-3"
              animate={{ opacity: [0.4, 1, 0.4] }}
              transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
            >
              {'\u{26A1}'}
            </motion.div>
            <p className="text-sm">正在等待辩论开始...</p>
          </div>
        )}

        <AnimatePresence initial={false}>
          {messages.map((msg, i) => {
            const meta = roleMeta[msg.role]
            const showDivider = rounds.includes(i) && i > 0

            return (
              <div key={i}>
                {showDivider && <RoundDivider round={msg.round} />}

                <motion.div
                  initial={{ opacity: 0, x: -24, scale: 0.97 }}
                  animate={{ opacity: 1, x: 0, scale: 1 }}
                  exit={{ opacity: 0, x: 16 }}
                  transition={{
                    duration: 0.4,
                    ease: [0.22, 1, 0.36, 1],
                  }}
                  className={`
                    relative rounded-xl border-l-4 ${meta.border} ${meta.bg}
                    p-4 shadow-sm hover:shadow-md transition-shadow
                    ${msg.role === 'arbiter' ? 'ring-1 ring-purple-200' : ''}
                  `}
                >
                  {/* Role header */}
                  <div className="flex items-center gap-2 mb-2.5">
                    <span
                      className={`w-8 h-8 rounded-full flex items-center justify-center text-base ${meta.accent} text-white shadow-sm`}
                    >
                      {meta.icon}
                    </span>
                    <div className="flex flex-col">
                      <span
                        className={`text-xs font-bold tracking-wide ${meta.text}`}
                      >
                        {meta.label}
                      </span>
                      <span className="text-[10px] text-gray-400">
                        第 {msg.round} 轮发言
                      </span>
                    </div>
                    <span className="ml-auto text-xs text-gray-300 font-mono">
                      #{i + 1}
                    </span>
                  </div>

                  {/* Content */}
                  <div
                    className={`text-sm leading-relaxed whitespace-pre-wrap ${
                      msg.role === 'arbiter' ? 'text-purple-900' : 'text-gray-700'
                    }`}
                  >
                    {renderContent(msg.content)}
                  </div>

                  {/* Evidence badges */}
                  <EvidenceBadges content={msg.content} />

                  {/* Arbiter verdict marker */}
                  {msg.role === 'arbiter' && (
                    <div className="mt-3 pt-2.5 border-t border-purple-200/50 flex items-center gap-2">
                      <span className="text-xs font-bold text-purple-700">
                        {'\u{2705}'} 仲裁结论
                      </span>
                    </div>
                  )}
                </motion.div>
              </div>
            )
          })}
        </AnimatePresence>

        {isLoading && <LoadingDots />}
      </div>
    </div>
  )
}
