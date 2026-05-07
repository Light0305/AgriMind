import { motion, AnimatePresence } from 'framer-motion'
import type { AgentMessage } from '../types'

const roleMeta: Record<
  AgentMessage['role'],
  { label: string; icon: string; bg: string; border: string; text: string }
> = {
  proposer: {
    label: '提出者',
    icon: '🔬',
    bg: 'bg-blue-50',
    border: 'border-blue-300',
    text: 'text-blue-700',
  },
  challenger: {
    label: '质询者',
    icon: '🔍',
    bg: 'bg-orange-50',
    border: 'border-orange-300',
    text: 'text-orange-700',
  },
  arbiter: {
    label: '仲裁者',
    icon: '🏛️',
    bg: 'bg-purple-50',
    border: 'border-purple-300',
    text: 'text-purple-700',
  },
}

interface DebateViewerProps {
  messages: AgentMessage[]
}

export default function DebateViewer({ messages }: DebateViewerProps) {
  return (
    <div className="flex flex-col gap-3 overflow-y-auto max-h-[60vh] pr-1">
      {messages.length === 0 && (
        <div className="flex flex-col items-center justify-center py-12 text-gray-400">
          <div className="text-5xl mb-3 animate-pulse">⚡</div>
          <p className="text-sm">正在等待辩论开始…</p>
        </div>
      )}

      <AnimatePresence initial={false}>
        {messages.map((msg, i) => {
          const meta = roleMeta[msg.role]
          return (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.35, ease: 'easeOut' }}
              className={`rounded-xl border ${meta.border} ${meta.bg} p-4 shadow-sm`}
            >
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xl">{meta.icon}</span>
                <span className={`text-xs font-bold uppercase tracking-wider ${meta.text}`}>
                  {meta.label}
                </span>
                <span className="ml-auto text-xs text-gray-400 font-mono">
                  R{msg.round}
                </span>
              </div>
              <p className={`text-sm leading-relaxed whitespace-pre-wrap ${meta.text}`}>
                {msg.content}
              </p>
            </motion.div>
          )
        })}
      </AnimatePresence>
    </div>
  )
}
