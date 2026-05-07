import type { AgentMessage } from '../types'

const roleMeta: Record<
  AgentMessage['role'],
  { label: string; icon: string; color: string }
> = {
  proposer: { label: '提出者', icon: '🔬', color: 'text-blue-600 bg-blue-50 border-blue-200' },
  challenger: { label: '质询者', icon: '🔍', color: 'text-orange-600 bg-orange-50 border-orange-200' },
  arbiter: { label: '仲裁者', icon: '🏛️', color: 'text-purple-600 bg-purple-50 border-purple-200' },
}

interface ChatPanelProps {
  messages: AgentMessage[]
}

export default function ChatPanel({ messages }: ChatPanelProps) {
  return (
    <div className="flex flex-col gap-3 overflow-y-auto max-h-[60vh] pr-1">
      {messages.length === 0 && (
        <div className="flex flex-col items-center justify-center py-12 text-gray-400">
          <div className="text-5xl mb-3">💬</div>
          <p className="text-sm">上传图片后，AI 智能体将在此展开对话</p>
        </div>
      )}

      {messages.map((msg, i) => {
        const meta = roleMeta[msg.role]
        return (
          <div
            key={i}
            className={`rounded-lg border p-3 ${meta.color}`}
          >
            <div className="flex items-center gap-2 mb-1">
              <span className="text-lg">{meta.icon}</span>
              <span className="text-xs font-semibold uppercase tracking-wide">
                {meta.label}
              </span>
              <span className="text-xs opacity-60 ml-auto">
                第 {msg.round} 轮
              </span>
            </div>
            <p className="text-sm leading-relaxed whitespace-pre-wrap">
              {msg.content}
            </p>
          </div>
        )
      })}
    </div>
  )
}
