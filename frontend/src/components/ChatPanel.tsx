import { useState, useRef, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import type { ChatMessage } from '../types'

/* ── props ───────────────────────────────────────────────── */

interface ChatPanelProps {
  messages: ChatMessage[]
  onSend: (text: string) => void
  onImageAdd?: (file: File) => void
  isThinking?: boolean
  disabled?: boolean
}

/* ── typing indicator ────────────────────────────────────── */

function TypingIndicator() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -4 }}
      className="flex items-center gap-2 px-4 py-3 bg-emerald-50 rounded-xl max-w-[280px] border border-emerald-100"
    >
      <div className="flex gap-1">
        {[0, 1, 2].map((i) => (
          <motion.span
            key={i}
            className="w-2 h-2 rounded-full bg-emerald-400"
            animate={{ y: [0, -6, 0] }}
            transition={{
              duration: 0.6,
              repeat: Infinity,
              delay: i * 0.15,
              ease: 'easeInOut',
            }}
          />
        ))}
      </div>
      <span className="text-xs text-emerald-600 font-medium">AI 正在思考...</span>
    </motion.div>
  )
}

/* ── message bubble ──────────────────────────────────────── */

function MessageBubble({ msg }: { msg: ChatMessage }) {
  if (msg.role === 'system') {
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex justify-start"
      >
        <div className="max-w-[85%] rounded-2xl rounded-tl-sm px-4 py-3 bg-gradient-to-br from-emerald-50 to-teal-50 border border-emerald-200 shadow-sm">
          <div className="flex items-center gap-1.5 mb-1.5">
            <span className="w-5 h-5 rounded-full bg-emerald-500 flex items-center justify-center text-white text-[10px] font-bold">
              AI
            </span>
            <span className="text-[10px] text-emerald-600 font-medium">
              智能问诊
            </span>
          </div>
          <p className="text-sm leading-relaxed text-gray-800 whitespace-pre-wrap">
            {msg.content}
          </p>
        </div>
      </motion.div>
    )
  }

  if (msg.role === 'user') {
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex justify-end"
      >
        <div className="max-w-[85%]">
          {msg.imageUrl && (
            <div className="mb-1.5 flex justify-end">
              <img
                src={msg.imageUrl}
                alt="上传图片"
                className="h-32 rounded-xl object-cover border border-gray-200 shadow-sm"
              />
            </div>
          )}
          {msg.content && (
            <div className="rounded-2xl rounded-tr-sm px-4 py-3 bg-emerald-600 text-white shadow-sm">
              <p className="text-sm leading-relaxed whitespace-pre-wrap">
                {msg.content}
              </p>
            </div>
          )}
        </div>
      </motion.div>
    )
  }

  // assistant
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex justify-start"
    >
      <div className="max-w-[85%] rounded-2xl rounded-tl-sm px-4 py-3 bg-gray-100 border border-gray-200 shadow-sm">
        <p className="text-sm leading-relaxed text-gray-800 whitespace-pre-wrap">
          {msg.content}
        </p>
      </div>
    </motion.div>
  )
}

/* ── main component ──────────────────────────────────────── */

export default function ChatPanel({
  messages,
  onSend,
  onImageAdd,
  isThinking = false,
  disabled = false,
}: ChatPanelProps) {
  const [input, setInput] = useState('')
  const scrollRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  // Auto-scroll
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages.length, isThinking])

  const handleSend = useCallback(() => {
    const text = input.trim()
    if (!text || disabled) return
    onSend(text)
    setInput('')
    inputRef.current?.focus()
  }, [input, disabled, onSend])

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        handleSend()
      }
    },
    [handleSend],
  )

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file && onImageAdd) {
        if (!file.type.match(/^image\/(jpeg|png)$/)) {
          return
        }
        onImageAdd(file)
      }
      // reset so same file can be selected again
      e.target.value = ''
    },
    [onImageAdd],
  )

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-2 px-1 pb-3 border-b border-gray-100">
        <span className="text-lg">{'\u{1F4AC}'}</span>
        <span className="text-sm font-bold text-gray-700">智能问诊</span>
        <span className="ml-auto text-[10px] text-gray-400">
          {messages.length} 条消息
        </span>
      </div>

      {/* Messages area */}
      <div
        ref={scrollRef}
        className="flex-1 flex flex-col gap-3 overflow-y-auto pt-3 pr-1 scroll-smooth"
      >
        {messages.length === 0 && !isThinking && (
          <div className="flex flex-col items-center justify-center py-16 text-gray-400">
            <div className="text-5xl mb-3">{'\u{1F4AC}'}</div>
            <p className="text-sm">上传图片后，AI 将在此进行智能问诊</p>
            <p className="text-xs mt-1 text-gray-300">
              您也可以直接输入 &ldquo;没有了&rdquo; 跳过问诊
            </p>
          </div>
        )}

        <AnimatePresence initial={false}>
          {messages.map((msg) => (
            <MessageBubble key={msg.id} msg={msg} />
          ))}
        </AnimatePresence>

        <AnimatePresence>
          {isThinking && <TypingIndicator />}
        </AnimatePresence>
      </div>

      {/* Input area */}
      <div className="pt-3 border-t border-gray-100">
        <div className="flex items-center gap-2">
          {/* Attach image */}
          {onImageAdd && (
            <>
              <input
                ref={fileRef}
                type="file"
                accept="image/jpeg,image/png"
                onChange={handleFileSelect}
                className="hidden"
              />
              <button
                type="button"
                onClick={() => fileRef.current?.click()}
                disabled={disabled}
                className="flex-shrink-0 w-9 h-9 rounded-lg bg-gray-100 hover:bg-gray-200 flex items-center justify-center text-gray-500 transition-colors disabled:opacity-40"
                title="添加图片"
              >
                {'\u{1F4F7}'}
              </button>
            </>
          )}

          {/* Text input */}
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={disabled}
            placeholder={disabled ? '等待中...' : '输入回复，或输入"没有了"跳过'}
            className="flex-1 h-10 px-4 rounded-xl border border-gray-200 bg-white text-sm text-gray-800 placeholder-gray-400 outline-none focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100 transition-all disabled:bg-gray-50 disabled:text-gray-400"
          />

          {/* Send */}
          <button
            type="button"
            onClick={handleSend}
            disabled={disabled || !input.trim()}
            className="flex-shrink-0 h-10 px-4 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white text-sm font-medium transition-colors disabled:opacity-40 disabled:cursor-not-allowed shadow-sm"
          >
            发送
          </button>
        </div>
      </div>
    </div>
  )
}
