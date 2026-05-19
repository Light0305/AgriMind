import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import ImageUpload from '../components/ImageUpload'
import ImageAnnotation from '../components/ImageAnnotation'
import ChatPanel from '../components/ChatPanel'
import DebateViewer from '../components/DebateViewer'
import DiagnosisReport from '../components/DiagnosisReport'
import SimilarCases from '../components/SimilarCases'
import { useDiagnosis } from '../hooks/useDiagnosis'

/* -- status label map  --------------------------------------------------- */

const statusLabels: Record<string, { text: string; dot: string }> = {
  idle: { text: '就绪 — 请上传病害图片', dot: 'bg-gray-400' },
  uploading: { text: '正在上传...', dot: 'bg-blue-500 animate-pulse' },
  questioning: {
    text: 'AI 问诊中 — 请回答问题或点击"跳过问诊"',
    dot: 'bg-amber-500 animate-pulse',
  },
  debating: {
    text: 'AI 专家辩论进行中...',
    dot: 'bg-yellow-500 animate-pulse',
  },
  complete: { text: '诊断完成', dot: 'bg-emerald-500' },
  error: { text: '出现错误', dot: 'bg-red-500' },
}

/* -- main page ----------------------------------------------------------- */

export default function DiagnosisPage() {
  const {
    session,
    apiKey,
    setApiKey,
    uploadImage,
    addChatImage,
    sendChatMessage,
    skipToDebate,
    resetSession,
  } = useDiagnosis()

  const [activeTab, setActiveTab] = useState<'chat' | 'debate'>('chat')
  const [selectedImageIdx, setSelectedImageIdx] = useState(0)

  const status = session?.status ?? 'idle'
  const isDebating = status === 'debating' || status === 'complete'
  const hasResult = !!session?.result
  const images = session?.images ?? []
  const boxes = session?.result?.grounding_boxes ?? []

  // Auto-switch to debate tab when debating starts
  useEffect(() => {
    if (status === 'debating') {
      setActiveTab('debate')
    }
  }, [status])

  const statusInfo = statusLabels[status] ?? statusLabels.idle

  return (
    <div className="flex-1 flex flex-col gap-6 p-6">
      {/* -- API mode toggle ---------------------------------------- */}
      <div className="flex items-center gap-3 px-4 py-2 bg-gray-50 rounded-lg text-sm">
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={!!apiKey}
            onChange={(e) => {
              if (e.target.checked) {
                setApiKey('sk-1aa3647be3dc4122ba4a9487aab9d7da')
              } else {
                setApiKey('')
              }
            }}
            className="w-4 h-4 rounded accent-emerald-600"
          />
          <span className="font-medium">API 模式 (无需本地 GPU)</span>
        </label>
        {apiKey && (
          <input
            type="text"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="输入 DashScope API Key"
            className="flex-1 px-2 py-1 border rounded text-xs font-mono"
          />
        )}
        {!apiKey && (
          <span className="text-gray-400 text-xs">使用本地 Qwen2.5-VL-7B 模型</span>
        )}
      </div>

      {/* -- Top: two-column layout ----------------------------------- */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1 min-h-0">
        {/* --- Left column: images ----------------------------------- */}
        <div className="lg:col-span-4 xl:col-span-3 flex flex-col gap-4">
          {/* Upload card */}
          <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-4">
            <h2 className="text-sm font-bold text-gray-700 mb-3 flex items-center gap-2">
              <span>{'\u{1F4F8}'}</span> 上传病害图片
            </h2>
            <ImageUpload
              onUpload={uploadImage}
              images={images}
              disabled={status === 'debating' || status === 'uploading' || status === 'complete'}
            />
          </div>

          {/* Image annotation card (when we have images) */}
          {images.length > 0 && (
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-4">
              <h2 className="text-sm font-bold text-gray-700 mb-3 flex items-center gap-2">
                <span>{'\u{1F50D}'}</span> 图片标注
              </h2>

              {/* Image selector tabs when multiple images */}
              {images.length > 1 && (
                <div className="flex gap-1.5 mb-3 flex-wrap">
                  {images.map((_, i) => (
                    <button
                      key={i}
                      type="button"
                      onClick={() => setSelectedImageIdx(i)}
                      className={`w-8 h-8 rounded-lg text-xs font-bold transition-all ${
                        selectedImageIdx === i
                          ? 'bg-emerald-100 text-emerald-700 ring-2 ring-emerald-300'
                          : 'bg-gray-100 text-gray-500 hover:bg-gray-200'
                      }`}
                    >
                      {i + 1}
                    </button>
                  ))}
                </div>
              )}

              <ImageAnnotation
                imageUrl={images[selectedImageIdx] ?? images[0]}
                boxes={hasResult ? boxes : []}
                showLabels
              />
            </div>
          )}

          {/* Skip-to-debate button (replaces "startDiagnosis") */}
          <AnimatePresence>
            {status === 'questioning' && (
              <motion.button
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                onClick={skipToDebate}
                className="w-full py-3.5 px-4 rounded-2xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 text-white font-bold text-sm transition-all shadow-lg hover:shadow-xl active:scale-[0.98]"
              >
                {'\u{1F680}'} 跳过问诊，直接 AI 辩论诊断
              </motion.button>
            )}
          </AnimatePresence>

          {/* Error display */}
          {session?.error && (
            <div className="bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-sm text-red-700">
              {session.error}
            </div>
          )}

          {/* Status indicator */}
          <div className="flex items-center gap-2 text-sm text-gray-500 px-1">
            <span className={`w-2.5 h-2.5 rounded-full ${statusInfo.dot}`} />
            <span className="text-xs">{statusInfo.text}</span>
          </div>
        </div>

        {/* --- Right column: chat / debate ----------------------------- */}
        <div className="lg:col-span-8 xl:col-span-9 flex flex-col bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
          {/* Tab bar */}
          <div className="flex border-b border-gray-200">
            <button
              type="button"
              onClick={() => setActiveTab('chat')}
              className={`flex-1 py-3.5 text-sm font-semibold transition-all ${
                activeTab === 'chat'
                  ? 'text-emerald-700 border-b-2 border-emerald-500 bg-emerald-50/40'
                  : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
              }`}
            >
              <span className="mr-1.5">{'\u{1F4AC}'}</span>
              问诊对话
              {(session?.chatMessages?.length ?? 0) > 0 && (
                <span className="ml-2 inline-flex items-center justify-center w-5 h-5 text-[10px] font-bold rounded-full bg-emerald-100 text-emerald-700">
                  {session!.chatMessages.length}
                </span>
              )}
            </button>
            <button
              type="button"
              onClick={() => isDebating && setActiveTab('debate')}
              className={`flex-1 py-3.5 text-sm font-semibold transition-all ${
                activeTab === 'debate'
                  ? 'text-emerald-700 border-b-2 border-emerald-500 bg-emerald-50/40'
                  : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
              } ${!isDebating ? 'opacity-40 cursor-not-allowed' : ''}`}
              disabled={!isDebating}
            >
              <span className="mr-1.5">{'\u{26A1}'}</span>
              专家辩论
              {(session?.messages?.length ?? 0) > 0 && (
                <span className="ml-2 inline-flex items-center justify-center w-5 h-5 text-[10px] font-bold rounded-full bg-amber-100 text-amber-700">
                  {session!.messages.length}
                </span>
              )}
            </button>
          </div>

          {/* Content */}
          <div className="flex-1 p-4 overflow-hidden flex flex-col">
            {activeTab === 'chat' ? (
              <ChatPanel
                messages={session?.chatMessages ?? []}
                onSend={sendChatMessage}
                onImageAdd={status === 'questioning' ? addChatImage : undefined}
                isThinking={
                  status === 'questioning' &&
                  (session?.chatMessages?.length ?? 0) > 0 &&
                  session!.chatMessages[session!.chatMessages.length - 1].role ===
                    'user'
                }
                disabled={status === 'debating' || status === 'complete' || status === 'idle'}
              />
            ) : (
              <DebateViewer
                messages={session?.messages ?? []}
                isLoading={status === 'debating'}
              />
            )}
          </div>
        </div>
      </div>

      {/* -- Bottom: diagnosis report (slides up) ---------------------- */}
      <AnimatePresence>
        {hasResult && session?.result && (
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 40 }}
            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
            className="space-y-6"
          >
            <div className="flex items-center gap-2 px-1">
              <span>{'\u{1F4CB}'}</span>
              <h2 className="text-base font-bold text-gray-800">诊断报告</h2>
            </div>

            <DiagnosisReport result={session.result} />

            {/* Similar cases */}
            {session.result.similar_cases &&
              session.result.similar_cases.length > 0 && (
                <SimilarCases
                  cases={session.result.similar_cases}
                  currentImage={images[0]}
                />
              )}

            {/* New diagnosis button */}
            <div className="flex justify-center pt-4">
              <button
                onClick={resetSession}
                className="px-8 py-3.5 rounded-2xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 text-white font-bold text-base transition-all shadow-lg hover:shadow-xl active:scale-[0.98]"
              >
                {'\u{1F504}'} 开始新的诊断
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
