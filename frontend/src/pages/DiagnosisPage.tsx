import { useState } from 'react'
import ImageUpload from '../components/ImageUpload'
import ChatPanel from '../components/ChatPanel'
import DebateViewer from '../components/DebateViewer'
import DiagnosisReport from '../components/DiagnosisReport'
import { useDiagnosis } from '../hooks/useDiagnosis'

export default function DiagnosisPage() {
  const { session, uploadImage, startDiagnosis } = useDiagnosis()
  const [activeTab, setActiveTab] = useState<'chat' | 'debate'>('chat')

  const isDebating = session?.status === 'debating' || session?.status === 'complete'

  return (
    <div className="flex-1 flex flex-col gap-6 p-6">
      {/* Top: two-column layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1 min-h-0">
        {/* Left column — upload & images */}
        <div className="lg:col-span-1 flex flex-col gap-4">
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4">
            <h2 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
              <span>📤</span> 上传病害图片
            </h2>
            <ImageUpload onUpload={uploadImage} />
          </div>

          {/* Thumbnails */}
          {session && session.images.length > 0 && (
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4">
              <h2 className="text-sm font-semibold text-gray-700 mb-3">已上传图片</h2>
              <div className="grid grid-cols-3 gap-2">
                {session.images.map((src, i) => (
                  <img
                    key={i}
                    src={src}
                    alt={`上传 ${i + 1}`}
                    className="rounded-lg w-full h-20 object-cover border border-gray-100"
                  />
                ))}
              </div>
            </div>
          )}

          {/* Start diagnosis button */}
          {session && session.status === 'questioning' && (
            <button
              onClick={startDiagnosis}
              className="w-full py-3 px-4 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white font-semibold text-sm transition-colors shadow-md hover:shadow-lg"
            >
              🚀 开始 AI 辩论诊断
            </button>
          )}

          {/* Status indicator */}
          {session && (
            <div className="flex items-center gap-2 text-sm text-gray-500">
              <span
                className={`w-2 h-2 rounded-full ${
                  session.status === 'complete'
                    ? 'bg-emerald-500'
                    : session.status === 'debating'
                      ? 'bg-yellow-500 animate-pulse'
                      : 'bg-gray-400'
                }`}
              />
              {{
                uploading: '正在上传…',
                questioning: '就绪 — 点击开始诊断',
                debating: 'AI 智能体辩论中…',
                complete: '诊断完成',
              }[session.status]}
            </div>
          )}
        </div>

        {/* Right column — chat / debate */}
        <div className="lg:col-span-2 flex flex-col bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
          {/* Tab bar */}
          <div className="flex border-b border-gray-200">
            <button
              onClick={() => setActiveTab('chat')}
              className={`flex-1 py-3 text-sm font-medium transition-colors ${
                activeTab === 'chat'
                  ? 'text-emerald-700 border-b-2 border-emerald-600 bg-emerald-50/50'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              💬 对话面板
            </button>
            <button
              onClick={() => setActiveTab('debate')}
              className={`flex-1 py-3 text-sm font-medium transition-colors ${
                activeTab === 'debate'
                  ? 'text-emerald-700 border-b-2 border-emerald-600 bg-emerald-50/50'
                  : 'text-gray-500 hover:text-gray-700'
              } ${!isDebating ? 'opacity-50 cursor-not-allowed' : ''}`}
              disabled={!isDebating}
            >
              ⚡ 辩论过程
            </button>
          </div>

          {/* Content */}
          <div className="flex-1 p-4 overflow-y-auto">
            {activeTab === 'chat' ? (
              <ChatPanel messages={session?.messages ?? []} />
            ) : (
              <DebateViewer messages={session?.messages ?? []} />
            )}
          </div>
        </div>
      </div>

      {/* Bottom: report */}
      {session?.result && (
        <div>
          <h2 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
            <span>📋</span> 诊断报告
          </h2>
          <DiagnosisReport result={session.result} />
        </div>
      )}
    </div>
  )
}
