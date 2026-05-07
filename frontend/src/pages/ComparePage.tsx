export default function ComparePage() {
  return (
    <div className="flex-1 flex items-center justify-center p-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full max-w-4xl">
        {/* Left placeholder */}
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-8 flex flex-col items-center justify-center min-h-[320px]">
          <div className="text-5xl mb-4">🧪</div>
          <h3 className="text-lg font-semibold text-gray-700 mb-2">单模型诊断</h3>
          <p className="text-sm text-gray-400 text-center">
            传统单一模型推理结果
          </p>
        </div>

        {/* Right placeholder */}
        <div className="bg-white rounded-xl border-2 border-emerald-300 shadow-sm p-8 flex flex-col items-center justify-center min-h-[320px]">
          <div className="text-5xl mb-4">🌾</div>
          <h3 className="text-lg font-semibold text-emerald-700 mb-2">AgriMind AVD</h3>
          <p className="text-sm text-gray-400 text-center">
            多智能体对抗验证诊断
          </p>
        </div>
      </div>

      {/* Coming soon overlay */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <div className="bg-white/80 backdrop-blur-sm rounded-2xl px-8 py-4 shadow-lg border border-gray-200">
          <p className="text-lg font-semibold text-gray-600">
            对比演示页面 — 开发中
          </p>
        </div>
      </div>
    </div>
  )
}
