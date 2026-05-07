export default function ExperimentPage() {
  return (
    <div className="flex-1 flex items-center justify-center p-6">
      <div className="max-w-md w-full bg-white rounded-xl border border-gray-200 shadow-sm p-8 flex flex-col items-center">
        <div className="text-5xl mb-4">📊</div>
        <h3 className="text-lg font-semibold text-gray-700 mb-2">
          实验数据页面
        </h3>
        <p className="text-sm text-gray-400 text-center mb-6">
          实验数据页面 — 开发中
        </p>

        {/* Placeholder chart preview */}
        <div className="w-full space-y-3">
          {['准确率对比', '推理延迟', '置信度分布'].map((label, i) => (
            <div key={i} className="flex items-center gap-3">
              <span className="text-xs text-gray-500 w-20 text-right">{label}</span>
              <div className="flex-1 bg-gray-100 rounded-full h-3 overflow-hidden">
                <div
                  className="h-full bg-emerald-400 rounded-full"
                  style={{ width: `${60 + i * 15}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
