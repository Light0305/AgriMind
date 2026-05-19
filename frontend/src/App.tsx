import { BrowserRouter, Routes, Route } from 'react-router-dom'
import DiagnosisPage from './pages/DiagnosisPage'

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50 flex flex-col">
        {/* Header */}
        <header className="bg-white border-b border-gray-200 shadow-sm sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-14">
              {/* Logo */}
              <div className="flex items-center gap-2">
                <span className="text-2xl">🌾</span>
                <span className="text-lg font-bold text-emerald-700 tracking-tight">
                  AgriMind
                </span>
                <span className="text-xs text-gray-400 ml-1 hidden sm:inline">
                  作物病害智能诊断系统
                </span>
              </div>
            </div>
          </div>
        </header>

        {/* Main content */}
        <main className="flex-1 flex flex-col max-w-7xl w-full mx-auto">
          <Routes>
            <Route path="/" element={<DiagnosisPage />} />
          </Routes>
        </main>

        {/* Footer */}
        <footer className="border-t border-gray-200 bg-white py-4 text-center text-xs text-gray-400">
          AgriMind © 2026 · 多智能体对抗验证诊断系统 · 西北农林科技大学
        </footer>
      </div>
    </BrowserRouter>
  )
}
