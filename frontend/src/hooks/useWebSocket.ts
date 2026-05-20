import { useState, useEffect, useRef, useCallback } from 'react'
import type { AgentMessage, ChatMessage, WSMessage, DebateResult } from '../types'

/* ── Return type ────────────────────────────────────────── */

interface UseWebSocketReturn {
  agentMessages: AgentMessage[]
  chatMessages: ChatMessage[]
  result: DebateResult | null
  isConnected: boolean
  error: string | null
  isDebating: boolean
  isComplete: boolean
  sendMessage: (data: unknown) => void
}

/* ── ID generator ───────────────────────────────────────── */

let chatIdCounter = 0
function nextChatId(): string {
  return `ws-${Date.now()}-${++chatIdCounter}`
}

/* ── Hook ───────────────────────────────────────────────── */

export function useWebSocket(sessionId: string | null, apiKey?: string): UseWebSocketReturn {
  const [agentMessages, setAgentMessages] = useState<AgentMessage[]>([])
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
  const [result, setResult] = useState<DebateResult | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [isDebating, setIsDebating] = useState(false)
  const [isComplete, setIsComplete] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const apiKeyRef = useRef(apiKey)
  apiKeyRef.current = apiKey
  const sessionIdRef = useRef(sessionId)
  sessionIdRef.current = sessionId

  useEffect(() => {
    if (!sessionId) return

    // Reset state
    setAgentMessages([])
    setChatMessages([])
    setResult(null)
    setError(null)
    setIsDebating(false)
    setIsComplete(false)

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    const keyParam = apiKeyRef.current ? `?api_key=${encodeURIComponent(apiKeyRef.current)}` : ''
    const url = `${protocol}//${host}/ws/diagnose/${sessionId}${keyParam}`

    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      console.log('WS OPEN', sessionId)
      setIsConnected(true)
      setError(null)
    }

    ws.onmessage = (event) => {
      console.log('WS MSG:', event.data.substring(0, 100))
      try {
        const msg: WSMessage = JSON.parse(event.data)

        switch (msg.type) {
          case 'agent_message': {
            const agentMsg = msg.data as AgentMessage
            setAgentMessages((prev) => [...prev, agentMsg])
            break
          }

          case 'avd_question': {
            const q = msg.data as { question: string }
            const chatMsg: ChatMessage = {
              id: nextChatId(),
              role: 'system',
              content: q.question,
              timestamp: Date.now(),
            }
            setChatMessages((prev) => [...prev, chatMsg])
            break
          }

          case 'avd_sufficient': {
            const s = msg.data as { summary: string; confidence: number; forced: boolean }
            const chatMsg: ChatMessage = {
              id: nextChatId(),
              role: 'system',
              content: s.forced
                ? `${s.summary}\n正在进入专家辩论阶段...`
                : `${s.summary}\n正在进入专家辩论阶段...`,
              timestamp: Date.now(),
            }
            setChatMessages((prev) => [...prev, chatMsg])
            break
          }

          case 'status': {
            const d = msg.data as { status?: string; message?: string }
            if (d.status === 'debating') {
              setIsDebating(true)
            }
            const chatMsg: ChatMessage = {
              id: nextChatId(),
              role: 'system',
              content: d.message ?? (d.status === 'debating' ? '专家辩论已开始...' : JSON.stringify(d)),
              timestamp: Date.now(),
            }
            setChatMessages((prev) => [...prev, chatMsg])
            break
          }

          case 'result': {
            setResult(msg.data as DebateResult)
            setIsComplete(true)
            setIsDebating(false)
            break
          }

          case 'error': {
            const errData = msg.data as { message?: string }
            setError(errData.message ?? '发生未知错误')
            break
          }
        }
      } catch {
        console.warn('Failed to parse WebSocket message:', event.data)
      }
    }

    ws.onclose = (e) => {
      console.log('WS CLOSED', sessionId, 'code:', e.code, 'reason:', e.reason)
      setIsConnected(false)
      wsRef.current = null
    }

    ws.onerror = () => {
      ws.close()
    }

    // Cleanup ONLY when sessionId changes (not on every render)
    return () => {
      ws.close()
      wsRef.current = null
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId])  // ONLY depend on sessionId, nothing else

  const sendMessage = useCallback((data: unknown) => {
    const ws = wsRef.current
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(data))
    } else {
      console.error('WebSocket not open! readyState:', ws?.readyState, 'data:', data)
    }
  }, [])

  return {
    agentMessages,
    chatMessages,
    result,
    isConnected,
    error,
    isDebating,
    isComplete,
    sendMessage,
  }
}
