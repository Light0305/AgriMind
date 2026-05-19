import { useState, useEffect, useRef, useCallback } from 'react'
import type { AgentMessage, AVDQuestion, AVDSufficient, ChatMessage, WSMessage, DebateResult } from '../types'

/* ── Return type ────────────────────────────────────────── */

interface UseWebSocketReturn {
  agentMessages: AgentMessage[]
  chatMessages: ChatMessage[]
  result: DebateResult | null
  isConnected: boolean
  error: string | null
  /** True once the server has sent "status: debating" */
  isDebating: boolean
  /** True once the server has sent a "result" message */
  isComplete: boolean
  /** Send a JSON payload over the WebSocket */
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
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  /** Prevent reconnect after a clean close (result received or user navigated away). */
  const shouldReconnect = useRef(true)

  const connect = useCallback(() => {
    if (!sessionId) return

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    const url = `${protocol}//${host}/ws/diagnose/${sessionId}`

    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      setIsConnected(true)
      setError(null)
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current)
        reconnectTimer.current = null
      }
      // Send API config if using API mode
      if (apiKey) {
        ws.send(JSON.stringify({ type: 'config', api_key: apiKey }))
      }
    }

    ws.onmessage = (event) => {
      try {
        const msg: WSMessage = JSON.parse(event.data)

        switch (msg.type) {
          case 'agent_message': {
            const agentMsg = msg.data as AgentMessage
            setAgentMessages((prev) => [...prev, agentMsg])
            break
          }

          case 'avd_question': {
            const q = msg.data as AVDQuestion
            const chatMsg: ChatMessage = {
              id: nextChatId(),
              role: 'system',
              content: [
                q.question,
                q.reason ? `\n\n> ${q.reason}` : '',
                q.target_part ? `\n\n请拍摄：**${q.target_part}**` : '',
              ].join(''),
              timestamp: Date.now(),
            }
            setChatMessages((prev) => [...prev, chatMsg])
            break
          }

          case 'avd_sufficient': {
            const s = msg.data as AVDSufficient
            const chatMsg: ChatMessage = {
              id: nextChatId(),
              role: 'system',
              content: s.forced
                ? `信息采集已结束（用户跳过）。${s.summary}\n正在进入专家辩论阶段...`
                : `信息采集完成（置信度 ${Math.round(s.confidence * 100)}%）。${s.summary}\n正在进入专家辩论阶段...`,
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
            shouldReconnect.current = false
            break
          }

          case 'error': {
            const errData = msg.data as { message?: string }
            setError(errData.message ?? '发生未知错误')
            break
          }
        }
      } catch {
        // Legacy fallback: plain AgentMessage
        try {
          const agentMsg: AgentMessage = JSON.parse(event.data)
          if (agentMsg.role && agentMsg.content) {
            setAgentMessages((prev) => [...prev, agentMsg])
          }
        } catch {
          console.warn('Failed to parse WebSocket message:', event.data)
        }
      }
    }

    ws.onclose = () => {
      setIsConnected(false)
      wsRef.current = null
      // Only reconnect if we haven't completed / aren't being torn down
      if (shouldReconnect.current) {
        reconnectTimer.current = setTimeout(() => {
          connect()
        }, 2000)
      }
    }

    ws.onerror = () => {
      ws.close()
    }
  }, [sessionId])

  useEffect(() => {
    // Reset state when session changes
    setAgentMessages([])
    setChatMessages([])
    setResult(null)
    setError(null)
    setIsDebating(false)
    setIsComplete(false)
    shouldReconnect.current = true

    connect()

    return () => {
      shouldReconnect.current = false
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current)
      }
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, [connect])

  const sendMessage = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data))
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
