import { useState, useEffect, useRef, useCallback } from 'react'
import type { AgentMessage, ChatMessage, WSMessage, DebateResult } from '../types'

interface UseWebSocketReturn {
  agentMessages: AgentMessage[]
  chatMessages: ChatMessage[]
  result: DebateResult | null
  isConnected: boolean
  error: string | null
  sendMessage: (data: unknown) => void
}

let chatIdCounter = 0
function nextChatId(): string {
  return `ws-${Date.now()}-${++chatIdCounter}`
}

export function useWebSocket(sessionId: string | null): UseWebSocketReturn {
  const [agentMessages, setAgentMessages] = useState<AgentMessage[]>([])
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
  const [result, setResult] = useState<DebateResult | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

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
            const chatMsg: ChatMessage = {
              id: nextChatId(),
              role: 'system',
              content: typeof msg.data === 'object' && msg.data !== null
                ? (msg.data as { question?: string }).question ?? JSON.stringify(msg.data)
                : String(msg.data),
              timestamp: Date.now(),
            }
            setChatMessages((prev) => [...prev, chatMsg])
            break
          }

          case 'avd_sufficient': {
            const chatMsg: ChatMessage = {
              id: nextChatId(),
              role: 'system',
              content: '信息采集完成，正在进入专家辩论阶段...',
              timestamp: Date.now(),
            }
            setChatMessages((prev) => [...prev, chatMsg])
            break
          }

          case 'status': {
            const chatMsg: ChatMessage = {
              id: nextChatId(),
              role: 'system',
              content: String(
                typeof msg.data === 'object' && msg.data !== null
                  ? (msg.data as { message?: string }).message ?? JSON.stringify(msg.data)
                  : msg.data,
              ),
              timestamp: Date.now(),
            }
            setChatMessages((prev) => [...prev, chatMsg])
            break
          }

          case 'result': {
            setResult(msg.data as DebateResult)
            break
          }

          case 'error': {
            setError(
              typeof msg.data === 'string'
                ? msg.data
                : (msg.data as { message?: string }).message ?? '发生未知错误',
            )
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
      reconnectTimer.current = setTimeout(() => {
        connect()
      }, 2000)
    }

    ws.onerror = () => {
      ws.close()
    }
  }, [sessionId])

  useEffect(() => {
    setAgentMessages([])
    setChatMessages([])
    setResult(null)
    setError(null)
    connect()

    return () => {
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

  return { agentMessages, chatMessages, result, isConnected, error, sendMessage }
}
