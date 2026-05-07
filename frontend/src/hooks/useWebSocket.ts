import { useState, useEffect, useRef, useCallback } from 'react'
import type { AgentMessage } from '../types'

interface UseWebSocketReturn {
  messages: AgentMessage[]
  isConnected: boolean
  sendMessage: (data: unknown) => void
}

export function useWebSocket(sessionId: string | null): UseWebSocketReturn {
  const [messages, setMessages] = useState<AgentMessage[]>([])
  const [isConnected, setIsConnected] = useState(false)
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
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current)
        reconnectTimer.current = null
      }
    }

    ws.onmessage = (event) => {
      try {
        const msg: AgentMessage = JSON.parse(event.data)
        setMessages((prev) => [...prev, msg])
      } catch {
        console.warn('Failed to parse WebSocket message:', event.data)
      }
    }

    ws.onclose = () => {
      setIsConnected(false)
      wsRef.current = null
      // Auto-reconnect after 2 seconds
      reconnectTimer.current = setTimeout(() => {
        connect()
      }, 2000)
    }

    ws.onerror = () => {
      ws.close()
    }
  }, [sessionId])

  useEffect(() => {
    setMessages([])
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

  return { messages, isConnected, sendMessage }
}
