import { useState, useCallback } from 'react'
import type { DiagnosisSession, ChatMessage } from '../types'
import { useWebSocket } from './useWebSocket'

let chatIdSeq = 0
function nextId(): string {
  return `diag-${Date.now()}-${++chatIdSeq}`
}

interface UseDiagnosisReturn {
  session: DiagnosisSession | null
  uploadImage: (file: File) => Promise<void>
  addChatImage: (file: File) => void
  sendChatMessage: (text: string) => void
  startDiagnosis: () => Promise<void>
}

export function useDiagnosis(): UseDiagnosisReturn {
  const [session, setSession] = useState<DiagnosisSession | null>(null)

  const wsSessionId =
    session && (session.status === 'questioning' || session.status === 'debating')
      ? session.id
      : null

  const ws = useWebSocket(wsSessionId)

  // Sync WS agent messages into session
  if (
    session &&
    ws.agentMessages.length > 0 &&
    session.messages.length !== ws.agentMessages.length
  ) {
    setSession((prev) => (prev ? { ...prev, messages: ws.agentMessages } : prev))
  }

  // Sync WS chat messages into session
  if (
    session &&
    ws.chatMessages.length > 0 &&
    session.chatMessages.length !== ws.chatMessages.length
  ) {
    setSession((prev) => (prev ? { ...prev, chatMessages: ws.chatMessages } : prev))
  }

  // Sync result
  if (session && ws.result && !session.result) {
    setSession((prev) =>
      prev ? { ...prev, status: 'complete', result: ws.result! } : prev,
    )
  }

  const uploadImage = useCallback(async (file: File) => {
    setSession((prev) => {
      if (prev) return { ...prev, status: 'uploading' as const }
      return {
        id: '',
        status: 'uploading' as const,
        images: [],
        messages: [],
        chatMessages: [],
      }
    })

    const formData = new FormData()
    formData.append('file', file)

    try {
      const resp = await fetch('/api/diagnose', {
        method: 'POST',
        body: formData,
      })

      if (!resp.ok) throw new Error(`Upload failed: ${resp.status}`)

      const data = await resp.json()

      setSession((prev) => ({
        id: data.session_id ?? prev?.id ?? '',
        status: 'questioning' as const,
        images: [...(prev?.images ?? []), URL.createObjectURL(file)],
        messages: prev?.messages ?? [],
        chatMessages: prev?.chatMessages ?? [],
      }))
    } catch (err) {
      console.error('Upload error:', err)
      setSession((prev) =>
        prev ? { ...prev, status: 'questioning' as const } : prev,
      )
    }
  }, [])

  const addChatImage = useCallback((file: File) => {
    const url = URL.createObjectURL(file)
    setSession((prev) => {
      if (!prev) return prev
      const msg: ChatMessage = {
        id: nextId(),
        role: 'user',
        content: '',
        imageUrl: url,
        timestamp: Date.now(),
      }
      return {
        ...prev,
        images: [...prev.images, url],
        chatMessages: [...prev.chatMessages, msg],
      }
    })

    // Also upload to server
    const formData = new FormData()
    formData.append('file', file)
    if (session?.id) {
      formData.append('session_id', session.id)
    }
    fetch('/api/diagnose/image', { method: 'POST', body: formData }).catch(
      console.error,
    )
  }, [session?.id])

  const sendChatMessage = useCallback(
    (text: string) => {
      if (!text.trim()) return
      setSession((prev) => {
        if (!prev) return prev
        const msg: ChatMessage = {
          id: nextId(),
          role: 'user',
          content: text,
          timestamp: Date.now(),
        }
        return { ...prev, chatMessages: [...prev.chatMessages, msg] }
      })
      ws.sendMessage({ type: 'user_reply', content: text })
    },
    [ws],
  )

  const startDiagnosis = useCallback(async () => {
    if (!session?.id) return

    setSession((prev) => (prev ? { ...prev, status: 'debating' as const } : prev))

    try {
      const resp = await fetch(`/api/diagnose/${session.id}/start`, {
        method: 'POST',
      })

      if (!resp.ok) throw new Error(`Start failed: ${resp.status}`)

      const result = await resp.json()

      if (result.final_diagnosis) {
        setSession((prev) =>
          prev ? { ...prev, status: 'complete' as const, result } : prev,
        )
      }
      // Otherwise, results come via WebSocket
    } catch {
      // Stay in debating — WebSocket handles streaming
    }
  }, [session?.id])

  return { session, uploadImage, addChatImage, sendChatMessage, startDiagnosis }
}
