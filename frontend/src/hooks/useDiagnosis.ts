import { useState, useCallback, useEffect, useRef } from 'react'
import type {
  DiagnosisSession,
  ChatMessage,
  CreateDiagnosisResponse,
  UploadImageResponse,
} from '../types'
import { useWebSocket } from './useWebSocket'

/* ── ID generator ───────────────────────────────────────── */

let chatIdSeq = 0
function nextId(): string {
  return `diag-${Date.now()}-${++chatIdSeq}`
}

/* ── Public API ─────────────────────────────────────────── */

interface UseDiagnosisReturn {
  session: DiagnosisSession | null
  /** Upload initial image(s) to create a diagnosis session */
  uploadImage: (file: File, context?: string) => Promise<void>
  /** Upload an additional image during AVD questioning */
  addChatImage: (file: File) => Promise<void>
  /** Send a text chat message (unused for now but keeps the chat panel wired) */
  sendChatMessage: (text: string) => void
  /** Skip AVD and jump directly to DDP debate */
  skipToDebate: () => void
}

/* ── Hook ───────────────────────────────────────────────── */

export function useDiagnosis(): UseDiagnosisReturn {
  const [session, setSession] = useState<DiagnosisSession | null>(null)
  const [apiKey, setApiKey] = useState<string>('')

  // --- Determine when to connect the WebSocket ---
  const wsSessionId = session?.id || null

  const ws = useWebSocket(wsSessionId, apiKey)

  // --- Sync WS state into session via effects (not during render) ---

  const prevAgentLen = useRef(0)
  const prevChatLen = useRef(0)

  useEffect(() => {
    if (!session) return
    if (ws.agentMessages.length > prevAgentLen.current) {
      prevAgentLen.current = ws.agentMessages.length
      setSession((prev) =>
        prev ? { ...prev, messages: ws.agentMessages } : prev,
      )
    }
  }, [ws.agentMessages, session])

  useEffect(() => {
    if (!session) return
    if (ws.chatMessages.length > prevChatLen.current) {
      prevChatLen.current = ws.chatMessages.length
      setSession((prev) =>
        prev ? { ...prev, chatMessages: [...(prev.chatMessages ?? []), ...ws.chatMessages.slice(prev.chatMessages.length)] } : prev,
      )
    }
  }, [ws.chatMessages, session])

  // Transition to "questioning" when WS connects and AVD hasn't finished
  useEffect(() => {
    if (!session) return
    if (ws.isConnected && session.status === 'uploading') {
      setSession((prev) =>
        prev ? { ...prev, status: 'questioning' } : prev,
      )
    }
  }, [ws.isConnected, session])

  // Transition to "debating" when server signals
  useEffect(() => {
    if (!session) return
    if (ws.isDebating && session.status !== 'debating') {
      setSession((prev) =>
        prev ? { ...prev, status: 'debating' } : prev,
      )
    }
  }, [ws.isDebating, session])

  // Transition to "complete" when result arrives
  useEffect(() => {
    if (!session) return
    if (ws.result && !session.result) {
      setSession((prev) =>
        prev ? { ...prev, status: 'complete', result: ws.result! } : prev,
      )
    }
  }, [ws.result, session])

  // Surface WS errors
  useEffect(() => {
    if (!session) return
    if (ws.error) {
      setSession((prev) =>
        prev ? { ...prev, error: ws.error ?? undefined } : prev,
      )
    }
  }, [ws.error, session])

  // --- Actions ---

  /**
   * Upload the initial image and create a diagnosis session.
   * The backend expects FormData with field name `files` (list[UploadFile]).
   */
  const uploadImage = useCallback(async (file: File, context?: string) => {
    // Reset ref counters for a new session
    prevAgentLen.current = 0
    prevChatLen.current = 0

    setSession({
      id: '',
      status: 'uploading',
      images: [],
      messages: [],
      chatMessages: [],
    })

    const formData = new FormData()
    formData.append('files', file)
    if (context) {
      formData.append('context', context)
    }

    try {
      const resp = await fetch('/api/diagnose', {
        method: 'POST',
        body: formData,
      })

      if (!resp.ok) {
        const text = await resp.text()
        throw new Error(`Upload failed (${resp.status}): ${text}`)
      }

      const data: CreateDiagnosisResponse = await resp.json()

      setSession((prev) => ({
        id: data.session_id,
        status: 'uploading', // will transition to 'questioning' when WS connects
        images: [URL.createObjectURL(file)],
        messages: prev?.messages ?? [],
        chatMessages: prev?.chatMessages ?? [],
      }))
    } catch (err) {
      const message = err instanceof Error ? err.message : '上传失败'
      console.error('Upload error:', err)
      setSession((prev) =>
        prev
          ? { ...prev, status: 'error', error: message }
          : { id: '', status: 'error', images: [], messages: [], chatMessages: [], error: message },
      )
    }
  }, [])

  /**
   * Upload an additional image mid-conversation (during AVD).
   * Posts to POST /api/diagnose/{session_id}/upload, then tells the WS
   * to continue with `{ action: "continue" }`.
   */
  const addChatImage = useCallback(async (file: File) => {
    if (!session?.id) return

    const localUrl = URL.createObjectURL(file)

    // Optimistically add to chat + images
    const userMsg: ChatMessage = {
      id: nextId(),
      role: 'user',
      content: '',
      imageUrl: localUrl,
      timestamp: Date.now(),
    }
    setSession((prev) => {
      if (!prev) return prev
      return {
        ...prev,
        images: [...prev.images, localUrl],
        chatMessages: [...prev.chatMessages, userMsg],
      }
    })

    try {
      const formData = new FormData()
      formData.append('file', file)

      const resp = await fetch(`/api/diagnose/${session.id}/upload`, {
        method: 'POST',
        body: formData,
      })

      if (!resp.ok) {
        throw new Error(`Image upload failed: ${resp.status}`)
      }

      await resp.json() as UploadImageResponse

      // Tell the WS to re-assess with the new image
      ws.sendMessage({ action: 'continue' })
    } catch (err) {
      console.error('Additional image upload error:', err)
      const errMsg: ChatMessage = {
        id: nextId(),
        role: 'system',
        content: '图片上传失败，请重试。',
        timestamp: Date.now(),
      }
      setSession((prev) =>
        prev
          ? { ...prev, chatMessages: [...prev.chatMessages, errMsg] }
          : prev,
      )
    }
  }, [session?.id, ws])

  /**
   * Send a text message in the chat panel.
   * If the user types "skip" / "没有了" / "跳过", we treat it as a skip action.
   */
  const sendChatMessage = useCallback(
    (text: string) => {
      if (!text.trim()) return

      const userMsg: ChatMessage = {
        id: nextId(),
        role: 'user',
        content: text,
        timestamp: Date.now(),
      }
      setSession((prev) =>
        prev
          ? { ...prev, chatMessages: [...prev.chatMessages, userMsg] }
          : prev,
      )

      // Send answer to AVD interview, or skip if user says so
      const lower = text.trim()
      if (['skip', '跳过', '没有了', '不需要了', '直接诊断'].includes(lower)) {
        ws.sendMessage({ action: 'skip' })
      } else {
        ws.sendMessage({ action: 'continue', answer: text })
      }
    },
    [ws],
  )

  /**
   * Skip the AVD questioning phase and proceed directly to DDP debate.
   * Sends `{ action: "skip" }` over the WebSocket.
   */
  const skipToDebate = useCallback(() => {
    ws.sendMessage({ action: 'skip' })

    const skipMsg: ChatMessage = {
      id: nextId(),
      role: 'user',
      content: '跳过问诊，直接开始辩论诊断',
      timestamp: Date.now(),
    }
    setSession((prev) =>
      prev
        ? { ...prev, chatMessages: [...prev.chatMessages, skipMsg] }
        : prev,
    )
  }, [ws])

  return { session, apiKey, setApiKey, uploadImage, addChatImage, sendChatMessage, skipToDebate }
}
