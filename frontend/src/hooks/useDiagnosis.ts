import { useState, useCallback } from 'react'
import type { DiagnosisSession, DebateResult } from '../types'
import { useWebSocket } from './useWebSocket'

interface UseDiagnosisReturn {
  session: DiagnosisSession | null
  uploadImage: (file: File) => Promise<void>
  startDiagnosis: () => Promise<void>
}

export function useDiagnosis(): UseDiagnosisReturn {
  const [session, setSession] = useState<DiagnosisSession | null>(null)
  const { messages } = useWebSocket(
    session?.status === 'debating' ? session.id : null
  )

  // Sync WebSocket messages into session
  if (session && messages.length > 0 && session.messages.length !== messages.length) {
    setSession((prev) =>
      prev ? { ...prev, messages } : prev
    )
  }

  const uploadImage = useCallback(async (file: File) => {
    setSession((prev) => {
      if (prev) return { ...prev, status: 'uploading' }
      return {
        id: '',
        status: 'uploading',
        images: [],
        messages: [],
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
        status: 'questioning',
        images: [...(prev?.images ?? []), URL.createObjectURL(file)],
        messages: prev?.messages ?? [],
      }))
    } catch (err) {
      console.error('Upload error:', err)
      setSession((prev) =>
        prev ? { ...prev, status: 'questioning' } : prev
      )
    }
  }, [])

  const startDiagnosis = useCallback(async () => {
    if (!session?.id) return

    setSession((prev) =>
      prev ? { ...prev, status: 'debating' } : prev
    )

    try {
      const resp = await fetch(`/api/diagnose/${session.id}/start`, {
        method: 'POST',
      })

      if (!resp.ok) throw new Error(`Start failed: ${resp.status}`)

      const result: DebateResult = await resp.json()
      setSession((prev) =>
        prev ? { ...prev, status: 'complete', result } : prev
      )
    } catch (err) {
      console.error('Diagnosis error:', err)
      // Stay in debating state — WebSocket will handle streaming
    }
  }, [session?.id])

  return { session, uploadImage, startDiagnosis }
}
