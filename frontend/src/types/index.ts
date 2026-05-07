export type Confidence = 'high' | 'medium' | 'low'
export type AgentRole = 'proposer' | 'challenger' | 'arbiter'

export interface AgentMessage {
  role: AgentRole
  content: string
  round: number
}

export interface RejectedDiagnosis {
  name: string
  reason: string
}

export interface GroundingBox {
  x: number
  y: number
  width: number
  height: number
  label: string
}

export interface DebateResult {
  final_diagnosis: string
  confidence: Confidence
  supporting_evidence: string[]
  rejected_diagnoses: RejectedDiagnosis[]
  uncertainty_notes: string[]
  debate_transcript: AgentMessage[]
  grounding_boxes: GroundingBox[]
  treatment?: TreatmentInfo
  similar_cases?: SimilarCase[]
}

export interface DiagnosisSession {
  id: string
  status: 'idle' | 'uploading' | 'questioning' | 'debating' | 'complete'
  images: string[]
  messages: AgentMessage[]
  chatMessages: ChatMessage[]
  result?: DebateResult
}

// --- AVD types ---

export interface AVDQuestion {
  question: string
  reason: string
  target_part: string
}

export interface AVDAssessment {
  status: 'questioning' | 'sufficient' | 'forced'
  confidence: number
  question?: AVDQuestion
  summary: string
}

export interface SimilarCase {
  image_url: string
  label: string
  similarity: number
  source: string
}

export interface TreatmentInfo {
  text: string
  source: string
}

// --- Chat message (unified for AVD panel) ---

export type ChatMessageRole = 'system' | 'user' | 'assistant'

export interface ChatMessage {
  id: string
  role: ChatMessageRole
  content: string
  imageUrl?: string
  timestamp: number
}

// --- WebSocket ---

export type WSMessageType =
  | 'status'
  | 'agent_message'
  | 'avd_question'
  | 'avd_sufficient'
  | 'result'
  | 'error'

export interface WSMessage {
  type: WSMessageType
  data: unknown
}
