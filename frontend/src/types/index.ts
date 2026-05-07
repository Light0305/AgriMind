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
}

export interface DiagnosisSession {
  id: string
  status: 'uploading' | 'questioning' | 'debating' | 'complete'
  images: string[]
  messages: AgentMessage[]
  result?: DebateResult
}
