const BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

export interface HealthResponse {
  status: string;
  version: string;
  collection: string;
  bm25_docs: number;
  models: { embedding: string; reranker: string; llm: string };
  milvus: string;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

export interface QueryRequest {
  question: string;
  experiment: string;
  source_document?: string | null; // friendly name, e.g. "Roborock G10S" (V8)
}

export interface Source {
  chunk_id: string;
  source_file: string;
  page_number: number;
  content_type: string;
  rerank_score: number | null;
}

export interface QueryResponse {
  question: string;
  answer: string;
  final_status: string;
  experiment: string;
  sources: Source[];
  evidence_chunk_ids: string[];
  verification: {
    supported: boolean;
    confidence: number;
    reason: string;
    unsupported_claims?: string[];
    grounding_meta?: {
      scorer?: string;
      scorer_floor?: number;
      support_ratio?: number;
      n_sentences?: number;
      n_supported?: number;
      n_skipped_short?: number;
      min_support_ratio?: number;
      initial_threshold?: number;
      threshold_floor?: number;
      decay?: number;
    } | null;
    sentence_evidence?: Array<{
      sentence: string;
      supported: boolean;
      best_similarity: number | null;
      effective_threshold: number | null;
      status: string;
    }> | null;
  };
  timing_s: number;
  cache_hit?: boolean;
  cache_source?: "exact" | "semantic" | null;
}

export async function fetchQuery(req: QueryRequest): Promise<QueryResponse> {
  const res = await fetch(`${BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`Query failed: ${res.status}`);
  return res.json();
}

export interface IngestResponse {
  document_id: string;
  version: string;
  chunks: number;
  pages: number;
  status: string;
  added?: number;
  unchanged?: number;
  modified?: number;
  deleted?: number;
  reprocessed_pages?: number;
  reused_chunks?: number;
  embedded_chunks?: number;
  removed_chunks?: number;
}

export async function uploadDocument(file: File): Promise<IngestResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/documents/ingest`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error((detail as any).detail || `Upload failed: ${res.status}`);
  }
  return res.json();
}

export interface KnowledgeDocument {
  document_id: string;
  source_file: string;
  version: string;
  num_chunks: number;
  status: string;
}

export async function fetchDocuments(): Promise<{ documents: KnowledgeDocument[] }> {
  const res = await fetch(`${BASE}/documents`);
  if (!res.ok) throw new Error(`Failed to list documents: ${res.status}`);
  return res.json();
}

export interface ExperimentListItem {
  id: string;
  name: string;
  description: string;
  key_metric: string;
  available: boolean;
  files: string[];
}

export interface ExperimentMetrics {
  hit_rate?: number | null;
  recall_at_5?: number | null;
  mrr?: number | null;
  faithfulness?: number | null;
  context_precision?: number | null;
  context_recall?: number | null;
  answer_relevancy?: number | null;
  top5_hit_rate?: number | null;
  top1_hit_rate?: number | null;
  [key: string]: unknown;
}

export interface ExperimentSummary {
  id: string;
  metadata: Record<string, unknown>;
  metrics: ExperimentMetrics;
  incremental_metrics?: Record<string, number>;
  files: string[];
}

export async function fetchExperiments(): Promise<{ experiments: ExperimentListItem[] }> {
  const res = await fetch(`${BASE}/experiments`);
  if (!res.ok) throw new Error(`Failed: ${res.status}`);
  return res.json();
}

export async function fetchExperiment(id: string): Promise<ExperimentSummary> {
  const res = await fetch(`${BASE}/experiments/${id}`);
  if (!res.ok) throw new Error(`Failed: ${res.status}`);
  return res.json();
}

export interface FinalEvalMetrics {
  run_id: string;
  timestamp: string;
  total_questions: number;
  ragas_version: string;
  versions: Record<string, {
    version: string;
    retrieval_metrics: {
      evaluated_question_count: number;
      hit_at_5: number;
      recall_at_5: number;
      mrr: number;
      top1_hit_rate: number;
      avg_retrieval_latency_s: number;
      avg_generation_latency_s: number;
    };
    ragas_metrics: {
      evaluated_question_count: number;
      faithfulness: number;
      context_precision: number;
      context_recall: number;
      answer_relevancy: number;
      avg_ragas_latency_s: number;
      valid_sample_counts: Record<string, number>;
    };
  }>;
  v5_incremental_metrics: Record<string, number>;
}

export async function fetchFinalEval(): Promise<FinalEvalMetrics> {
  const res = await fetch(`${BASE}/final_eval`);
  if (!res.ok) throw new Error(`Final eval not available: ${res.status}`);
  return res.json();
}

export async function fetchEvaluation(id: string): Promise<Record<string, unknown>> {
  const res = await fetch(`${BASE}/evaluate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ experiment: id }),
  });
  if (!res.ok) throw new Error(`Failed: ${res.status}`);
  return res.json();
}

// ── V6–V9 highlights (GET /versions) ──

export interface VersionHighlights {
  version: string;
  v6_grounding: {
    total: number; answered: number; avg_support_ratio: number;
    poison_flagged: number; poison_total: number; poison_rate: number;
    over_refused: number; retries_used: number;
  };
  v7_gateway: {
    timeout_s: number; max_retries: number; circuit_threshold: number;
    cooldown_s: number; configured_providers: number;
  };
  v8_multidoc: {
    questions: number; text: number; image: number;
    v3_hit_at_5: number; v3_mrr: number; v3_top1: number; ecovacs_mrr: number;
  };
  v9_cache: {
    warmed: number; exact_hits: number; semantic_hits: number;
    overall_hit_rate: number; avg_cached_s: number; avg_uncached_s: number; llm_calls_saved: number;
  };
}

export async function fetchVersions(): Promise<VersionHighlights> {
  const res = await fetch(`${BASE}/versions`);
  if (!res.ok) throw new Error(`Failed: ${res.status}`);
  return res.json();
}

// ── Live system state (GET /system) ──

export interface SystemResponse {
  version: string;
  gateway: {
    providers: Array<{
      name: string; state: string; consecutive_failures: number;
      opened_at: number | null; seconds_until_half_open: number | null;
    }>;
  };
  cache: {
    entries: number; hits: number; miss: number; hit_rate: number; threshold: number;
  } | null;
  grounding: {
    verifier_mode: string; scorer: string; scorer_floor: number; min_support_ratio: number;
  };
}

export async function fetchSystem(): Promise<SystemResponse> {
  const res = await fetch(`${BASE}/system`);
  if (!res.ok) throw new Error(`Failed: ${res.status}`);
  return res.json();
}
