import { useState } from "react";
import { fetchQuery } from "../api/client";
import type { QueryResponse } from "../api/client";

const EXAMPLES = [
  "设备无法开机怎么办？",
  "机器人会不会从楼梯摔下去？",
  "如何清洁集尘盒？",
];

const STATUS_CONFIG: Record<string, { label: string; cls: string }> = {
  answered: { label: "已通过验证", cls: "badge-success" },
  refused: { label: "已拒答", cls: "badge-warning" },
  fallback: { label: "降级回答", cls: "badge-warning" },
};

const TYPE_ICONS: Record<string, string> = {
  text: "📝",
  image: "🖼️",
  table: "📊",
};

export function QAPanel() {
  const [question, setQuestion] = useState("");
  const [experiment, setExperiment] = useState("v4");
  const [sourceDocument, setSourceDocument] = useState(""); // "" = 全部说明书
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);

  const doSubmit = async (q: string) => {
    if (!q.trim()) return;
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const data = await fetchQuery({
        question: q,
        experiment,
        source_document: sourceDocument || undefined,
      });
      setResult(data);
    } catch (e: any) {
      setError(e.message || "请求失败");
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      doSubmit(question);
    }
  };

  const statusCfg = result ? (STATUS_CONFIG[result.final_status] || { label: result.final_status, cls: "badge-neutral" }) : null;

  return (
    <div>
      <h1 className="mb-3">维保问答</h1>
      <p className="section-subtitle">
        LangGraph 可信问答：Hybrid Retrieval → Reranker → 生成 → V6 句级接地验证 → 引用答案（V9 语义缓存加速）
      </p>

      {/* Input area */}
      <div className="card mb-4">
        <textarea
          className="textarea"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="输入问题，例如：设备无法开机怎么办？"
          rows={3}
        />
        <div className="flex-row mt-3" style={{ flexWrap: "wrap" }}>
          <button
            className="btn btn-primary"
            onClick={() => doSubmit(question)}
            disabled={loading || !question.trim()}
          >
            {loading ? "查询中..." : "提交问题"}
          </button>
          <select
            className="select"
            value={sourceDocument}
            onChange={(e) => setSourceDocument(e.target.value)}
            title="限定检索到单本说明书（V8 doc_filter）"
          >
            <option value="">📚 全部说明书</option>
            <option value="Roborock G10S">Roborock G10S</option>
            <option value="Ecovacs DEEBOT T30C">Ecovacs DEEBOT T30C</option>
          </select>
          <select
            className="select"
            value={experiment}
            onChange={(e) => setExperiment(e.target.value)}
          >
            <option value="v4">v4 (LangGraph)</option>
            <option value="v3">v3 (Reranker)</option>
            <option value="v2">v2 (Hybrid)</option>
          </select>
        </div>
      </div>

      {/* Examples */}
      {!result && !loading && (
        <div className="mb-4">
          <div className="muted mb-3" style={{ fontSize: "var(--text-sm)" }}>示例问题：</div>
          <div className="tag-row">
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                className="btn btn-ghost btn-sm"
                onClick={() => { setQuestion(ex); doSubmit(ex); }}
              >
                {ex}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="error-box mb-4">
          <strong>请求失败：</strong>{error}
          <div style={{ fontSize: "var(--text-sm)", marginTop: 4 }}>请确认后端服务已启动：uvicorn main:app --host 127.0.0.1 --port 8000</div>
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="empty-state" style={{ padding: 40 }}>
          <div style={{ fontSize: 32, marginBottom: 8 }}>⏳</div>
          <div>正在检索并生成答案，请稍候...</div>
          <div className="muted mt-1" style={{ fontSize: "var(--text-xs)" }}>通常需要 20–40 秒</div>
        </div>
      )}

      {/* Result */}
      {result && (
        <div>
          {/* Status + timing */}
          <div className="flex-between mb-4">
            <span className={`badge ${statusCfg?.cls || "badge-neutral"}`}>
              {statusCfg?.label || result.final_status}
            </span>
            <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
              {result.cache_hit && (
                <span className="badge badge-success">⚡ 缓存命中 {result.cache_source === "semantic" ? "(语义)" : ""}</span>
              )}
              <span className="muted" style={{ fontSize: "var(--text-sm)" }}>
                耗时 {result.timing_s}s{result.cache_hit ? "（缓存，~50ms 级）" : ""}
              </span>
            </div>
          </div>

          {/* Answer */}
          <div className="card mb-3" style={{ lineHeight: "var(--leading-relaxed)", whiteSpace: "pre-wrap", fontSize: "var(--text-md)" }}>
            {result.answer}
          </div>

          {/* Verification */}
          <div className="card card-sm mb-4">
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: "var(--text-md)" }}>验证结果</div>
            <div className="flex-row" style={{ gap: 24, flexWrap: "wrap" }}>
              <div>
                <span className="muted">supported: </span>
                <span style={{
                  fontWeight: 600,
                  color: result.verification.supported ? "var(--color-success)" : "var(--color-error)",
                }}>
                  {String(result.verification.supported)}
                </span>
              </div>
              <div>
                <span className="muted">confidence: </span>
                <span style={{ fontWeight: 600 }}>{(result.verification.confidence * 100).toFixed(0)}%</span>
              </div>
              {result.verification.grounding_meta && (
                <div>
                  <span className="muted">句级支撑: </span>
                  <span style={{ fontWeight: 600 }}>
                    {result.verification.grounding_meta.n_supported ?? 0}/{result.verification.grounding_meta.n_sentences ?? 0} 句
                    <span className="muted ml-2" style={{ fontSize: "var(--text-xs)" }}>
                      {result.verification.grounding_meta.scorer === "cross_encoder" ? "交叉编码器" : "余弦"}
                    </span>
                  </span>
                </div>
              )}
            </div>
            {result.verification.reason && (
              <div className="mt-3" style={{ color: "var(--color-text-secondary)", fontSize: "var(--text-sm)" }}>
                {result.verification.reason}
              </div>
            )}
            {result.verification.unsupported_claims && result.verification.unsupported_claims.length > 0 && (
              <div className="mt-3">
                <div className="muted" style={{ fontSize: "var(--text-sm)", marginBottom: 4 }}>
                  无支撑句子（{result.verification.unsupported_claims.length}）：
                </div>
                {result.verification.unsupported_claims.map((c, i) => (
                  <div key={i} style={{ color: "var(--color-error)", fontSize: "var(--text-sm)" }}>• {c}</div>
                ))}
              </div>
            )}
          </div>

          {/* Sources */}
          <div style={{ fontWeight: 600, marginBottom: 8, fontSize: "var(--text-md)" }}>
            来源 ({result.sources.length})
          </div>
          <div style={{ display: "grid", gap: 8 }}>
            {result.sources.map((s, i) => (
              <div key={i} className="card card-sm" style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
                <span style={{ fontSize: 22 }}>{TYPE_ICONS[s.content_type] || "📄"}</span>
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: 600, fontSize: "var(--text-base)" }}>
                    {s.source_file || "unknown"}
                    <span className="ml-2 muted" style={{ fontSize: "var(--text-xs)" }}>
                      第{s.page_number}页
                    </span>
                  </div>
                  <div className="flex-row mt-1" style={{ gap: 16, fontSize: "var(--text-xs)", color: "var(--color-text-muted)" }}>
                    <span>类型: {s.content_type}</span>
                    <span className="mono">ID: {s.chunk_id.slice(0, 12)}...</span>
                    {s.rerank_score != null && (
                      <span>rerank: {s.rerank_score.toFixed(4)}</span>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
