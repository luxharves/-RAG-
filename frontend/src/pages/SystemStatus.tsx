import { useEffect, useState } from "react";
import { fetchHealth, fetchSystem } from "../api/client";
import type { HealthResponse, SystemResponse } from "../api/client";

const STATE_COLOR: Record<string, string> = {
  CLOSED: "var(--color-success)",
  HALF_OPEN: "var(--color-warning)",
  OPEN: "var(--color-error)",
};

export function SystemStatus() {
  const [data, setData] = useState<HealthResponse | null>(null);
  const [sys, setSys] = useState<SystemResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    Promise.all([fetchHealth(), fetchSystem().catch(() => null)])
      .then(([d, s]) => { setData(d); setSys(s); setError(null); })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <StatusBox status="loading" message="正在连接后端服务..." />;
  if (error) return <StatusBox status="error" message={`API 不可用: ${error}`} />;
  if (!data) return <StatusBox status="error" message="未收到响应" />;

  const cache = sys?.cache;
  const grounding = sys?.grounding;

  return (
    <div>
      <h1 className="mb-4">系统状态</h1>
      <StatusBox status="ok" message="所有服务正常运行" />

      <div className="grid-2" style={{ marginTop: 24 }}>
        <InfoCard label="API 版本" value={data.version} ok />
        <InfoCard label="Milvus" value={data.collection ? "已连接" : "未连接"} ok={!!data.collection} />
        <InfoCard label="BM25 文档数" value={`${data.bm25_docs} docs`} ok={data.bm25_docs > 0} />
        <InfoCard label="Embedding" value={data.models.embedding.split("/").pop() || "?"} ok />
        <InfoCard label="Reranker" value={data.models.reranker.split("/").pop() || "?"} ok />
        <InfoCard label="LLM" value={data.models.llm} ok />
      </div>

      {/* V7 LLM Gateway */}
      {sys && (
        <>
          <h2 className="section-title" style={{ marginTop: 32 }}>V7 LLM Gateway</h2>
          <div style={{ display: "grid", gap: 8 }}>
            {sys.gateway.providers.map((p) => (
              <div key={p.name} className="card card-sm" style={{ display: "flex", alignItems: "center", gap: 16 }}>
                <span className="mono" style={{ minWidth: 110, fontWeight: 600 }}>{p.name}</span>
                <span className="status-dot" style={{ background: STATE_COLOR[p.state] || "var(--color-text-muted)" }} />
                <span style={{ fontWeight: 700 }}>{p.state}</span>
                <span className="muted" style={{ fontSize: "var(--text-sm)" }}>
                  连续失败 {p.consecutive_failures} 次
                  {p.seconds_until_half_open != null ? ` · ${p.seconds_until_half_open}s 后探测` : ""}
                </span>
              </div>
            ))}
          </div>
        </>
      )}

      {/* V9 semantic cache */}
      {cache && (
        <>
          <h2 className="section-title" style={{ marginTop: 32 }}>V9 语义缓存</h2>
          <div className="grid-stat-sm">
            <InfoCard label="缓存条目" value={`${cache.entries}`} ok={cache.entries > 0} />
            <InfoCard label="命中 / 未命中" value={`${cache.hits} / ${cache.miss}`} ok={cache.hits > 0} />
            <InfoCard label="命中率" value={`${(cache.hit_rate * 100).toFixed(1)}%`} ok={cache.hit_rate > 0} />
            <InfoCard label="语义阈值" value={`cos ≥ ${cache.threshold}`} ok />
          </div>
        </>
      )}

      {/* V6 grounding */}
      {grounding && (
        <>
          <h2 className="section-title" style={{ marginTop: 32 }}>V6 接地验证配置</h2>
          <div className="grid-stat-sm">
            <InfoCard label="验证模式" value={grounding.verifier_mode === "grounding" ? "确定性接地" : "LLM-as-judge"} ok />
            <InfoCard label="打分器" value={grounding.scorer === "reranker" ? "交叉编码器" : "BGE-M3 余弦"} ok />
            <InfoCard label="支撑阈值" value={`≥ ${grounding.scorer_floor}`} ok />
            <InfoCard label="支撑率门槛" value={`${(grounding.min_support_ratio * 100).toFixed(0)}%`} ok />
          </div>
        </>
      )}
    </div>
  );
}

function StatusBox({ status, message }: { status: "ok" | "error" | "loading"; message: string }) {
  const icons: Record<string, string> = { ok: "✓", error: "✗", loading: "⟳" };
  return (
    <div className={`status-box status-box-${status}`}>
      <span className="status-box-icon">{icons[status]}</span>
      <span>{message}</span>
    </div>
  );
}

function InfoCard({ label, value, ok }: { label: string; value: string; ok: boolean }) {
  return (
    <div className="card card-sm">
      <div className="stat-label mb-3">{label}</div>
      <div className="flex-row">
        <span className={`status-dot ${ok ? "status-dot-ok" : "status-dot-error"}`} />
        <span style={{ fontSize: "var(--text-md)" }}>{value}</span>
      </div>
    </div>
  );
}
