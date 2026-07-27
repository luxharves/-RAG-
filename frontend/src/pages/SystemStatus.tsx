import { useEffect, useState } from "react";
import { fetchHealth } from "../api/client";
import type { HealthResponse } from "../api/client";

export function SystemStatus() {
  const [data, setData] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetchHealth()
      .then((d) => { setData(d); setError(null); })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <StatusBox status="loading" message="正在连接后端服务..." />;
  if (error) return <StatusBox status="error" message={`API 不可用: ${error}`} />;
  if (!data) return <StatusBox status="error" message="未收到响应" />;

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
