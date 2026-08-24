export function Dashboard() {
  const tags = ["BGE-M3", "BM25", "BGE-Reranker", "LangGraph", "Milvus", "RRF Fusion", "Qwen3-VL", "FastAPI"];

  return (
    <div>
      <h1>多模态 RAG 智能硬件维保助手</h1>
      <p className="section-subtitle">
        基于 BGE-M3 Embedding + BM25 Hybrid Retrieval + BGE-Reranker + LangGraph 验证的可信问答系统
      </p>

      <div className="tag-row mb-4" style={{ marginBottom: 32 }}>
        {tags.map((t) => (
          <span key={t} className="badge badge-accent">{t}</span>
        ))}
      </div>

      <div className="grid-2 mb-4">
        <StatCard title="系统版本" value="V9（V0–V9 演进）" />
        <StatCard title="检索通道" value="Dense + BM25 + RRF + Rerank" />
        <StatCard title="精排模型" value="BGE-Reranker-v2-m3" />
        <StatCard title="验证引擎" value="确定性句级接地" />
        <StatCard title="多模态" value="文本 + 图片 + 表格" />
        <StatCard title="Golden Dataset" value="100 条 + 123 条扩展" />
      </div>

      <h2 className="section-title" style={{ marginTop: 40 }}>核心能力</h2>
      <div className="grid-3">
        <CapCard title="多模态 RAG" desc="文本 + 图片语义描述 + 表格结构化的统一向量检索" />
        <CapCard title="Hybrid Retrieval" desc="Dense + BM25 双通道 + RRF 融合，MRR 提升 15%" />
        <CapCard title="可信问答" desc="LangGraph Verify Node + 证据链追踪 + 无证据拒答" />
        <CapCard title="增量更新" desc="SHA256 Hash 检测 + added/unchanged/modified/deleted 分类" />
      </div>
    </div>
  );
}

function StatCard({ title, value }: { title: string; value: string }) {
  return (
    <div className="card card-stat">
      <div className="stat-label">{title}</div>
      <div className="stat-value stat-value-accent">{value}</div>
    </div>
  );
}

function CapCard({ title, desc }: { title: string; desc: string }) {
  return (
    <div className="card">
      <div className="cap-title">{title}</div>
      <div className="cap-desc">{desc}</div>
    </div>
  );
}
