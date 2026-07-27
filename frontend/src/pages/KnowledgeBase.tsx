import { useEffect, useRef, useState } from "react";
import { uploadDocument, fetchDocuments } from "../api/client";
import type { IngestResponse, KnowledgeDocument } from "../api/client";

export function KnowledgeBase() {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<IngestResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [docs, setDocs] = useState<KnowledgeDocument[]>([]);
  const [docsLoading, setDocsLoading] = useState(true);
  const [docsError, setDocsError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const loadDocs = async () => {
    setDocsLoading(true);
    try {
      const data = await fetchDocuments();
      setDocs(data.documents);
      setDocsError(null);
    } catch (e: any) {
      setDocsError(e.message || "加载失败");
    } finally {
      setDocsLoading(false);
    }
  };

  useEffect(() => { loadDocs(); }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    if (!f.name.toLowerCase().endsWith(".pdf")) {
      setError("上传失败：只支持 PDF 文件");
      setFile(null);
      return;
    }
    setFile(f);
    setError(null);
    setResult(null);
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);
    setResult(null);
    try {
      const data = await uploadDocument(file);
      setResult(data);
      await loadDocs();
    } catch (e: any) {
      setError(e.message || "上传失败：后端服务暂时不可用");
    } finally {
      setUploading(false);
    }
  };

  const formatSize = (bytes: number) =>
    bytes < 1024 ? `${bytes} B` : bytes < 1048576 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1048576).toFixed(1)} MB`;

  const wasReused = (result?.reused_chunks ?? 0) > 0 && (result?.embedded_chunks ?? 0) === 0;

  const statCards = result
    ? [
        { label: "新增文档", value: result.added ?? 0 },
        { label: "未变化文档", value: result.unchanged ?? 0 },
        { label: "修改文档", value: result.modified ?? 0 },
        { label: "删除文档", value: result.deleted ?? 0 },
        { label: "重新处理页数", value: result.reprocessed_pages ?? 0 },
        { label: "复用 Chunk", value: result.reused_chunks ?? 0, highlight: (result.reused_chunks ?? 0) > 0 },
        { label: "重新 Embedding", value: result.embedded_chunks ?? 0 },
        { label: "删除 Chunk", value: result.removed_chunks ?? 0 },
      ]
    : [];

  return (
    <div>
      <h1 className="mb-3">知识库管理</h1>
      <p className="section-subtitle">
        上传 PDF 说明书，自动解析、切片、Embedding 并写入 Milvus + BM25 索引
      </p>

      {/* Upload area */}
      <div className="card mb-4">
        <input ref={fileRef} type="file" accept=".pdf" onChange={handleFileChange} style={{ display: "none" }} />
        <div className="flex-row flex-wrap">
          <button
            className="btn btn-secondary"
            onClick={() => fileRef.current?.click()}
            disabled={uploading}
            style={{ color: "var(--color-info)" }}
          >
            选择 PDF 文件
          </button>
          {file && (
            <button
              className="btn btn-primary"
              onClick={handleUpload}
              disabled={uploading || !file}
            >
              {uploading ? "上传中..." : "上传到知识库"}
            </button>
          )}
        </div>
        {file && !uploading && (
          <div className="mt-3" style={{ fontSize: "var(--text-base)", color: "var(--color-text)" }}>
            已选择：{file.name} ({formatSize(file.size)})
          </div>
        )}
      </div>

      {/* Error */}
      {error && <div className="error-box mb-4">{error}</div>}

      {/* Uploading indicator */}
      {uploading && (
        <div className="empty-state mb-4" style={{ padding: 24 }}>
          <div style={{ fontSize: 28, marginBottom: 8 }}>📤</div>
          <div>正在上传并处理 PDF，请稍候...</div>
        </div>
      )}

      {/* Success result */}
      {result && (
        <div className="mb-4">
          <div
            className="card card-sm mb-3"
            style={{
              background: wasReused ? "var(--color-warning-soft)" : "var(--color-success-soft)",
              borderColor: wasReused ? "oklch(65% 0.16 85 / 0.3)" : "oklch(55% 0.16 160 / 0.3)",
            }}
          >
            <div style={{ fontWeight: 600, fontSize: "var(--text-md)", marginBottom: 4 }}>
              {wasReused ? "文档未发生变化，已复用已有 Chunk" : "新文档已加入知识库"}
            </div>
            <div className="muted" style={{ fontSize: "var(--text-sm)" }}>
              文件名：{file?.name} · Document ID：{result.document_id}
            </div>
            <div className="muted" style={{ fontSize: "var(--text-sm)" }}>
              版本：{result.version} · 处理页数：{result.pages} · Chunk 数：{result.chunks}
            </div>
            <div className="muted" style={{ fontSize: "var(--text-sm)" }}>
              状态：{result.status}
            </div>
          </div>

          <div className="grid-stat-sm">
            {statCards.map((s) => (
              <div
                key={s.label}
                className="card card-sm"
                style={{
                  textAlign: "center",
                  border: s.highlight ? "2px solid var(--color-success)" : undefined,
                }}
              >
                <div className="stat-label">{s.label}</div>
                <div style={{
                  fontSize: 22, fontWeight: 700,
                  color: s.highlight ? "var(--color-success)" : "var(--color-text)",
                }}>
                  {s.value}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Document list */}
      <h2 className="section-title">当前文档列表</h2>
      {docsLoading && <div className="empty-state" style={{ padding: "var(--space-3)" }}>加载中...</div>}
      {docsError && <div className="error-box mb-4">加载失败：{docsError}</div>}
      {!docsLoading && !docsError && docs.length === 0 && (
        <div className="empty-state">暂无文档，请上传说明书 PDF</div>
      )}
      {docs.length > 0 && (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>文件名</th>
                <th>Document ID</th>
                <th>版本</th>
                <th>Chunk 数</th>
                <th>状态</th>
              </tr>
            </thead>
            <tbody>
              {docs.map((d) => (
                <tr key={d.document_id}>
                  <td>{d.source_file}</td>
                  <td className="mono muted">{d.document_id}</td>
                  <td className="mono muted">{d.version}</td>
                  <td>{d.num_chunks}</td>
                  <td>
                    <span className="badge badge-success">{d.status}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
