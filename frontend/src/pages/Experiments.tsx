import { useEffect, useState } from "react";
import { fetchExperiments, fetchExperiment, fetchHealth, fetchFinalEval, fetchVersions } from "../api/client";
import type { ExperimentListItem, ExperimentMetrics, ExperimentSummary, FinalEvalMetrics, VersionHighlights } from "../api/client";

const VERSION_NAMES: Record<string, string> = {
  v0_baseline: "V0 Baseline", v1_multimodal: "V1 Multimodal",
  v2_comparison: "V2 Hybrid", v3_rerank: "V3 Reranker",
  v4_verified: "V4 Verify", v5_incremental: "V5 Incremental",
  v6_grounding: "V6 Grounding", v8_multidoc: "V8 Multi-doc",
  v9_cache: "V9 Cache",
};

// Generic RAGAS metrics table only compares V0–V5 (the RAGAS-comparable set);
// V6/V8/V9 have their own dedicated sections below.
const RAGAS_ORDER = ["v0_baseline", "v1_multimodal", "v2_comparison", "v3_rerank", "v4_verified", "v5_incremental"];

const FINAL_EVAL_MAP: Record<string, string> = {
  V0: "v0_baseline", V1: "v1_multimodal", V2: "v2_comparison",
  V3: "v3_rerank", V4: "v4_verified",
};

const EVOLUTION_STEPS = [
  { v: "V0", title: "文本 + Dense Retrieval", desc: "PDF 文本解析、BGE-M3 Embedding、Milvus 密集检索、LLM 生成引用答案", icon: "📄" },
  { v: "V1", title: "+ 多模态解析", desc: "PDF 图片渲染、Qwen3-VL 语义描述、表格结构化、image/table Chunk", icon: "🖼️" },
  { v: "V2", title: "+ Hybrid Retrieval", desc: "BM25 关键词检索 + Dense + RRF 排名融合、Q19 从 MISS→HIT", icon: "🔀" },
  { v: "V3", title: "+ BGE-Reranker", desc: "二阶段精排、Top-1 +5%、MRR +1.9%、20/20 排序变化", icon: "🎯" },
  { v: "V4", title: "+ LangGraph Verify", desc: "验证节点 + 证据链追踪 + 越界拒答 + 重试机制", icon: "✅" },
  { v: "V5", title: "+ 增量更新", desc: "SHA256 Hash 检测、unchanged 0 Embedding、BM25 增量", icon: "📦" },
  { v: "V6", title: "+ 确定性接地", desc: "句级交叉编码器验证（替代 LLM 自查）+ 引用审计，投毒测试拦截 82%", icon: "🛡️" },
  { v: "V7", title: "+ LLM Gateway", desc: "60s 超时 + 指数退避重试 + 断路器 + 多 Provider 故障转移", icon: "🚦" },
  { v: "V8", title: "+ 多文档检索", desc: "双说明书 + doc_filter 元数据隔离，123 题 retrieval-only MRR 0.89", icon: "📚" },
  { v: "V9", title: "+ 语义缓存", desc: "SHA256 精确 + BGE-M3 语义两级缓存，命中 ~31ms 省 LLM 调用", icon: "⚡" },
];

const METRIC_LABELS: Record<string, string> = {
  recall_at_5: "Recall@5", mrr: "MRR", hit_rate: "Hit Rate",
  top5_hit_rate: "Top-5 Hit", top1_hit_rate: "Top-1 Hit",
  faithfulness: "Faithfulness", answer_relevancy: "Answer Relevancy",
  context_precision: "Context Precision", context_recall: "Context Recall",
};

function na(v: unknown): string {
  if (v === null || v === undefined) return "N/A";
  if (typeof v === "number") return v.toFixed(4);
  return String(v);
}

export function Experiments() {
  const [exps, setExps] = useState<ExperimentListItem[]>([]);
  const [metrics, setMetrics] = useState<Record<string, ExperimentMetrics>>({});
  const [summaries, setSummaries] = useState<Record<string, ExperimentSummary>>({});
  const [finalEval, setFinalEval] = useState<FinalEvalMetrics | null>(null);
  const [versions, setVersions] = useState<VersionHighlights | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [banner, setBanner] = useState<string | null>(null);

  useEffect(() => {
    fetchHealth().then(() => setBanner(null)).catch(() => setBanner("后端服务不可用，数据可能不完整"));
  }, []);

  useEffect(() => {
    (async () => {
      setLoading(true);
      try {
        const [expData, finalEvalData, verData] = await Promise.all([
          fetchExperiments().catch(() => ({ experiments: [] })),
          fetchFinalEval().catch(() => null),
          fetchVersions().catch(() => null),
        ]);
        if (verData) setVersions(verData);
        setExps(expData.experiments.filter(e => e.available));
        if (finalEvalData) setFinalEval(finalEvalData);

        const m: Record<string, ExperimentMetrics> = {};
        const s: Record<string, ExperimentSummary> = {};
        for (const exp of expData.experiments) {
          if (!exp.available) continue;
          try {
            const detail = await fetchExperiment(exp.id);
            s[exp.id] = detail;
            if (detail.metrics && typeof detail.metrics === "object") {
              m[exp.id] = detail.metrics as ExperimentMetrics;
            }
          } catch {}
        }

        if (finalEvalData?.versions) {
          for (const [vKey, vData] of Object.entries(finalEvalData.versions)) {
            const expId = FINAL_EVAL_MAP[vKey];
            if (!expId) continue;
            const rm = vData.retrieval_metrics;
            m[expId] = {
              ...m[expId],
              recall_at_5: rm.recall_at_5,
              mrr: rm.mrr,
              hit_rate: rm.hit_at_5,
              top5_hit_rate: rm.hit_at_5,
              top1_hit_rate: rm.top1_hit_rate,
              faithfulness: vData.ragas_metrics.faithfulness,
              context_precision: vData.ragas_metrics.context_precision,
              context_recall: vData.ragas_metrics.context_recall,
              answer_relevancy: vData.ragas_metrics.answer_relevancy,
            };
          }
          if (finalEvalData.v5_incremental_metrics) {
            s["v5_incremental"] = {
              ...s["v5_incremental"], id: "v5_incremental", metadata: {}, metrics: {}, files: [],
              incremental_metrics: finalEvalData.v5_incremental_metrics,
            } as any;
          }
        }
        setMetrics(m); setSummaries(s); setError(null);
      } catch (e: any) { setError(e.message); }
      finally { setLoading(false); }
    })();
  }, []);

  // Generic RAGAS table compares only the V0–V5 set (V6/V8/V9 have dedicated sections)
  const available = new Set(exps.map(e => e.id));
  const displayOrder = RAGAS_ORDER.filter(id => available.has(id));

  let bestRecall = 0, bestMrr = 0, bestFaith = 0, bestCtxRecall = 0;
  let bestRecallV = "", bestMrrV = "", bestFaithV = "", bestCtxRecallV = "";
  for (const vid of displayOrder) {
    const m = metrics[vid];
    if (!m) continue;
    const r = m.recall_at_5 ?? m.top5_hit_rate;
    if (r != null && r > bestRecall) { bestRecall = r; bestRecallV = vid; }
    if (m.mrr != null && m.mrr > bestMrr) { bestMrr = m.mrr; bestMrrV = vid; }
    if (m.faithfulness != null && m.faithfulness > bestFaith) { bestFaith = m.faithfulness; bestFaithV = vid; }
    if (m.context_recall != null && m.context_recall > bestCtxRecall) { bestCtxRecall = m.context_recall; bestCtxRecallV = vid; }
  }

  const v5Summary = summaries["v5_incremental"];
  const incrementalData: Record<string, unknown> = (v5Summary as any)?.incremental_metrics || {};

  return (
    <div>
      <h1 className="mb-3">实验评估</h1>
      <p className="section-subtitle">
        基于固定 100 条 Golden Dataset 的 RAGAS 对比（V0–V4）+ 增量更新（V5）+ 确定性接地（V6）
        + 多文档检索（V8）+ 语义缓存（V9）
      </p>

      {banner && (
        <div className="card card-sm mb-4" style={{ background: "var(--color-warning-soft)" }}>
          {banner}
        </div>
      )}
      {loading && <div className="empty-state" style={{ padding: 40 }}>⏳ 加载实验数据...</div>}
      {error && <div className="error-box mb-4">加载失败：{error}</div>}
      {!loading && !error && displayOrder.length === 0 && (
        <div className="empty-state">暂无实验数据</div>
      )}

      {displayOrder.length > 0 && (
        <>
          {/* Best metrics cards */}
          <div className="grid-2 mb-4" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))" }}>
            <MiniCard label="最佳 Recall@5" value={bestRecall > 0 ? bestRecall.toFixed(4) : "N/A"} version={bestRecallV} />
            <MiniCard label="最佳 MRR" value={bestMrr > 0 ? bestMrr.toFixed(4) : "N/A"} version={bestMrrV} />
            <MiniCard label="最佳 Faithfulness" value={bestFaith > 0 ? bestFaith.toFixed(4) : "N/A"} version={bestFaithV} />
            <MiniCard label="最佳 Context Recall" value={bestCtxRecall > 0 ? bestCtxRecall.toFixed(4) : "N/A"} version={bestCtxRecallV} />
          </div>

          {/* Metrics comparison table */}
          <h2 className="section-title">指标对比</h2>
          <div className="table-wrap mb-4">
            <table style={{ fontSize: "var(--text-sm)" }}>
              <thead>
                <tr>
                  <th>指标</th>
                  {displayOrder.map(vid => (
                    <th key={vid} style={{ textAlign: "center" }}>{VERSION_NAMES[vid] || vid}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {["recall_at_5", "mrr", "hit_rate", "faithfulness", "context_precision", "context_recall", "answer_relevancy"].map(mk => {
                  const vals = displayOrder.map(vid => metrics[vid]?.[mk] ?? metrics[vid]?.[mk === "recall_at_5" ? "top5_hit_rate" : ""]);
                  const best = Math.max(...vals.filter(v => typeof v === "number"));
                  return (
                    <tr key={mk}>
                      <td style={{ fontWeight: 600 }}>{METRIC_LABELS[mk] || mk}</td>
                      {vals.map((v, i) => {
                        const isBest = typeof v === "number" && v === best && best > 0;
                        return (
                          <td key={i} style={{
                            textAlign: "center",
                            background: isBest ? "var(--color-success-soft)" : undefined,
                            fontWeight: isBest ? 700 : 400,
                            color: v === null || v === undefined ? "var(--color-text-muted)" : undefined,
                          }}>{na(v)}</td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* CSS bar chart */}
          <h2 className="section-title">V0–V4 检索指标趋势</h2>
          <div className="card mb-4">
            <div className="muted mb-3" style={{ fontSize: "var(--text-sm)" }}>Recall@5</div>
            <div style={{ display: "flex", alignItems: "flex-end", gap: 16, height: 160 }}>
              {displayOrder.filter(v => v !== "v5_incremental").map(vid => {
                const v = metrics[vid]?.recall_at_5 ?? metrics[vid]?.top5_hit_rate ?? 0;
                const pct = Math.min(100, (typeof v === "number" ? v : 0) * 100);
                return (
                  <div key={vid} style={{ flex: 1, textAlign: "center" }}>
                    <div style={{ fontSize: "var(--text-xs)", fontWeight: 600, marginBottom: 4 }}>{na(v)}</div>
                    <div style={{
                      height: 120, background: "var(--color-surface-hover)", borderRadius: "4px 4px 0 0",
                      position: "relative", overflow: "hidden",
                    }}>
                      <div style={{
                        position: "absolute", bottom: 0, width: "100%", height: `${pct}%`,
                        background: pct > 85 ? "var(--color-success)" : pct > 75 ? "var(--color-warning)" : "var(--color-accent)",
                        borderRadius: "4px 4px 0 0",
                      }} />
                    </div>
                    <div className="muted mt-1" style={{ fontSize: 11 }}>
                      {VERSION_NAMES[vid]?.split(" ")[0] || vid}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Evolution timeline */}
          <h2 className="section-title">技术演进</h2>
          <div style={{ display: "flex", flexDirection: "column", gap: 0, marginBottom: 24 }}>
            {EVOLUTION_STEPS.map((step, i) => (
              <div key={step.v} style={{ display: "flex", gap: 16, padding: "12px 0" }}>
                <div style={{ textAlign: "center", minWidth: 48 }}>
                  <div style={{
                    width: 32, height: 32, borderRadius: "50%",
                    background: "var(--color-accent)", color: "var(--color-text-on-accent)",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 18, margin: "0 auto",
                  }}>{step.icon}</div>
                  {i < EVOLUTION_STEPS.length - 1 && (
                    <div style={{ width: 2, height: 24, background: "var(--color-border)", margin: "4px auto" }} />
                  )}
                </div>
                <div>
                  <div style={{ fontWeight: 700, fontSize: "var(--text-md)" }}>{step.v} {step.title}</div>
                  <div className="muted mt-1" style={{ fontSize: "var(--text-sm)" }}>{step.desc}</div>
                </div>
              </div>
            ))}
          </div>

          {/* Incremental update */}
          <h2 className="section-title">增量更新 (V5)</h2>
          {Object.keys(incrementalData).length > 0 ? (
            <div className="grid-stat-sm mb-4">
              {["added_count", "unchanged_count", "modified_count", "deleted_count", "reprocessed_pages", "reused_chunks", "embedded_chunks", "removed_chunks"].map(k => (
                <div key={k} className="card card-sm" style={{ textAlign: "center" }}>
                  <div className="stat-label">{k}</div>
                  <div style={{
                    fontSize: 20, fontWeight: 700,
                    color: incrementalData[k] === 0 ? "var(--color-success)" : "var(--color-text)",
                  }}>
                    {String(incrementalData[k] ?? "N/A")}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-state mb-4">暂无缓存实验数据</div>
          )}

          {/* V6 deterministic grounding */}
          {versions && (
            <>
              <h2 className="section-title">V6 确定性接地验证</h2>
              <div className="grid-stat-sm mb-4">
                <VStat label="全量评测 answered" value={`${versions.v6_grounding.answered}/${versions.v6_grounding.total}`} good />
                <VStat label="句级支撑率" value={`${(versions.v6_grounding.avg_support_ratio * 100).toFixed(1)}%`} good />
                <VStat label="投毒拦截率" value={`${versions.v6_grounding.poison_flagged}/${versions.v6_grounding.poison_total}（${(versions.v6_grounding.poison_rate * 100).toFixed(0)}%）`} good />
                <VStat label="过度拒答" value={`${versions.v6_grounding.over_refused}/100`} />
              </div>
              <div className="card mb-4" style={{ lineHeight: "var(--leading-relaxed)" }}>
                <p><strong>验证机制</strong>：答案拆句 → 每句与检索 chunk 用 BGE-Reranker 交叉编码器逐对打分 → 支撑率 ≥ 0.7 才放行；引用由系统"计算"而非 LLM"声称"，并逐条审计 <code>[来源: X, 第Y页]</code>。投毒测试对真实答案追加编造句（"本产品由核聚变反应堆提供动力"等），交叉编码器拦截 {versions.v6_grounding.poison_flagged}/{versions.v6_grounding.poison_total}；BGE-M3 余弦方案为 0/{versions.v6_grounding.poison_total}（主题词抬高相似度，无判别力）。</p>
              </div>
            </>
          )}

          {/* V7 LLM gateway */}
          {versions && (
            <>
              <h2 className="section-title">V7 LLM Gateway 韧性</h2>
              <div className="grid-stat-sm mb-4">
                <VStat label="单次超时" value={`${versions.v7_gateway.timeout_s}s`} />
                <VStat label="最大重试" value={`${versions.v7_gateway.max_retries}`} />
                <VStat label="熔断阈值" value={`连败 ${versions.v7_gateway.circuit_threshold} 次`} />
                <VStat label="冷却时间" value={`${versions.v7_gateway.cooldown_s}s`} />
              </div>
            </>
          )}

          {/* V8 multi-doc */}
          {versions && (
            <>
              <h2 className="section-title">V8 多文档检索</h2>
              <div className="grid-stat-sm mb-4">
                <VStat label="扩展评测题数" value={`${versions.v8_multidoc.questions}`} good />
                <VStat label="V3 Hit@5" value={versions.v8_multidoc.v3_hit_at_5.toFixed(4)} good />
                <VStat label="V3 MRR" value={versions.v8_multidoc.v3_mrr.toFixed(4)} good />
                <VStat label="Ecovacs 跨语言 MRR" value={versions.v8_multidoc.ecovacs_mrr.toFixed(2)} good />
              </div>
              <div className="card mb-4" style={{ lineHeight: "var(--leading-relaxed)" }}>
                <p>第二本英文说明书（Ecovacs DEEBOT T30C）与中文 Roborock 同库。检索按 <code>source_file</code> 元数据过滤（doc_filter），融合前隔离，避免跨书页码冲突与来源混淆。123 题 retrieval-only 评测（{versions.v8_multidoc.text} 文本 + {versions.v8_multidoc.image} 图片），V3 Hit@5 {versions.v8_multidoc.v3_hit_at_5.toFixed(4)}、MRR {versions.v8_multidoc.v3_mrr.toFixed(4)}。</p>
              </div>
            </>
          )}

          {/* V9 semantic cache */}
          {versions && (
            <>
              <h2 className="section-title">V9 语义缓存</h2>
              <div className="grid-stat-sm mb-4">
                <VStat label="精确命中" value={`${versions.v9_cache.exact_hits}/${versions.v9_cache.warmed}`} good />
                <VStat label="语义命中" value={`${versions.v9_cache.semantic_hits}`} />
                <VStat label="命中延迟" value={`${(versions.v9_cache.avg_cached_s * 1000).toFixed(0)}ms`} good />
                <VStat label="节省 LLM 调用" value={`${versions.v9_cache.llm_calls_saved}`} good />
              </div>
              <div className="card mb-4" style={{ lineHeight: "var(--leading-relaxed)" }}>
                <p>两级缓存：SHA256 精确 + BGE-M3 语义余弦（阈值 0.9）。重复/近似问题跳过全管线（未命中 {versions.v9_cache.avg_uncached_s}s）直接返回，平均命中 {(versions.v9_cache.avg_cached_s * 1000).toFixed(0)}ms，精确重跑 {versions.v9_cache.exact_hits}/{versions.v9_cache.warmed}，省 {versions.v9_cache.llm_calls_saved} 次 LLM 调用。SQLite 持久化、TTL 支持；按 doc_filter 加盐隔离不同说明书的缓存。</p>
              </div>
            </>
          )}

          {/* Cases */}
          <h2 className="section-title">典型案例</h2>
          <div style={{ display: "grid", gap: 8, marginBottom: 24 }}>
            <CaseCard status="success" title="Q18 多模态图片问题" content="V0 无法回答（悬崖传感器为图示标注）。V1 加入第 6 页图片语义描述后，V3 Reranker 将 image Chunk 排至 Top-1，最终 answered。" />
            <CaseCard status="success" title="Q19 召回困难问题" content="V0/V1 Dense 均未命中 gold pages [5,24]。V2 Hybrid (RRF) 将 page 24 推入 Top-5，从 MISS→HIT。" />
            <CaseCard status="success" title="越界问题拒答" content="'如何更换核聚变反应堆？' — V4 LangGraph Verify 通过 relevance check 正确拒答（refused），未编造答案。" />
            <CaseCard status="info" title="Reranker 排序变化" content="V3 BGE-Reranker 使 20/20 题排序发生变化，Top-1 命中率从 65% 提升到 70%。" />
          </div>

          {/* Conclusion */}
          <h2 className="section-title">实验结论</h2>
          {finalEval && (
            <div className="card card-sm mb-3" style={{ background: "var(--color-success-soft)", color: "var(--color-success)" }}>
              最终评测：{finalEval.total_questions} 题 · {finalEval.timestamp} · RAGAS {finalEval.ragas_version}
            </div>
          )}
          <div className="card" style={{ lineHeight: "var(--leading-relaxed)" }}>
            <p>✅ <strong>多模态解析</strong>（V1）改善了图片/表格问题的检索覆盖，Q18 从无法回答变为可回答。但增加噪声导致 Recall@5 从 0.89 降至 0.81。</p>
            <p>✅ <strong>Hybrid Retrieval</strong>（V2）通过 BM25+RRF 改善关键词匹配，MRR 从 V0 的 0.76 提升至 0.84（+10.4%），性价比最优（仅 3.3s 延迟）。</p>
            <p>✅ <strong>BGE-Reranker</strong>（V3）全面提升：Recall@5 0.91、MRR 0.86、Faithfulness 0.95、Context Precision 0.87 —— 全部指标最优，代价是 20s 延迟（6x）。</p>
            <p>✅ <strong>LangGraph Verify</strong>（V4）保持 V3 的检索质量，Context Recall 最优（0.88），越界问题正确 refused。Faithfulness 略低于 V3（0.90 vs 0.95），验证机制可能过度保守。</p>
            <p>✅ <strong>增量更新</strong>（V5）unchanged 文档 0 Embedding，39 chunks 全部复用，Hash 检测准确，BM25 增量同步正常。</p>
            <p>🛡️ <strong>确定性接地</strong>（V6）句级交叉编码器验证替代 LLM 自查，投毒测试拦截 82% 编造句（余弦为 0%）；引用由系统计算而非 LLM 声称。</p>
            <p>🚦 <strong>LLM Gateway</strong>（V7）60s 超时 + 指数退避重试 + 每 Provider 断路器 + 多 Provider 故障转移，全挂时兜底句自动落入可信拒答。</p>
            <p>📚 <strong>多文档检索</strong>（V8）双说明书 + doc_filter 元数据隔离，跨语言检索 MRR 0.90，123 题 retrieval-only Hit@5 0.9756。</p>
            <p>⚡ <strong>语义缓存</strong>（V9）两级缓存命中 ~31ms（全管线 20s+），精确重跑 12/12，节省 LLM 调用成本。</p>
            <p style={{ color: "var(--color-accent)" }}>⚠️ V2→V3 的 Reranker 带来全面但昂贵（6x 延迟）的提升。Context Precision 从 V0 的 0.69 提升至 V3 的 0.87，进步显著；V6 已知边界为 1/100 过度拒答（否定释义短句），V9 缓存知识库更新后需清理。</p>
          </div>
        </>
      )}
    </div>
  );
}

function VStat({ label, value, good }: { label: string; value: string; good?: boolean }) {
  return (
    <div className="card card-sm" style={{ textAlign: "center" }}>
      <div className="stat-label">{label}</div>
      <div style={{ fontSize: 20, fontWeight: 700, color: good ? "var(--color-success)" : "var(--color-text)" }}>{value}</div>
    </div>
  );
}

function MiniCard({ label, value, version }: { label: string; value: string; version: string }) {
  return (
    <div className="card" style={{ padding: "var(--space-4)" }}>
      <div className="stat-label">{label}</div>
      <div style={{ fontSize: 24, fontWeight: 700, color: "var(--color-accent)" }}>{value}</div>
      <div className="muted mt-1" style={{ fontSize: "var(--text-xs)" }}>{version ? VERSION_NAMES[version] || version : ""}</div>
    </div>
  );
}

function CaseCard({ title, content, status }: { title: string; content: string; status: string }) {
  return (
    <div className="card card-sm" style={{
      borderLeft: `4px solid ${status === "success" ? "var(--color-success)" : "var(--color-info)"}`,
    }}>
      <div style={{ fontWeight: 600, fontSize: "var(--text-base)", marginBottom: 4 }}>{title}</div>
      <div className="muted" style={{ fontSize: "var(--text-sm)", lineHeight: "var(--leading-normal)" }}>{content}</div>
    </div>
  );
}
