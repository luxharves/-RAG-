import { useState } from "react";
import { Dashboard } from "./pages/Dashboard";
import { SystemStatus } from "./pages/SystemStatus";
import { QAPanel } from "./pages/QAPanel";
import { KnowledgeBase } from "./pages/KnowledgeBase";
import { Experiments } from "./pages/Experiments";

type Page = "dashboard" | "qa" | "knowledge" | "experiments" | "status";

export default function App() {
  const [page, setPage] = useState<Page>("dashboard");

  const navItems: { key: Page; label: string; icon: string }[] = [
    { key: "dashboard", label: "仪表盘", icon: "◫" },
    { key: "qa", label: "维保问答", icon: "▣" },
    { key: "knowledge", label: "知识库管理", icon: "◰" },
    { key: "experiments", label: "实验评估", icon: "▤" },
    { key: "status", label: "系统状态", icon: "◉" },
  ];

  return (
    <div className="app-shell">
      <nav className="sidebar">
        <div className="sidebar-brand">智能维保助手</div>
        {navItems.map((item) => (
          <button
            key={item.key}
            onClick={() => setPage(item.key)}
            className={"sidebar-nav-btn" + (page === item.key ? " active" : "")}
          >
            <span style={{ marginRight: 8, opacity: page === item.key ? 1 : 0.5 }}>{item.icon}</span>
            {item.label}
          </button>
        ))}
        <div className="sidebar-footer">V5 · FastAPI</div>
      </nav>

      <main className="main-content">
        {page === "dashboard" && <Dashboard />}
        {page === "qa" && <QAPanel />}
        {page === "knowledge" && <KnowledgeBase />}
        {page === "experiments" && <Experiments />}
        {page === "status" && <SystemStatus />}
      </main>
    </div>
  );
}
