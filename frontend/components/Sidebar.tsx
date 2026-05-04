"use client";

import { Activity, BookOpen, Bot, FileDown, FlaskConical, GitBranch, LayoutDashboard, Library, Network, Play, SearchCheck } from "lucide-react";
import { normalizeRiskLabel } from "@/lib/risk";
import type { DashboardState, TabId } from "@/lib/types";

const tabs: { id: TabId; label: string; icon: typeof LayoutDashboard }[] = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "askQuick", label: "ASK Quick Mode", icon: Bot },
  { id: "compareDetectors", label: "Compare Detectors", icon: FlaskConical },
  { id: "report", label: "Report / Export", icon: FileDown },
  { id: "results", label: "Results", icon: SearchCheck },
  { id: "samples", label: "Samples", icon: BookOpen },
  { id: "flow", label: "Method Flow", icon: GitBranch },
  { id: "library", label: "Method Library", icon: Library }
];

function widgetTone(label?: string) {
  const normalized = normalizeRiskLabel(label);
  if (normalized === "Low") return { border: "border-emerald-300/30", glow: "shadow-[0_0_34px_rgba(16,185,129,0.18)]", icon: "text-emerald-100 bg-emerald-300/12 ring-emerald-300/30", text: "text-emerald-100" };
  if (normalized === "Medium") return { border: "border-amber-300/30", glow: "shadow-[0_0_34px_rgba(251,191,36,0.18)]", icon: "text-amber-100 bg-amber-300/12 ring-amber-300/30", text: "text-amber-100" };
  if (normalized === "High") return { border: "border-rose-300/30", glow: "shadow-[0_0_34px_rgba(244,63,94,0.20)]", icon: "text-rose-100 bg-rose-300/12 ring-rose-300/30", text: "text-rose-100" };
  return { border: "border-cyan-300/25", glow: "shadow-[0_0_34px_rgba(34,211,238,0.16)]", icon: "text-cyan-100 bg-cyan-300/12 ring-cyan-300/30", text: "text-cyan-100" };
}

export function Sidebar({
  activeTab,
  onTabChange,
  state,
  onRunSample
}: {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
  state: DashboardState;
  onRunSample: () => void;
}) {
  const lastRisk = state.selectedResult?.risk_label;
  const tone = widgetTone(lastRisk);
  const modeLabel = state.mode === "compare" ? "Compare Detectors" : "ASK Quick Mode";
  const scanText = state.loading ? "Scanning..." : lastRisk ? `Last scan: ${normalizeRiskLabel(lastRisk)} risk` : "Ready to scan";

  return (
    <aside className="glass sticky top-5 hidden h-[calc(100vh-2.5rem)] w-72 shrink-0 flex-col rounded-3xl p-5 lg:flex">
      <div className="mb-7 flex items-center gap-3">
        <div className="grid h-12 w-12 place-items-center rounded-2xl bg-cyan-300/15 text-cyan-100 ring-1 ring-cyan-300/30">
          <Network size={23} />
        </div>
        <div>
          <p className="text-sm font-semibold text-white">Hallucination</p>
          <p className="text-xs text-slate-400">Detection Studio</p>
        </div>
      </div>
      <nav className="space-y-2">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const active = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={`flex w-full items-center gap-3 rounded-2xl px-3 py-3 text-left text-sm transition ${
                active ? "bg-cyan-300 text-slate-950 shadow-glow" : "text-slate-300 hover:bg-white/8 hover:text-white"
              }`}
            >
              <Icon size={18} />
              {tab.label}
            </button>
          );
        })}
      </nav>
      <div className={`glass-panel-ai mt-auto rounded-3xl border ${tone.border} bg-slate-950/45 p-4 ${tone.glow}`}>
        <div className="flex items-center gap-3">
          <div className={`relative grid h-12 w-12 shrink-0 place-items-center rounded-2xl ring-1 ${tone.icon}`}>
            <Activity className="relative" size={21} />
          </div>
          <div className="min-w-0">
            <p className="text-xs uppercase tracking-[0.18em] text-cyan-200">AI Safety Engine</p>
            <p className={`mt-1 truncate text-sm font-semibold ${tone.text}`}>{scanText}</p>
          </div>
        </div>
        <div className="mt-4 grid gap-2 text-xs">
          <div className="flex items-center justify-between gap-3 rounded-2xl bg-white/5 px-3 py-2">
            <span className="text-slate-400">Backend</span>
            <span className={state.backendOnline ? "font-semibold text-emerald-100" : "font-semibold text-rose-100"}>{state.backendOnline ? "Online" : "Offline"}</span>
          </div>
          <div className="flex items-center justify-between gap-3 rounded-2xl bg-white/5 px-3 py-2">
            <span className="text-slate-400">Detectors</span>
            <span className="font-semibold text-slate-100">{state.methods.length} ready</span>
          </div>
          <div className="rounded-2xl bg-white/5 px-3 py-2">
            <span className="text-slate-400">Mode</span>
            <p className="mt-1 truncate font-semibold text-slate-100">{modeLabel}</p>
          </div>
        </div>
        <button onClick={onRunSample} className="mt-4 inline-flex w-full items-center justify-center gap-2 rounded-2xl bg-cyan-300 px-3 py-2.5 text-xs font-semibold text-slate-950 transition hover:bg-cyan-200">
          <Play size={14} /> Run sample
        </button>
      </div>
    </aside>
  );
}
