"use client";

import { BookOpen, ShieldCheck, ShieldX } from "lucide-react";
import type { DashboardState, TabId } from "@/lib/types";
import { EmptyState } from "./EmptyState";
import { RiskBadge } from "./RiskBadge";
import { VisualResultCard } from "./VisualResultCard";

export function ResultsPage({
  state,
  setSelectedResult,
  setTab,
  loadRiskSample
}: {
  state: DashboardState;
  setSelectedResult: (name: string) => void;
  setTab: (tab: TabId) => void;
  loadRiskSample: (risk: "Low" | "High", tab?: "ask" | "analyze") => void;
}) {
  const result = state.selectedResult;
  if (!result) {
    return (
      <EmptyState
        icon={BookOpen}
        title="No detector result yet"
        message="Load a curated low-risk or high-risk sample, or run Ask Studio / Analyze to generate real method outputs."
        actions={<><button onClick={() => loadRiskSample("Low", "ask")} className="inline-flex items-center gap-2 rounded-2xl bg-emerald-300/15 px-4 py-3 text-sm font-semibold text-emerald-100"><ShieldCheck size={16} /> Load low-risk sample</button><button onClick={() => loadRiskSample("High", "ask")} className="inline-flex items-center gap-2 rounded-2xl bg-rose-300/15 px-4 py-3 text-sm font-semibold text-rose-100"><ShieldX size={16} /> Load high-risk sample</button><button onClick={() => setTab("ask")} className="rounded-2xl bg-cyan-300 px-4 py-3 text-sm font-semibold text-slate-950">Go to Ask Studio</button><button onClick={() => setTab("samples")} className="rounded-2xl bg-white/10 px-4 py-3 text-sm font-semibold text-white">Go to Samples</button></>}
      />
    );
  }
  const other = state.results.filter((item) => item.method_name !== result.method_name);
  return (
    <div className="w-full space-y-5">
      <div className="glass rounded-3xl p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div><h1 className="text-2xl font-semibold text-white">Results</h1><p className="mt-1 text-sm text-slate-400">Premium detector report powered by the shared backend result schema.</p></div>
          <div className="flex flex-wrap gap-2">{state.results.map((r) => <button key={r.method_name} onClick={() => setSelectedResult(r.method_name)} className={`rounded-full px-3 py-2 text-xs ${r.method_name === result.method_name ? "bg-cyan-300 text-slate-950" : "bg-white/8 text-slate-300 hover:bg-white/12"}`}>{r.method_name}</button>)}</div>
        </div>
      </div>
      <VisualResultCard result={result} state={state} />
      {other.length > 0 && (
        <div className="glass rounded-3xl p-5">
          <h2 className="text-xl font-semibold text-white">Other method results</h2>
          <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {other.map((item) => (
              <button key={item.method_name} onClick={() => setSelectedResult(item.method_name)} className="rounded-2xl border border-white/10 bg-white/[0.05] p-4 text-left transition hover:border-cyan-300/35 hover:bg-white/[0.075]">
                <div className="flex items-center justify-between gap-2"><p className="line-clamp-1 text-sm font-semibold text-white">{item.method_name}</p><RiskBadge label={item.risk_label} /></div>
                <p className="mt-2 text-xs text-slate-500">{item.family}</p>
                <p className="mt-3 text-2xl font-semibold text-white">{item.risk_score ?? "N/A"}</p>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
