"use client";

import { BarChart3, Bot, Gauge, Play, ShieldAlert, ShieldCheck, Sparkles } from "lucide-react";
import type { DashboardState, TabId } from "@/lib/types";
import { MetricCard } from "./MetricCard";
import { RiskChart } from "./RiskChart";
import { RiskBadge } from "./RiskBadge";

export function OverviewPage({ state, setTab, loadSample }: { state: DashboardState; setTab: (tab: TabId) => void; loadSample: () => void }) {
  const last = state.selectedResult;
  return (
    <div className="space-y-6">
      <section className="overflow-hidden rounded-[2rem] border border-white/10 bg-[linear-gradient(135deg,rgba(34,211,238,0.18),rgba(168,85,247,0.12),rgba(244,63,94,0.10))] p-7 shadow-glow md:p-9">
        <p className="text-xs uppercase tracking-[0.24em] text-cyan-100">AI safety research product</p>
        <h1 className="mt-4 max-w-4xl text-4xl font-semibold leading-tight text-white md:text-6xl">Hallucination Detection Studio</h1>
        <p className="mt-4 max-w-3xl text-base leading-7 text-slate-300">Audit LLM answers with internal signals, source grounding, retrieval checks, staged verification, CoVe-style revision, and CRITIC-lite tool traces.</p>
        <div className="mt-7 flex flex-wrap gap-3">
          <button onClick={loadSample} className="rounded-2xl bg-cyan-300 px-5 py-3 text-sm font-semibold text-slate-950">Start with sample</button>
          <button onClick={() => setTab("ask")} className="rounded-2xl bg-white/10 px-5 py-3 text-sm font-semibold text-white">Open Ask Studio</button>
          <button onClick={() => setTab("analyze")} className="rounded-2xl bg-fuchsia-300/15 px-5 py-3 text-sm font-semibold text-fuchsia-100">Run Compare Mode</button>
        </div>
      </section>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard icon={Sparkles} label="Backend status" value={state.backendOnline ? "Online" : "Offline"} detail={state.backendOnline ? "FastAPI connected" : "UI fallback metadata active"} tone={state.backendOnline ? "green" : "rose"} />
        <MetricCard icon={Gauge} label="Methods available" value={state.methods.length} detail="All detector families visible" tone="cyan" />
        <MetricCard icon={Bot} label="Selected methods" value={state.selectedMethods.length} detail={state.mode === "quick" ? "Quick Check" : "Compare Methods"} tone="purple" />
        <MetricCard icon={ShieldAlert} label="Last risk" value={last?.risk_label ?? "No run"} detail={last?.risk_score == null ? "Load a sample or run analysis" : `${Math.round(last.risk_score)} risk score`} tone={last?.risk_label === "High" ? "rose" : "amber"} />
        <MetricCard icon={ShieldCheck} label="Claims checked" value={state.summary?.claims_checked ?? 0} tone="indigo" />
        <MetricCard icon={Gauge} label="Analysis history" value={state.analysisHistory.length} detail={state.lastRunAt ? new Date(state.lastRunAt).toLocaleString() : "Session-local runs"} tone="cyan" />
        <MetricCard icon={ShieldCheck} label="Low risk" value={state.summary?.low ?? 0} tone="green" />
        <MetricCard icon={ShieldAlert} label="Medium risk" value={state.summary?.medium ?? 0} tone="amber" />
        <MetricCard icon={ShieldAlert} label="High risk" value={state.summary?.high ?? 0} tone="rose" />
      </div>
      <div className="grid gap-5 xl:grid-cols-[1fr_360px]">
        <div className="glass rounded-3xl p-5">
          <div className="mb-4 flex items-center justify-between"><h2 className="text-xl font-semibold text-white">Risk distribution</h2><BarChart3 className="text-cyan-100" /></div>
          <RiskChart results={state.results} />
        </div>
        <div className="glass rounded-3xl p-5">
          <h2 className="text-xl font-semibold text-white">Recent analysis history</h2>
          <div className="mt-4 space-y-3">
            {state.analysisHistory.slice(0, 4).map((item) => (
              <button key={item.id} onClick={() => setTab("results")} className="w-full rounded-2xl bg-white/5 p-3 text-left hover:bg-white/8">
                <div className="flex items-center justify-between gap-2"><p className="line-clamp-1 text-sm font-semibold text-white">{item.question_preview}</p><RiskBadge label={item.highest_risk_label} /></div>
                <p className="mt-2 text-xs text-slate-500">{item.selected_methods.length} method(s) · avg risk {item.avg_risk}</p>
              </button>
            ))}
            {!state.analysisHistory.length && <p className="text-sm leading-6 text-slate-400">Load a curated case or paste a question and answer in Ask Studio to generate a real detector response.</p>}
          </div>
          <button onClick={() => setTab(state.results.length ? "results" : "ask")} className="mt-5 inline-flex items-center gap-2 rounded-2xl bg-emerald-300 px-4 py-3 text-sm font-semibold text-slate-950"><Play size={16} /> Continue</button>
        </div>
      </div>
    </div>
  );
}
