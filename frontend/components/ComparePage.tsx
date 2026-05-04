"use client";

import { BarChart3, Gauge, ShieldAlert, ShieldCheck, TrendingUp } from "lucide-react";
import { riskTone } from "@/lib/risk";
import type { DashboardState, DetectorResult, TabId } from "@/lib/types";
import { ComparisonTable } from "./ComparisonTable";
import { EmptyState } from "./EmptyState";
import { RiskBadge } from "./RiskBadge";
import { RiskChart } from "./RiskChart";

function SummaryCard({ title, result, value, icon: Icon }: { title: string; result?: DetectorResult; value?: string | number; icon: typeof Gauge }) {
  const tone = riskTone(result?.risk_label);
  return (
    <div className={`rounded-3xl bg-gradient-to-br ${tone.border} p-px ${tone.glow}`}>
      <div className="h-full rounded-3xl border border-white/10 bg-slate-950/75 p-5 backdrop-blur-xl">
        <p className="flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-cyan-200"><Icon size={14} /> {title}</p>
        {result ? (
          <>
            <div className="mt-4 flex items-center justify-between gap-2"><p className="line-clamp-1 text-base font-semibold text-white">{result.method_name}</p><RiskBadge label={result.risk_label} /></div>
            <p className="mt-3 text-4xl font-semibold text-white">{result.risk_score ?? "N/A"}</p>
            <p className="mt-2 text-xs text-slate-500">{result.family}</p>
          </>
        ) : <p className="mt-4 text-4xl font-semibold text-white">{value}</p>}
      </div>
    </div>
  );
}

export function ComparePage({ state, setTab }: { state: DashboardState; setTab: (tab: TabId) => void }) {
  if (state.results.length < 2) {
    return (
      <EmptyState
        icon={BarChart3}
        title="Compare Results needs multiple detector outputs"
        message="Run two or more methods in Compare Detectors to view charts, rankings, confidence comparison, and disagreement summary."
        actions={<button onClick={() => setTab("compareDetectors")} className="rounded-2xl bg-cyan-300 px-4 py-3 text-sm font-semibold text-slate-950">Open Compare Detectors</button>}
      />
    );
  }

  const ranked = [...state.results].sort((a, b) => (b.risk_score ?? -1) - (a.risk_score ?? -1));
  const highest = ranked[0];
  const lowest = ranked[ranked.length - 1];
  const avgRisk = Math.round(state.results.reduce((sum, r) => sum + (r.risk_score ?? 0), 0) / state.results.length);
  const grounded = state.results.filter((r) => /ground|retrieval|rag|source/i.test(`${r.family} ${r.method_name}`));
  const internal = state.results.filter((r) => /internal|sep/i.test(`${r.family} ${r.method_name}`));
  const groundedAvg = grounded.length ? grounded.reduce((sum, r) => sum + (r.risk_score ?? 0), 0) / grounded.length : null;
  const internalAvg = internal.length ? internal.reduce((sum, r) => sum + (r.risk_score ?? 0), 0) / internal.length : null;
  const spread = (highest?.risk_score ?? 0) - (lowest?.risk_score ?? 0);
  const hasContradiction = state.results.some((r) => r.claim_findings?.some((claim) => /contradict/i.test(String(claim.verdict ?? claim.status ?? ""))));

  return (
    <div className="w-full space-y-5">
      <div className="glass rounded-3xl p-5">
        <h1 className="text-2xl font-semibold text-white">Compare Results</h1>
        <p className="mt-1 text-sm text-slate-400">Charts, rankings, risk scores, confidence, and disagreement across real detector outputs.</p>
      </div>
      <div className="grid gap-4 lg:grid-cols-3">
        <SummaryCard title="Highest risk" result={highest} icon={ShieldAlert} />
        <SummaryCard title="Lowest risk" result={lowest} icon={ShieldCheck} />
        <SummaryCard title="Average risk" value={avgRisk} icon={TrendingUp} />
      </div>
      <div className="glass rounded-3xl p-5">
        <h2 className="mb-4 text-xl font-semibold text-white">Risk chart</h2>
        <div className="min-h-[360px]"><RiskChart results={state.results} /></div>
      </div>
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {ranked.map((result) => {
          const tone = riskTone(result.risk_label);
          const score = Math.min(100, Math.max(0, result.risk_score ?? 0));
          return (
            <div key={result.method_name} className={`rounded-3xl bg-gradient-to-br ${tone.border} p-px`}>
              <div className="h-full rounded-3xl bg-slate-950/75 p-4 backdrop-blur-xl">
                <div className="flex items-start justify-between gap-2"><p className="text-sm font-semibold text-white">{result.method_name}</p><RiskBadge label={result.risk_label} /></div>
                <p className="mt-2 text-xs text-slate-500">{result.family}</p>
                <div className="mt-4 h-2 rounded-full bg-white/10"><div className={`h-full rounded-full ${tone.fill}`} style={{ width: `${score}%` }} /></div>
                <p className="mt-3 text-sm text-slate-300">Risk {result.risk_score ?? "N/A"} · confidence {result.confidence == null ? "N/A" : `${Math.round(result.confidence * 100)}%`}</p>
              </div>
            </div>
          );
        })}
      </div>
      <div className="glass rounded-3xl p-5">
        <h2 className="mb-4 text-xl font-semibold text-white">Comparison table</h2>
        <ComparisonTable results={state.results} methods={state.methods} />
      </div>
      <div className="glass rounded-3xl p-5">
        <h2 className="text-xl font-semibold text-white">Disagreement summary</h2>
        <div className="mt-4 grid gap-3 text-sm leading-7 text-slate-300 md:grid-cols-2">
          <p><b className="text-white">Highest risk:</b> {highest.method_name} returned {highest.risk_label} ({highest.risk_score ?? "N/A"}).</p>
          <p><b className="text-white">Lowest risk:</b> {lowest.method_name} returned {lowest.risk_label} ({lowest.risk_score ?? "N/A"}).</p>
          <p><b className="text-white">Spread:</b> {Math.round(spread)} points between the highest and lowest returned risk scores.</p>
          {groundedAvg != null && internalAvg != null && <p><b className="text-white">Grounded vs internal:</b> grounded methods average {Math.round(groundedAvg)} risk; internal-signal methods average {Math.round(internalAvg)} risk.</p>}
          <p><b className="text-white">Evidence status:</b> {hasContradiction ? "At least one method reported contradicted claims." : "Returned claims are mostly unsupported or supported rather than explicitly contradicted."}</p>
        </div>
      </div>
    </div>
  );
}
