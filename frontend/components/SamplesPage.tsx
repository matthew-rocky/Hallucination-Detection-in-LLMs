"use client";

import { useMemo, useState } from "react";
import { Server, ShieldCheck, ShieldX } from "lucide-react";
import { normalizeRiskLabel } from "@/lib/risk";
import type { DashboardState, SampleCase } from "@/lib/types";
import { SampleCard } from "./SampleCard";

export function SamplesPage({
  state,
  loadSample,
  runSample
}: {
  state: DashboardState;
  loadSample: (sample: SampleCase, tab: "askQuick" | "compareDetectors") => void;
  runSample: (sample: SampleCase) => void;
}) {
  const [query, setQuery] = useState("");
  const [method, setMethod] = useState("all");
  const [risk, setRisk] = useState("all");
  const filtered = useMemo(() => state.samples.filter((sample) => {
    const haystack = `${sample.title} ${sample.description} ${sample.question} ${sample.answer}`.toLowerCase();
    const matchesQuery = haystack.includes(query.toLowerCase());
    const matchesMethod = method === "all" || sample.method_targets.includes(method) || sample.recommended_methods?.includes(method);
    const matchesRisk = risk === "all" || normalizeRiskLabel(sample.risk_level) === risk;
    return matchesQuery && matchesMethod && matchesRisk;
  }), [state.samples, query, method, risk]);
  const low = filtered.filter((sample) => normalizeRiskLabel(sample.risk_level) === "Low");
  const high = filtered.filter((sample) => normalizeRiskLabel(sample.risk_level) === "High");

  return (
    <div className="w-full space-y-5">
      <div className="glass rounded-3xl p-5">
        <h1 className="text-2xl font-semibold text-white">Sample Cases</h1>
        <p className="mt-1 text-sm text-slate-400">Real curated low-risk and high-risk examples from `data/sample_cases.py`.</p>
        <div className="mt-5 grid gap-3 md:grid-cols-[1fr_260px_180px]">
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search samples..." className="rounded-2xl border border-white/10 bg-slate-950/60 px-4 py-3 text-sm outline-none" />
          <select value={method} onChange={(e) => setMethod(e.target.value)} className="rounded-2xl border border-white/10 bg-slate-950/60 px-4 py-3 text-sm outline-none"><option value="all">All methods</option>{state.methods.map((m) => <option key={m.name} value={m.name}>{m.name}</option>)}</select>
          <select value={risk} onChange={(e) => setRisk(e.target.value)} className="rounded-2xl border border-white/10 bg-slate-950/60 px-4 py-3 text-sm outline-none"><option value="all">All risk</option><option value="Low">Low risk</option><option value="High">High risk</option></select>
        </div>
      </div>
      {!state.samples.length && (
        <div className="rounded-3xl border border-amber-300/25 bg-amber-300/10 p-5 text-amber-100 backdrop-blur-xl">
          <div className="flex gap-3"><Server className="mt-1 shrink-0" /><div><p className="font-semibold">Samples are unavailable</p><p className="mt-2 text-sm leading-6">Start the backend to load curated sample cases.</p><code className="mt-3 block rounded-xl bg-slate-950/70 p-3 text-xs text-cyan-100">python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000</code></div></div>
        </div>
      )}
      <section className="space-y-4">
        <h2 className="flex items-center gap-2 text-lg font-semibold text-emerald-100"><ShieldCheck size={18} /> LOW-RISK SAMPLES</h2>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
          {low.map((sample) => <SampleCard key={sample.id} sample={sample} onLoadAsk={() => loadSample(sample, "askQuick")} onLoadCompareDetectors={() => loadSample(sample, "compareDetectors")} onRun={() => runSample(sample)} />)}
        </div>
        {!low.length && state.samples.length > 0 && <p className="rounded-2xl bg-white/5 p-4 text-sm text-slate-500">No low-risk samples match the current filters.</p>}
      </section>
      <section className="space-y-4">
        <h2 className="flex items-center gap-2 text-lg font-semibold text-rose-100"><ShieldX size={18} /> HIGH-RISK SAMPLES</h2>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
          {high.map((sample) => <SampleCard key={sample.id} sample={sample} onLoadAsk={() => loadSample(sample, "askQuick")} onLoadCompareDetectors={() => loadSample(sample, "compareDetectors")} onRun={() => runSample(sample)} />)}
        </div>
        {!high.length && state.samples.length > 0 && <p className="rounded-2xl bg-white/5 p-4 text-sm text-slate-500">No high-risk samples match the current filters.</p>}
      </section>
    </div>
  );
}
