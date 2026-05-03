"use client";

import { useState } from "react";
import { Bot, BookOpen, Send, ShieldCheck, ShieldX, SlidersHorizontal } from "lucide-react";
import type { DashboardState, MethodInfo, SampleCase, StudioForm, TabId } from "@/lib/types";
import { SamplePickerPanel } from "./SamplePickerPanel";
import { VisualResultCard } from "./VisualResultCard";

export function AskStudioPage({
  state,
  setField,
  toggleMethod,
  run,
  loadRiskSample,
  loadSelectedSample,
  runSample,
  setTab
}: {
  state: DashboardState;
  setField: (key: keyof StudioForm, value: string) => void;
  toggleMethod: (method: string) => void;
  run: () => void;
  loadRiskSample: (risk: "Low" | "High", tab?: "ask" | "analyze") => void;
  loadSelectedSample: (sample: SampleCase) => void;
  runSample: (sample: SampleCase) => void;
  setTab: (tab: TabId) => void;
}) {
  const result = state.selectedResult;
  const [pickerOpen, setPickerOpen] = useState(false);
  const loadFromPicker = (sample: SampleCase) => {
    loadSelectedSample(sample);
    setPickerOpen(false);
  };
  const runFromPicker = (sample: SampleCase) => {
    runSample(sample);
    setPickerOpen(false);
  };
  return (
    <div className="w-full">
      <section className="glass min-h-[calc(100vh-8rem)] rounded-3xl p-5 md:p-6">
        <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="grid h-12 w-12 place-items-center rounded-2xl bg-cyan-300/15 text-cyan-100 ring-1 ring-cyan-300/25"><Bot /></div>
            <div><h1 className="text-2xl font-semibold text-white">Ask Studio</h1><p className="text-sm text-slate-400">Conversational answer audit workspace</p></div>
          </div>
          <span className={`rounded-full px-3 py-2 text-xs font-semibold ${state.backendOnline ? "bg-emerald-300/12 text-emerald-100" : "bg-rose-300/12 text-rose-100"}`}>Backend {state.backendOnline ? "online" : "offline"}</span>
        </div>

        <div className="rounded-3xl border border-white/10 bg-white/[0.04] p-4 backdrop-blur-xl">
          <h2 className="text-sm font-semibold text-white">Method chips</h2>
          <div className="mt-3 flex flex-wrap gap-2">
            {state.methods.map((method: MethodInfo) => (
              <button key={method.name} onClick={() => toggleMethod(method.name)} className={`rounded-full px-3 py-2 text-xs transition ${state.selectedMethods.includes(method.name) ? "bg-cyan-300 text-slate-950 shadow-glow" : "bg-white/8 text-slate-300 hover:bg-white/12"}`}>{method.name}</button>
            ))}
          </div>
        </div>

        <div className="mt-4 flex flex-wrap gap-3">
          <button onClick={() => loadRiskSample("Low", "ask")} className="inline-flex items-center gap-2 rounded-2xl bg-emerald-300/15 px-4 py-3 text-sm font-semibold text-emerald-100 hover:bg-emerald-300/20"><ShieldCheck size={16} /> Load low-risk sample</button>
          <button onClick={() => loadRiskSample("High", "ask")} className="inline-flex items-center gap-2 rounded-2xl bg-rose-300/15 px-4 py-3 text-sm font-semibold text-rose-100 hover:bg-rose-300/20"><ShieldX size={16} /> Load high-risk sample</button>
          <button onClick={() => setPickerOpen(true)} className="inline-flex items-center gap-2 rounded-2xl bg-cyan-300 px-4 py-3 text-sm font-semibold text-slate-950"><BookOpen size={16} /> Browse samples</button>
          <button onClick={() => setTab("samples")} className="rounded-2xl bg-white/10 px-4 py-3 text-sm font-semibold text-white hover:bg-white/15">Open Samples page</button>
        </div>

        <div className="mt-5 grid gap-3">
          <textarea value={state.question} onChange={(e) => setField("question", e.target.value)} placeholder="Question / prompt" className="min-h-28 rounded-2xl border border-white/10 bg-slate-950/60 p-4 text-sm outline-none transition focus:border-cyan-300/60" />
          <textarea value={state.answer} onChange={(e) => setField("answer", e.target.value)} placeholder="LLM answer to evaluate" className="min-h-40 rounded-2xl border border-white/10 bg-slate-950/60 p-4 text-sm outline-none transition focus:border-cyan-300/60" />
          <details className="rounded-2xl border border-white/10 bg-white/5 p-4 backdrop-blur-xl">
            <summary className="flex cursor-pointer items-center gap-2 text-sm font-semibold text-white"><SlidersHorizontal size={16} /> Optional source, evidence, and sampled answers</summary>
            <div className="mt-4 grid gap-3 xl:grid-cols-3">
              <textarea value={state.source_text} onChange={(e) => setField("source_text", e.target.value)} placeholder="Source text" className="min-h-32 rounded-2xl border border-white/10 bg-slate-950/60 p-4 text-sm outline-none" />
              <textarea value={state.evidence_text} onChange={(e) => setField("evidence_text", e.target.value)} placeholder="Evidence text" className="min-h-32 rounded-2xl border border-white/10 bg-slate-950/60 p-4 text-sm outline-none" />
              <textarea value={state.sampled_answers_text} onChange={(e) => setField("sampled_answers_text", e.target.value)} placeholder="Sampled answers" className="min-h-32 rounded-2xl border border-white/10 bg-slate-950/60 p-4 text-sm outline-none" />
            </div>
          </details>
          <button disabled={!state.backendOnline || state.loading} onClick={run} className="inline-flex items-center justify-center gap-2 rounded-2xl bg-cyan-300 px-5 py-4 text-sm font-semibold text-slate-950 transition hover:bg-cyan-200 disabled:cursor-not-allowed disabled:opacity-50"><Send size={17} /> Analyze this answer</button>
          {result && (
            <div className="mt-5 w-full overflow-visible">
              <VisualResultCard result={result} state={state} />
            </div>
          )}
        </div>
      </section>
      <SamplePickerPanel open={pickerOpen} samples={state.samples} methods={state.methods} selectedMethods={state.selectedMethods} onLoad={loadFromPicker} onRun={runFromPicker} onClose={() => setPickerOpen(false)} />
    </div>
  );
}
