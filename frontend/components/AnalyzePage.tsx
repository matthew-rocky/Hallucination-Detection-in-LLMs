"use client";

import { RotateCcw, UploadCloud, Zap } from "lucide-react";
import type { DashboardState, FieldSpec, StudioForm } from "@/lib/types";
import { MethodCard } from "./MethodCard";
import { VisualResultCard } from "./VisualResultCard";

const order = ["question", "answer", "sampled_answers", "source_text", "evidence_text", "uploaded_documents"];
const keyMap: Record<string, keyof StudioForm> = {
  question: "question",
  answer: "answer",
  sampled_answers: "sampled_answers_text",
  source_text: "source_text",
  evidence_text: "evidence_text"
};

export function AnalyzePage({
  state,
  fields,
  files,
  toggleMethod,
  setField,
  setFiles,
  clear,
  run,
  loadSample
}: {
  state: DashboardState;
  fields: Record<string, FieldSpec>;
  files: File[];
  toggleMethod: (method: string) => void;
  setField: (key: keyof StudioForm, value: string) => void;
  setFiles: (files: File[]) => void;
  clear: () => void;
  run: () => void;
  loadSample: () => void;
}) {
  const selected = new Set(state.selectedMethods);
  const visible = order.filter((field) => field === "answer" || state.methods.some((method) => selected.has(method.name) && method.visible_fields.includes(field)));
  return (
    <div className="space-y-5">
      <div className="glass rounded-3xl p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div><h1 className="text-2xl font-semibold text-white">Compare Detectors</h1><p className="mt-1 text-sm text-slate-400">Advanced detector comparison workspace with method-aware inputs.</p></div>
          <span className="rounded-full bg-cyan-300/15 px-3 py-2 text-xs font-semibold text-cyan-100">Compare mode</span>
        </div>
        {state.selectedMethods.length < 2 && (
          <div className="mt-4 rounded-2xl border border-amber-300/25 bg-amber-300/10 p-3 text-sm text-amber-100">
            Select two or more detector methods to compare behavior, risk scoring, confidence, and evidence handling.
          </div>
        )}
        <div className="mt-5 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          {state.methods.map((method) => <MethodCard key={method.name} method={method} active={state.selectedMethods.includes(method.name)} onClick={() => toggleMethod(method.name)} compact />)}
        </div>
      </div>
      <div className="grid gap-5 xl:grid-cols-[1fr_340px]">
        <div className="glass rounded-3xl p-5">
          <h2 className="text-xl font-semibold text-white">Inputs</h2>
          <div className="mt-5 grid gap-4">
            {visible.map((field) => {
              if (field === "uploaded_documents") {
                return <label key={field} className="rounded-2xl border border-dashed border-cyan-300/35 bg-cyan-300/8 p-4 text-sm text-cyan-100"><span className="mb-2 flex items-center gap-2 font-semibold"><UploadCloud size={17} /> Upload evidence files</span><input type="file" multiple accept=".txt,.md,.json,.jsonl,.pdf" onChange={(e) => setFiles(Array.from(e.target.files ?? []))} className="block w-full text-sm file:mr-3 file:rounded-lg file:border-0 file:bg-cyan-300 file:px-3 file:py-2 file:text-slate-950" />{files.length > 0 && <p className="mt-2 text-xs text-slate-400">{files.length} file(s) selected</p>}</label>;
              }
              const spec = fields[field];
              const formKey = keyMap[field];
              if (!spec || !formKey) return null;
              const requiredBy = state.methods.filter((m) => selected.has(m.name) && m.required_fields.includes(field)).map((m) => m.name);
              return (
                <label key={field} className="block">
                  <span className="mb-2 flex flex-wrap items-center gap-2 text-sm font-semibold text-slate-100">{spec.label}{requiredBy.length > 0 && <span className="rounded-full bg-rose-300/15 px-2 py-1 text-[11px] text-rose-100">Required</span>}<span className="rounded-full bg-white/8 px-2 py-1 text-[11px] text-slate-400">{requiredBy.length ? requiredBy.join(", ") : "Optional"}</span></span>
                  <textarea value={state[formKey]} onChange={(e) => setField(formKey, e.target.value)} placeholder={spec.placeholder} className="min-h-32 w-full rounded-2xl border border-white/10 bg-slate-950/60 p-4 text-sm outline-none focus:border-cyan-300/60" />
                </label>
              );
            })}
          </div>
          <div className="mt-5 flex flex-wrap gap-3">
            <button disabled={!state.backendOnline || state.loading} onClick={run} className="inline-flex items-center gap-2 rounded-2xl bg-cyan-300 px-5 py-3 text-sm font-semibold text-slate-950 disabled:opacity-50"><Zap size={17} /> Compare selected detectors</button>
            <button onClick={clear} className="inline-flex items-center gap-2 rounded-2xl bg-white/10 px-5 py-3 text-sm font-semibold text-white"><RotateCcw size={17} /> Clear form</button>
            <button onClick={loadSample} className="rounded-2xl bg-fuchsia-300/15 px-5 py-3 text-sm font-semibold text-fuchsia-100">Load sample</button>
          </div>
          {state.selectedResult && <div className="mt-5"><VisualResultCard result={state.selectedResult} state={state} /></div>}
        </div>
        <div className="glass rounded-3xl p-5">
          <h2 className="text-xl font-semibold text-white">Selection profile</h2>
          <div className="mt-4 space-y-3">
            {state.selectedMethods.map((name) => {
              const method = state.methods.find((item) => item.name === name);
              if (!method) return null;
              return <div key={name} className="rounded-2xl bg-white/5 p-4"><p className="font-semibold text-white">{name}</p><p className="mt-2 text-sm leading-6 text-slate-400">{method.best_for}</p><p className="mt-2 text-xs text-slate-500">Visible: {method.visible_fields.join(", ")}</p></div>;
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
