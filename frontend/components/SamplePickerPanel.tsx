"use client";

import { motion } from "framer-motion";
import { BookOpen, Database, FileText, Layers3, Play, Server, ShieldCheck, ShieldX, X } from "lucide-react";
import { normalizeRiskLabel } from "@/lib/risk";
import type { MethodInfo, SampleCase } from "@/lib/types";
import { RiskBadge } from "./RiskBadge";

function preview(text?: string) {
  const compact = (text ?? "").trim().replace(/\s+/g, " ");
  return compact.length > 150 ? `${compact.slice(0, 149)}...` : compact;
}

function PickerCard({ sample, methods, onLoad, onRun }: { sample: SampleCase; methods: MethodInfo[]; onLoad: () => void; onRun: () => void }) {
  const methodName = sample.recommended_methods?.[0] ?? sample.method_targets?.[0] ?? "Recommended detector";
  const family = methods.find((method) => method.name === methodName)?.family;
  const inputs = [
    { label: "Source", active: Boolean(sample.source_text?.trim()), icon: FileText },
    { label: "Evidence", active: Boolean(sample.evidence_text?.trim()), icon: Database },
    { label: "Samples", active: Boolean((sample.answer_samples || sample.sampled_answers_text)?.trim()), icon: Layers3 }
  ];
  return (
    <motion.div whileHover={{ y: -2 }} className="glass-panel-ai rounded-3xl border border-white/10 bg-white/[0.055] p-4 shadow-2xl backdrop-blur-xl transition hover:border-cyan-300/35 hover:bg-white/[0.075]">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <RiskBadge label={normalizeRiskLabel(sample.risk_level)} />
          <h4 className="mt-3 text-base font-semibold text-white">{sample.title}</h4>
        </div>
        <span className="rounded-full bg-white/8 px-2 py-1 text-[11px] text-slate-300">{methodName}</span>
      </div>
      {family && <p className="mt-2 text-xs text-cyan-100">{family}</p>}
      <p className="mt-3 rounded-2xl bg-slate-950/45 p-3 text-xs leading-5 text-slate-300">{preview(sample.question)}</p>
      <p className="mt-2 line-clamp-2 text-xs leading-5 text-slate-500">{preview(sample.answer)}</p>
      <div className="mt-3 flex flex-wrap gap-2">
        {inputs.map(({ label, active, icon: Icon }) => <span key={label} className={`inline-flex items-center gap-1 rounded-full px-2 py-1 text-[11px] ${active ? "bg-cyan-300/10 text-cyan-100" : "bg-white/6 text-slate-500"}`}><Icon size={12} /> {label}</span>)}
      </div>
      <div className="mt-4 flex flex-wrap gap-2">
        <button onClick={onLoad} className="rounded-xl bg-cyan-300 px-3 py-2 text-xs font-semibold text-slate-950">Load sample</button>
        <button onClick={onRun} className="inline-flex items-center gap-1 rounded-xl bg-emerald-300/15 px-3 py-2 text-xs font-semibold text-emerald-100"><Play size={13} /> Run sample</button>
      </div>
    </motion.div>
  );
}

export function SamplePickerPanel({
  open,
  samples,
  methods,
  selectedMethods,
  onLoad,
  onRun,
  onClose
}: {
  open: boolean;
  samples: SampleCase[];
  methods: MethodInfo[];
  selectedMethods: string[];
  onLoad: (sample: SampleCase) => void;
  onRun: (sample: SampleCase) => void;
  onClose: () => void;
}) {
  if (!open) return null;
  const selected = new Set(selectedMethods);
  const methodSpecific = samples.filter((sample) => sample.method_targets?.some((method) => selected.has(method)) || sample.recommended_methods?.some((method) => selected.has(method)));
  const visible = methodSpecific.length ? methodSpecific : samples;
  const low = visible.filter((sample) => normalizeRiskLabel(sample.risk_level) === "Low");
  const high = visible.filter((sample) => normalizeRiskLabel(sample.risk_level) === "High");

  if (!samples.length) {
    return (
      <div className="fixed inset-0 z-50 grid place-items-center bg-slate-950/70 p-4 backdrop-blur-xl">
        <div className="glass-panel-ai w-full max-w-3xl rounded-3xl border border-amber-300/25 bg-slate-950/90 p-5 text-amber-100 shadow-2xl">
          <div className="mb-4 flex justify-end"><button onClick={onClose} className="rounded-full bg-white/10 p-2 text-white hover:bg-white/15"><X size={18} /></button></div>
          <div className="flex gap-3"><Server className="mt-1 shrink-0" /><div><p className="font-semibold">Samples are unavailable</p><p className="mt-2 text-sm leading-6">Start the FastAPI backend to load curated low-risk and high-risk examples.</p><code className="mt-3 block rounded-xl bg-slate-950/70 p-3 text-xs text-cyan-100">python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000</code></div></div>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/70 p-4 backdrop-blur-xl md:p-8">
      <motion.section initial={{ opacity: 0, y: 18, scale: 0.985 }} animate={{ opacity: 1, y: 0, scale: 1 }} className="glass-panel-ai ai-scan-bg mx-auto flex max-h-[calc(100vh-4rem)] w-full max-w-7xl flex-col overflow-hidden rounded-[2rem] border border-white/10 bg-slate-950/[0.88] p-5 shadow-2xl backdrop-blur-2xl">
        <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
          <div><p className="flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-cyan-200"><BookOpen size={14} /> Sample picker</p><h3 className="mt-1 text-xl font-semibold text-white">Choose a curated detector case</h3></div>
          <div className="flex items-center gap-3"><p className="text-xs text-slate-500">{methodSpecific.length ? "Filtered to selected method" : "Showing all methods"}</p><button onClick={onClose} className="rounded-full bg-white/10 p-2 text-white hover:bg-white/15"><X size={18} /></button></div>
        </div>
        <div className="min-h-0 flex-1 overflow-auto pr-1">
          <div className="grid gap-5 2xl:grid-cols-2">
            <div>
              <h4 className="mb-3 flex items-center gap-2 text-sm font-semibold text-emerald-100"><ShieldCheck size={16} /> LOW-RISK SAMPLES</h4>
              <div className="grid gap-3 xl:grid-cols-2">{low.map((sample) => <PickerCard key={sample.id} sample={sample} methods={methods} onLoad={() => onLoad(sample)} onRun={() => onRun(sample)} />)}</div>
              {!low.length && <p className="rounded-2xl bg-white/5 p-4 text-sm text-slate-500">No low-risk samples match the current method filter.</p>}
            </div>
            <div>
              <h4 className="mb-3 flex items-center gap-2 text-sm font-semibold text-rose-100"><ShieldX size={16} /> HIGH-RISK SAMPLES</h4>
              <div className="grid gap-3 xl:grid-cols-2">{high.map((sample) => <PickerCard key={sample.id} sample={sample} methods={methods} onLoad={() => onLoad(sample)} onRun={() => onRun(sample)} />)}</div>
              {!high.length && <p className="rounded-2xl bg-white/5 p-4 text-sm text-slate-500">No high-risk samples match the current method filter.</p>}
            </div>
          </div>
        </div>
      </motion.section>
    </div>
  );
}
