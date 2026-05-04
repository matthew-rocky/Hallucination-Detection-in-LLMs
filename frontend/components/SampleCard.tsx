"use client";

import { Play } from "lucide-react";
import { normalizeRiskLabel } from "@/lib/risk";
import type { SampleCase } from "@/lib/types";
import { RiskBadge } from "./RiskBadge";

export function SampleCard({
  sample,
  onLoadAsk,
  onLoadCompareDetectors,
  onRun
}: {
  sample: SampleCase;
  onLoadAsk: () => void;
  onLoadCompareDetectors: () => void;
  onRun: () => void;
}) {
  return (
    <div className="glass-panel-ai rounded-2xl border border-white/10 bg-white/5 p-4 transition hover:border-cyan-300/35 hover:bg-white/8">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <RiskBadge label={normalizeRiskLabel(sample.risk_level)} />
        <span className="rounded-full bg-white/8 px-2 py-1 text-[11px] text-slate-300">{sample.method_targets[0]}</span>
      </div>
      <h3 className="text-base font-semibold text-white">{sample.title}</h3>
      <p className="mt-2 text-sm leading-6 text-slate-400">{sample.description}</p>
      <p className="mt-3 line-clamp-3 rounded-xl bg-slate-950/35 p-3 text-xs leading-5 text-slate-300">{sample.question_preview || sample.question}</p>
      <p className="mt-2 line-clamp-2 text-xs leading-5 text-slate-500">{sample.answer_preview || sample.answer}</p>
      <div className="mt-3 flex flex-wrap gap-1">
        {(sample.available_inputs ?? []).map((item) => <span key={item} className="rounded-full bg-white/8 px-2 py-1 text-[10px] text-slate-400">{item}</span>)}
      </div>
      <div className="mt-4 flex flex-wrap gap-2">
        <button onClick={onLoadAsk} className="rounded-xl bg-cyan-300 px-3 py-2 text-xs font-semibold text-slate-950">Load into ASK Quick Mode</button>
        <button onClick={onLoadCompareDetectors} className="rounded-xl bg-white/10 px-3 py-2 text-xs font-semibold text-slate-100">Load into Compare Detectors</button>
        <button onClick={onRun} className="inline-flex items-center gap-1 rounded-xl bg-emerald-300/15 px-3 py-2 text-xs font-semibold text-emerald-100">
          <Play size={13} /> Run now
        </button>
      </div>
    </div>
  );
}
