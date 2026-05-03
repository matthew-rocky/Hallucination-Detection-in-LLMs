"use client";

import { BookOpen } from "lucide-react";
import type { SampleCase } from "@/lib/types";
import type { FormState } from "./InputPanel";

export function SampleBrowser({
  samples,
  selectedMethods,
  onLoad
}: {
  samples: SampleCase[];
  selectedMethods: string[];
  onLoad: (form: FormState) => void;
}) {
  const visible = samples.filter((sample) => !selectedMethods.length || sample.method_targets.some((target) => selectedMethods.includes(target)));
  return (
    <div className="glass rounded-3xl p-5">
      <div className="mb-5 flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.22em] text-fuchsia-200">Samples</p>
          <h2 className="mt-1 text-xl font-semibold text-white">Curated cases</h2>
        </div>
        <BookOpen className="text-fuchsia-100" />
      </div>
      <div className="grid max-h-[640px] gap-3 overflow-auto pr-1">
        {visible.map((sample) => (
          <button
            key={sample.id}
            onClick={() =>
              onLoad({
                question: sample.question,
                answer: sample.answer,
                source_text: sample.source_text,
                evidence_text: sample.evidence_text,
                sampled_answers_text: sample.answer_samples
              })
            }
            className="rounded-2xl border border-white/10 bg-white/5 p-4 text-left transition hover:border-fuchsia-200/40 hover:bg-fuchsia-300/8"
          >
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <span className={sample.risk_level === "low" ? "risk-low rounded-full px-2 py-1 text-[11px]" : "risk-high rounded-full px-2 py-1 text-[11px]"}>
                {sample.risk_level} risk
              </span>
              <span className="rounded-full bg-white/8 px-2 py-1 text-[11px] text-slate-300">{sample.method_targets[0]}</span>
            </div>
            <h3 className="text-sm font-semibold text-white">{sample.title}</h3>
            <p className="mt-2 text-xs leading-5 text-slate-400">{sample.description}</p>
          </button>
        ))}
      </div>
    </div>
  );
}
