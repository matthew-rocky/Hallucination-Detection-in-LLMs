"use client";

import { ChevronDown } from "lucide-react";
import type { DetectorResult } from "@/lib/types";

export function MethodTrace({ result }: { result?: DetectorResult }) {
  const steps = result?.intermediate_steps ?? [];
  if (!steps.length && !result?.revised_answer) {
    return <div className="rounded-2xl border border-white/10 bg-white/5 p-5 text-sm text-slate-500">No intermediate trace returned for the selected result.</div>;
  }
  return (
    <div className="space-y-3">
      {steps.map((step, index) => (
        <details key={index} className="group rounded-2xl border border-white/10 bg-white/5 p-4" open={index === 0}>
          <summary className="flex cursor-pointer list-none items-center justify-between gap-3 text-sm font-medium text-white">
            {String(step.stage ?? `Step ${index + 1}`)}
            <ChevronDown className="transition group-open:rotate-180" size={18} />
          </summary>
          <pre className="mt-3 max-h-72 overflow-auto rounded-xl bg-slate-950/60 p-3 text-xs leading-5 text-slate-300">{JSON.stringify(step.output ?? step, null, 2)}</pre>
        </details>
      ))}
      {result?.revised_answer && (
        <div className="rounded-2xl border border-emerald-300/20 bg-emerald-300/8 p-4">
          <p className="text-sm font-semibold text-emerald-100">Revised answer</p>
          <p className="mt-2 text-sm leading-6 text-slate-300">{result.revised_answer}</p>
        </div>
      )}
    </div>
  );
}
