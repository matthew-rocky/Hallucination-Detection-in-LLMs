import type { DetectorResult } from "@/lib/types";

export function TraceTimeline({ result }: { result?: DetectorResult }) {
  const steps = result?.intermediate_steps ?? [];
  if (!steps.length) return <p className="rounded-2xl border border-white/10 bg-white/5 p-5 text-sm text-slate-500">No intermediate trace returned.</p>;
  return (
    <div className="space-y-3">
      {steps.map((step, index) => (
        <details key={index} className="rounded-2xl border border-white/10 bg-white/5 p-4" open={index < 2}>
          <summary className="cursor-pointer text-sm font-semibold text-white">{String(step.stage ?? `Step ${index + 1}`)}</summary>
          <pre className="mt-3 max-h-72 overflow-auto rounded-xl bg-slate-950/60 p-3 text-xs leading-5 text-slate-300">{JSON.stringify(step.output ?? step, null, 2)}</pre>
        </details>
      ))}
    </div>
  );
}
