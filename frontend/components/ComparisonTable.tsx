"use client";

import type { DetectorResult, MethodInfo } from "@/lib/types";

export function ComparisonTable({ results, methods }: { results: DetectorResult[]; methods: MethodInfo[] }) {
  const methodByName = new Map(methods.map((method) => [method.name, method]));
  if (!results.length) {
    return <div className="rounded-2xl border border-white/10 bg-white/5 p-5 text-sm text-slate-500">No method results yet.</div>;
  }
  return (
    <div className="overflow-hidden rounded-2xl border border-white/10">
      <table className="w-full min-w-[780px] border-collapse text-left text-sm">
        <thead className="bg-white/8 text-xs uppercase tracking-[0.14em] text-slate-400">
          <tr>
            <th className="p-4">Method</th>
            <th className="p-4">Family</th>
            <th className="p-4">Risk</th>
            <th className="p-4">Confidence</th>
            <th className="p-4">Status</th>
            <th className="p-4">Reason</th>
          </tr>
        </thead>
        <tbody>
          {results.map((result) => (
            <tr key={result.method_name} className="border-t border-white/10 bg-slate-950/25 align-top">
              <td className="p-4 font-medium text-white">{result.method_name}</td>
              <td className="p-4 text-slate-300">{methodByName.get(result.method_name)?.family ?? result.family}</td>
              <td className="p-4 text-slate-200">{result.risk_score == null ? "N/A" : `${Math.round(result.risk_score)}%`} · {result.risk_label}</td>
              <td className="p-4 text-slate-300">{result.confidence == null ? "N/A" : `${Math.round(result.confidence * 100)}%`}</td>
              <td className="p-4"><span className="rounded-full bg-cyan-300/10 px-2 py-1 text-xs text-cyan-100">{result.implementation_status}</span></td>
              <td className="max-w-md p-4 text-slate-400">{result.summary || result.explanation}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
