"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { DetectorResult } from "@/lib/types";

export function RiskChart({ results }: { results: DetectorResult[] }) {
  const data = results.map((result) => ({
    method: result.method_name.replace(" Check", "").replace(" Verification", ""),
    risk: result.risk_score ?? 0,
    confidence: result.confidence == null ? 0 : Math.round(result.confidence * 100)
  }));

  if (!results.length) {
    return <div className="grid h-72 place-items-center rounded-2xl border border-white/10 bg-white/5 text-sm text-slate-500">Run an analysis to populate comparison charts.</div>;
  }

  return (
    <div className="h-80 rounded-2xl border border-white/10 bg-slate-950/35 p-4">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>
          <CartesianGrid stroke="rgba(148,163,184,0.15)" vertical={false} />
          <XAxis dataKey="method" stroke="#94a3b8" tick={{ fontSize: 11 }} interval={0} angle={-14} textAnchor="end" height={68} />
          <YAxis stroke="#94a3b8" domain={[0, 100]} />
          <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid rgba(148,163,184,0.25)", borderRadius: 12 }} />
          <Bar dataKey="risk" fill="#22d3ee" radius={[8, 8, 0, 0]} />
          <Bar dataKey="confidence" fill="#f0abfc" radius={[8, 8, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
