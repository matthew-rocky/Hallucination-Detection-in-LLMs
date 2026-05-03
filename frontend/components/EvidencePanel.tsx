"use client";

import { Quote } from "lucide-react";
import type { DetectorResult } from "@/lib/types";

function textOf(record: Record<string, unknown>, keys: string[]) {
  for (const key of keys) {
    const value = record[key];
    if (value != null && String(value).trim()) return String(value);
  }
  return "";
}

export function EvidencePanel({ result }: { result?: DetectorResult }) {
  const evidence = result?.evidence ?? [];
  const citations = result?.citations ?? [];
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div className="rounded-2xl border border-white/10 bg-white/5 p-5">
        <h3 className="mb-4 flex items-center gap-2 font-semibold text-white"><Quote size={18} /> Evidence used</h3>
        {evidence.length ? (
          <div className="space-y-3">
            {evidence.slice(0, 6).map((item, index) => (
              <div key={index} className="rounded-xl bg-slate-950/35 p-3">
                <p className="text-xs text-cyan-100">{textOf(item, ["title", "evidence_id", "source_type"]) || `Evidence ${index + 1}`}</p>
                <p className="mt-2 text-sm leading-6 text-slate-300">{textOf(item, ["content", "snippet", "text"])}</p>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-slate-500">{result?.evidence_used || "No evidence records returned."}</p>
        )}
      </div>
      <div className="rounded-2xl border border-white/10 bg-white/5 p-5">
        <h3 className="mb-4 font-semibold text-white">Citations</h3>
        {citations.length ? (
          <div className="space-y-3">
            {citations.slice(0, 6).map((item, index) => (
              <div key={index} className="rounded-xl bg-slate-950/35 p-3">
                <p className="text-xs text-fuchsia-100">{textOf(item, ["citation_id", "title"]) || `Citation ${index + 1}`}</p>
                <p className="mt-2 text-sm leading-6 text-slate-300">{textOf(item, ["snippet", "content", "text"])}</p>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-slate-500">No citation records returned.</p>
        )}
      </div>
    </div>
  );
}
