"use client";

import type { DetectorResult } from "@/lib/types";

export function ClaimFindings({ result }: { result?: DetectorResult }) {
  const findings = result?.claim_findings ?? [];
  if (!findings.length) {
    return <div className="rounded-2xl border border-white/10 bg-white/5 p-5 text-sm text-slate-500">No claim-level findings returned for the selected result.</div>;
  }
  return (
    <div className="overflow-hidden rounded-2xl border border-white/10">
      <table className="w-full min-w-[760px] border-collapse text-left text-sm">
        <thead className="bg-white/8 text-xs uppercase tracking-[0.14em] text-slate-400">
          <tr>
            <th className="p-4">Claim</th>
            <th className="p-4">Status</th>
            <th className="p-4">Score</th>
            <th className="p-4">Best match</th>
            <th className="p-4">Reason</th>
          </tr>
        </thead>
        <tbody>
          {findings.map((finding, index) => (
            <tr key={index} className="border-t border-white/10 bg-slate-950/25 align-top">
              <td className="max-w-sm p-4 text-white">{String(finding.claim ?? "")}</td>
              <td className="p-4 text-slate-200">{String(finding.status ?? "")}</td>
              <td className="p-4 text-slate-300">{finding.score == null ? "N/A" : String(finding.score)}</td>
              <td className="max-w-sm p-4 text-slate-400">{String(finding.best_match ?? "")}</td>
              <td className="max-w-md p-4 text-slate-400">{String(finding.reason ?? "")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
