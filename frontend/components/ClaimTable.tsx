import type { DetectorResult } from "@/lib/types";
import { RiskBadge } from "./RiskBadge";

export function ClaimTable({ result }: { result?: DetectorResult }) {
  const rows = result?.claim_findings ?? [];
  if (!rows.length) return <p className="rounded-2xl border border-white/10 bg-white/5 p-5 text-sm text-slate-500">No claim-level table returned for this method.</p>;
  return (
    <div className="overflow-auto rounded-2xl border border-white/10">
      <table className="w-full min-w-[780px] border-collapse text-left text-sm">
        <thead className="bg-white/8 text-xs uppercase tracking-[0.14em] text-slate-400">
          <tr><th className="p-4">Claim</th><th className="p-4">Status</th><th className="p-4">Score</th><th className="p-4">Best match</th><th className="p-4">Reason</th></tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index} className="border-t border-white/10 bg-slate-950/25 align-top">
              <td className="max-w-sm p-4 text-white">{String(row.claim ?? "")}</td>
              <td className="p-4"><RiskBadge label={String(row.status ?? "finding")} /></td>
              <td className="p-4 text-slate-300">{row.score == null ? "N/A" : String(row.score)}</td>
              <td className="max-w-sm p-4 text-slate-400">{String(row.best_match ?? "")}</td>
              <td className="max-w-md p-4 text-slate-400">{String(row.reason ?? "")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
