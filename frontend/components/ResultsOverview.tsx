"use client";

import { AlertTriangle, CheckCircle2, GaugeCircle } from "lucide-react";
import type { DetectorResult } from "@/lib/types";

function riskClass(label: string) {
  const lowered = label.toLowerCase();
  if (lowered === "low") return "risk-low";
  if (lowered === "medium") return "risk-medium";
  if (lowered === "high") return "risk-high";
  return "risk-na";
}

function reasonBullets(result: DetectorResult) {
  const findings = result.claim_findings ?? [];
  if (findings.length) {
    const contradicted = findings.filter((item) => String(item.status) === "contradicted").length;
    const unsupported = findings.filter((item) => ["unsupported", "insufficient evidence", "unresolved"].includes(String(item.status))).length;
    const supported = findings.filter((item) => ["supported", "abstractly_supported", "weakly_supported", "verified"].includes(String(item.status))).length;
    return [
      contradicted ? `${contradicted} claim(s) are contradicted by available evidence.` : "",
      unsupported ? `${unsupported} claim(s) remain unsupported or unresolved.` : "",
      supported ? `${supported} claim(s) are supported or partially supported.` : ""
    ].filter(Boolean);
  }
  return [result.summary, result.explanation].filter(Boolean).slice(0, 3) as string[];
}

export function ResultsOverview({ result }: { result?: DetectorResult }) {
  if (!result) {
    return <div className="glass rounded-3xl p-6 text-sm text-slate-400">Run a method to see the primary hallucination risk card.</div>;
  }
  const bullets = reasonBullets(result);
  return (
    <div className="glass rounded-3xl p-6">
      <div className="mb-5 flex flex-wrap items-center gap-3">
        <span className="rounded-full bg-white/8 px-3 py-1 text-xs text-slate-200">{result.method_name}</span>
        <span className="rounded-full bg-cyan-300/10 px-3 py-1 text-xs text-cyan-100">{result.implementation_status}</span>
      </div>
      <div className="grid gap-4 lg:grid-cols-[1.15fr_0.85fr]">
        <div className={`rounded-3xl p-5 ${riskClass(String(result.risk_label))}`}>
          <div className="flex items-center gap-3">
            <GaugeCircle size={28} />
            <div>
              <p className="text-sm opacity-80">Hallucination Risk</p>
              <p className="text-4xl font-semibold">{result.risk_label}</p>
            </div>
          </div>
          <p className="mt-5 text-5xl font-semibold">{result.risk_score == null ? "N/A" : `${Math.round(result.risk_score)}%`}</p>
        </div>
        <div className="grid gap-3">
          <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
            <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Confidence</p>
            <p className="mt-2 text-2xl font-semibold text-white">{result.confidence == null ? "N/A" : `${Math.round(result.confidence * 100)}%`}</p>
          </div>
          <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
            <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Availability</p>
            <p className="mt-2 flex items-center gap-2 text-lg font-semibold text-white">
              {result.available ? <CheckCircle2 className="text-emerald-300" size={19} /> : <AlertTriangle className="text-amber-300" size={19} />}
              {result.available ? "Completed" : "Unavailable"}
            </p>
          </div>
        </div>
      </div>
      <p className="mt-5 rounded-2xl border border-white/10 bg-slate-950/35 p-4 text-sm leading-6 text-slate-300">{result.summary || result.explanation}</p>
      <div className="mt-5">
        <p className="mb-2 text-sm font-semibold text-white">Why this score</p>
        <ul className="space-y-2 text-sm text-slate-300">
          {bullets.map((item, index) => (
            <li key={`${item}-${index}`} className="rounded-xl bg-white/5 px-3 py-2">{item}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}
