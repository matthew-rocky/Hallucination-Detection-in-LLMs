"use client";

import type { LucideIcon } from "lucide-react";
import { BarChart3, Bot, Clipboard, Download, FileJson, FileText, Gauge, ListChecks, RefreshCcw, ShieldAlert, ShieldCheck, Table2, TrendingUp } from "lucide-react";
import { normalizeRiskScore, riskTone } from "@/lib/risk";
import type { DashboardState, DetectorResult, TabId } from "@/lib/types";
import { ComparisonTable } from "./ComparisonTable";
import { EmptyState } from "./EmptyState";
import { RiskBadge } from "./RiskBadge";
import { RiskChart } from "./RiskChart";

function scoreText(score?: number | null) {
  return score == null ? "N/A" : `${Math.round(normalizeRiskScore(score))}`;
}

function confidenceText(confidence?: number | null) {
  return confidence == null ? "N/A" : `${Math.round(confidence * 100)}%`;
}

function compactText(value?: string | null, fallback = "No summary returned.") {
  const text = String(value || "").trim();
  return text || fallback;
}

function sortedResults(results: DetectorResult[]) {
  return [...results].sort((a, b) => (b.risk_score ?? -1) - (a.risk_score ?? -1));
}

function averageRisk(results: DetectorResult[], fallback?: number) {
  if (typeof fallback === "number") return Math.round(normalizeRiskScore(fallback));
  const scored = results.filter((result) => result.risk_score != null);
  if (!scored.length) return 0;
  return Math.round(scored.reduce((sum, result) => sum + normalizeRiskScore(result.risk_score), 0) / scored.length);
}

function claimStatus(claim: Record<string, unknown>) {
  return String(claim.verdict ?? claim.status ?? claim.label ?? claim.assessment ?? "Unspecified");
}

function claimText(claim: Record<string, unknown>) {
  return String(claim.claim ?? claim.statement ?? claim.text ?? claim.question ?? "Claim finding returned");
}

function buildStats(state: DashboardState) {
  const ranked = sortedResults(state.results);
  const highest = ranked[0];
  const lowest = ranked[ranked.length - 1];
  const avgRisk = averageRisk(state.results, state.summary?.avg_risk);
  const claims = state.results.flatMap((result) => (result.claim_findings ?? []).map((claim) => ({ method: result.method_name, claim })));
  const claimCounts = claims.reduce<Record<string, number>>((counts, item) => {
    const status = claimStatus(item.claim);
    counts[status] = (counts[status] ?? 0) + 1;
    return counts;
  }, {});
  const evidenceCount = state.results.reduce((sum, result) => sum + (result.evidence?.length ?? 0), 0);
  const citationCount = state.results.reduce((sum, result) => sum + (result.citations?.length ?? 0), 0);
  const spread = highest && lowest ? normalizeRiskScore(highest.risk_score) - normalizeRiskScore(lowest.risk_score) : 0;
  const riskLabels = Array.from(new Set(state.results.map((result) => result.risk_label).filter(Boolean)));
  return { ranked, highest, lowest, avgRisk, claims, claimCounts, evidenceCount, citationCount, spread, riskLabels };
}

function finalRecommendation(highest?: DetectorResult) {
  if (!highest) return "Run a detector before making a final review decision.";
  const score = normalizeRiskScore(highest.risk_score);
  if (highest.risk_label === "High" || score >= 70) {
    return `Prioritize manual review of ${highest.method_name}. Treat the answer as high risk until the returned findings are resolved or independently verified.`;
  }
  if (highest.risk_label === "Medium" || score >= 40) {
    return `Review the evidence and claim findings from ${highest.method_name} before relying on the answer.`;
  }
  return "Returned detector outputs indicate lower risk. Keep source-sensitive claims tied to evidence before export.";
}

function reportMarkdown(state: DashboardState) {
  const { ranked, highest, lowest, avgRisk, claimCounts, evidenceCount, citationCount, spread, riskLabels } = buildStats(state);
  const multiple = state.results.length > 1;
  const lines = [
    "# Hallucination Detection Report",
    "",
    `Mode: ${state.mode === "compare" ? "Compare Detectors" : "ASK Quick Mode"}`,
    `Detectors compared: ${state.results.length}`,
    `Selected methods: ${state.selectedMethods.join(", ") || "N/A"}`,
    `Question: ${state.question || "N/A"}`,
    "",
    "## Executive Summary",
    `Highest-risk detector: ${highest ? `${highest.method_name} (${highest.risk_label}, ${scoreText(highest.risk_score)})` : "N/A"}`,
    `Lowest-risk detector: ${lowest ? `${lowest.method_name} (${lowest.risk_label}, ${scoreText(lowest.risk_score)})` : "N/A"}`,
    `Average risk score: ${avgRisk}`,
    `Risk distribution: Low ${state.summary?.low ?? 0}, Medium ${state.summary?.medium ?? 0}, High ${state.summary?.high ?? 0}`,
    "",
    "## Key Disagreements",
    multiple
      ? `Risk labels returned: ${riskLabels.join(", ") || "N/A"}. Risk score spread: ${Math.round(spread)} points.`
      : "Only one detector returned, so there is no cross-detector disagreement to compare.",
    "",
    "## Claim Findings Summary",
    Object.keys(claimCounts).length
      ? Object.entries(claimCounts).map(([status, count]) => `- ${status}: ${count}`).join("\n")
      : "No claim findings were returned.",
    "",
    "## Evidence and Citations",
    `Evidence records returned: ${evidenceCount}`,
    `Citations returned: ${citationCount}`,
    "",
    "## Method Results",
    ...ranked.map((r) => `### ${r.method_name}
- Family: ${r.family}
- Risk: ${r.risk_label} (${scoreText(r.risk_score)})
- Confidence: ${confidenceText(r.confidence)}
- Summary: ${compactText(r.summary || r.explanation)}
- Evidence used: ${compactText(r.evidence_used, "N/A")}
- Limitations: ${compactText(r.limitations, "N/A")}${r.revised_answer ? `\n- Revised answer: ${r.revised_answer}` : ""}`),
    "",
    "## Final Review Recommendation",
    finalRecommendation(highest)
  ];
  return lines.join("\n");
}

function download(name: string, text: string, type: string) {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
}

function SummaryCard({ title, value, detail, icon: Icon, result }: { title: string; value: string | number; detail?: string; icon: LucideIcon; result?: DetectorResult }) {
  const tone = riskTone(result?.risk_label);
  return (
    <div className={`rounded-3xl bg-gradient-to-br ${tone.border} p-px ${tone.glow}`}>
      <div className="h-full rounded-3xl border border-white/10 bg-slate-950/75 p-5 backdrop-blur-xl">
        <p className="flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-cyan-200"><Icon size={14} /> {title}</p>
        <p className="mt-4 text-3xl font-semibold text-white">{value}</p>
        {detail && <p className="mt-2 text-sm leading-6 text-slate-400">{detail}</p>}
      </div>
    </div>
  );
}

function MethodResultCard({ result }: { result: DetectorResult }) {
  const tone = riskTone(result.risk_label);
  const score = normalizeRiskScore(result.risk_score);
  return (
    <div className={`rounded-3xl bg-gradient-to-br ${tone.border} p-px`}>
      <div className="h-full rounded-3xl bg-slate-950/75 p-5 backdrop-blur-xl">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h3 className="text-base font-semibold text-white">{result.method_name}</h3>
            <p className="mt-1 text-xs text-slate-500">{result.family}</p>
          </div>
          <RiskBadge label={result.risk_label} />
        </div>
        <div className="mt-4 h-2 rounded-full bg-white/10">
          <div className={`h-full rounded-full ${tone.fill}`} style={{ width: `${score}%` }} />
        </div>
        <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
          <div className="rounded-2xl bg-white/5 p-3">
            <p className="text-xs uppercase tracking-[0.14em] text-slate-500">Risk</p>
            <p className="mt-1 font-semibold text-white">{scoreText(result.risk_score)}</p>
          </div>
          <div className="rounded-2xl bg-white/5 p-3">
            <p className="text-xs uppercase tracking-[0.14em] text-slate-500">Confidence</p>
            <p className="mt-1 font-semibold text-white">{confidenceText(result.confidence)}</p>
          </div>
        </div>
        <p className="mt-4 text-sm leading-6 text-slate-300">{compactText(result.summary || result.explanation)}</p>
        {result.evidence_used && <p className="mt-3 text-xs leading-5 text-slate-500">Evidence: {result.evidence_used}</p>}
      </div>
    </div>
  );
}

export function ReportPage({ state, setTab }: { state: DashboardState; setTab: (tab: TabId) => void }) {
  if (!state.results.length) {
    return (
      <EmptyState
        icon={Clipboard}
        title="No report available"
        message="Run ASK Quick Mode for a single-method review or Compare Detectors for a multi-method report."
        actions={
          <>
            <button onClick={() => setTab("askQuick")} className="rounded-2xl bg-cyan-300 px-4 py-3 text-sm font-semibold text-slate-950">Go to ASK Quick Mode</button>
            <button onClick={() => setTab("compareDetectors")} className="rounded-2xl bg-fuchsia-300/15 px-4 py-3 text-sm font-semibold text-fuchsia-100">Go to Compare Detectors</button>
          </>
        }
      />
    );
  }

  const md = reportMarkdown(state);
  const { ranked, highest, lowest, avgRisk, claims, claimCounts, evidenceCount, citationCount, spread, riskLabels } = buildStats(state);
  const multiple = state.results.length > 1;
  const exportJson = JSON.stringify({
    mode: state.mode,
    selected_methods: state.selectedMethods,
    question: state.question,
    answer: state.answer,
    summary: state.summary,
    results: state.results
  }, null, 2);

  return (
    <div className="space-y-5">
      <div className="glass rounded-3xl p-5">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold text-white">Report / Export</h1>
            <p className="mt-1 text-sm text-slate-400">{multiple ? "Full comparison report from the latest real detector outputs." : "Single-method report from the latest real detector output."}</p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button onClick={() => navigator.clipboard.writeText(md)} className="inline-flex items-center gap-2 rounded-2xl bg-cyan-300 px-4 py-3 text-sm font-semibold text-slate-950"><Clipboard size={16} /> Copy Markdown</button>
            <button onClick={() => download("hallucination-report.json", exportJson, "application/json")} className="inline-flex items-center gap-2 rounded-2xl bg-white/10 px-4 py-3 text-sm font-semibold text-white"><FileJson size={16} /> Download JSON</button>
            <button onClick={() => download("hallucination-report.md", md, "text/markdown")} className="inline-flex items-center gap-2 rounded-2xl bg-fuchsia-300/15 px-4 py-3 text-sm font-semibold text-fuchsia-100"><Download size={16} /> Download Markdown</button>
          </div>
        </div>
      </div>

      <section className="glass glass-panel-ai rounded-3xl p-6">
        <p className="text-xs uppercase tracking-[0.22em] text-cyan-200">Executive summary</p>
        <h2 className="mt-3 text-2xl font-semibold text-white">{multiple ? `${state.results.length} detectors compared` : `${ranked[0]?.method_name} result`}</h2>
        <p className="mt-3 max-w-4xl text-sm leading-7 text-slate-300">
          {multiple
            ? `The latest comparison returned ${state.results.length} detector outputs. ${highest?.method_name ?? "N/A"} has the highest returned risk, and ${lowest?.method_name ?? "N/A"} has the lowest returned risk.`
            : compactText(ranked[0]?.summary || ranked[0]?.explanation)}
        </p>
      </section>

      <div className="grid gap-4 lg:grid-cols-4">
        <SummaryCard title="Detectors" value={state.results.length} detail={multiple ? "Compared in latest run" : "Single selected detector"} icon={Gauge} />
        <SummaryCard title="Highest risk" value={highest ? scoreText(highest.risk_score) : "N/A"} detail={highest ? `${highest.method_name} - ${highest.risk_label}` : "N/A"} icon={ShieldAlert} result={highest} />
        <SummaryCard title="Lowest risk" value={lowest ? scoreText(lowest.risk_score) : "N/A"} detail={lowest ? `${lowest.method_name} - ${lowest.risk_label}` : "N/A"} icon={ShieldCheck} result={lowest} />
        <SummaryCard title="Average risk" value={avgRisk} detail={`Low ${state.summary?.low ?? 0} / Medium ${state.summary?.medium ?? 0} / High ${state.summary?.high ?? 0}`} icon={TrendingUp} />
      </div>

      <div className="glass rounded-3xl p-5">
        <h2 className="mb-4 flex items-center gap-2 text-xl font-semibold text-white"><BarChart3 className="text-cyan-100" size={20} /> Risk distribution</h2>
        <RiskChart results={state.results} />
      </div>

      <section className="space-y-4">
        <h2 className="flex items-center gap-2 text-xl font-semibold text-white"><FileText className="text-cyan-100" size={20} /> Method-by-method results</h2>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {ranked.map((result) => <MethodResultCard key={result.method_name} result={result} />)}
        </div>
      </section>

      <div className="glass rounded-3xl p-5">
        <h2 className="mb-4 flex items-center gap-2 text-xl font-semibold text-white"><Table2 className="text-cyan-100" size={20} /> Comparison table</h2>
        <ComparisonTable results={state.results} methods={state.methods} />
      </div>

      <div className="grid gap-5 xl:grid-cols-2">
        <section className="glass rounded-3xl p-5">
          <h2 className="text-xl font-semibold text-white">Key disagreements</h2>
          <div className="mt-4 space-y-3 text-sm leading-7 text-slate-300">
            {multiple ? (
              <>
                <p><b className="text-white">Risk labels:</b> {riskLabels.join(", ") || "N/A"}</p>
                <p><b className="text-white">Risk spread:</b> {Math.round(spread)} points between {highest?.method_name ?? "N/A"} and {lowest?.method_name ?? "N/A"}.</p>
                {riskLabels.length > 1 ? <p>Detectors returned different risk labels, so review the method explanations before relying on one score.</p> : <p>Returned detectors share the same risk label; compare explanations and evidence for smaller differences.</p>}
              </>
            ) : (
              <p>Only one detector returned, so there is no cross-detector disagreement to compare.</p>
            )}
          </div>
        </section>

        <section className="glass rounded-3xl p-5">
          <h2 className="flex items-center gap-2 text-xl font-semibold text-white"><ListChecks className="text-cyan-100" size={20} /> Claim findings summary</h2>
          <div className="mt-4 space-y-3 text-sm leading-7 text-slate-300">
            <p><b className="text-white">Total claim findings:</b> {claims.length}</p>
            {Object.keys(claimCounts).length ? (
              <div className="flex flex-wrap gap-2">
                {Object.entries(claimCounts).map(([status, count]) => <span key={status} className="rounded-full bg-white/8 px-3 py-2 text-xs text-slate-200">{status}: {count}</span>)}
              </div>
            ) : <p>No claim findings were returned by the selected detector methods.</p>}
            {claims.slice(0, 3).map((item, index) => (
              <p key={`${item.method}-${index}`} className="rounded-2xl bg-white/5 p-3 text-xs leading-6 text-slate-400"><b className="text-slate-200">{item.method}:</b> {claimText(item.claim)}</p>
            ))}
          </div>
        </section>
      </div>

      <section className="glass rounded-3xl p-5">
        <h2 className="text-xl font-semibold text-white">Evidence and citation summary</h2>
        <div className="mt-4 grid gap-4 md:grid-cols-3">
          <div className="rounded-2xl bg-white/5 p-4"><p className="text-xs uppercase tracking-[0.16em] text-slate-500">Evidence records</p><p className="mt-2 text-2xl font-semibold text-white">{evidenceCount}</p></div>
          <div className="rounded-2xl bg-white/5 p-4"><p className="text-xs uppercase tracking-[0.16em] text-slate-500">Citations</p><p className="mt-2 text-2xl font-semibold text-white">{citationCount}</p></div>
          <div className="rounded-2xl bg-white/5 p-4"><p className="text-xs uppercase tracking-[0.16em] text-slate-500">Claims checked</p><p className="mt-2 text-2xl font-semibold text-white">{state.summary?.claims_checked ?? claims.length}</p></div>
        </div>
        <div className="mt-4 grid gap-3 md:grid-cols-2">
          {state.results.filter((result) => result.evidence_used).map((result) => (
            <p key={result.method_name} className="rounded-2xl bg-slate-950/35 p-3 text-xs leading-6 text-slate-400"><b className="text-slate-200">{result.method_name}:</b> {result.evidence_used}</p>
          ))}
        </div>
      </section>

      <section className="glass rounded-3xl p-5">
        <h2 className="text-xl font-semibold text-white">Final review recommendation</h2>
        <p className="mt-3 text-sm leading-7 text-slate-300">{finalRecommendation(highest)}</p>
      </section>

      <div className="glass rounded-3xl p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-xl font-semibold text-white">Next step</h2>
            <p className="mt-1 text-sm text-slate-400">Run another detector comparison without clearing the current form or results.</p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button onClick={() => setTab("compareDetectors")} className="inline-flex items-center gap-2 rounded-2xl bg-cyan-300 px-5 py-3 text-sm font-semibold text-slate-950"><RefreshCcw size={16} /> Compare again</button>
            <button onClick={() => setTab("askQuick")} className="inline-flex items-center gap-2 rounded-2xl bg-white/10 px-4 py-3 text-sm font-semibold text-white"><Bot size={16} /> Back to ASK Quick Mode</button>
          </div>
        </div>
      </div>
    </div>
  );
}
