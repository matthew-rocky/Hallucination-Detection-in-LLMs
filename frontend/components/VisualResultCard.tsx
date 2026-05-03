"use client";

import { motion } from "framer-motion";
import {
  Activity,
  AlertTriangle,
  BadgeCheck,
  BookOpenCheck,
  BrainCircuit,
  Braces,
  CheckCircle2,
  Clock3,
  Database,
  FileSearch,
  Gauge,
  GitBranch,
  Info,
  Network,
  Quote,
  Radar,
  ScanSearch,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Wrench
} from "lucide-react";
import { getRiskTheme, normalizeRiskScore, riskLabelFromScore } from "@/lib/risk";
import type { DashboardState, DetectorResult } from "@/lib/types";
import { RiskBadge } from "./RiskBadge";

type RecordValue = Record<string, unknown>;
type RiskTheme = ReturnType<typeof getRiskTheme>;

function textValue(value: unknown, fallback = "Not returned") {
  if (value == null || value === "") return fallback;
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return String(value);
  return JSON.stringify(value);
}

function short(text: unknown, limit = 280) {
  const compact = textValue(text, "").replace(/\s+/g, " ").trim();
  if (!compact) return "Not returned";
  return compact.length > limit ? `${compact.slice(0, limit - 1)}...` : compact;
}

function readableRecord(record: RecordValue, preferredKeys: string[], limit = 260) {
  for (const key of preferredKeys) {
    const value = record[key];
    if (value != null && value !== "") return short(value, limit);
  }
  const parts = Object.entries(record)
    .filter(([, value]) => value != null && value !== "")
    .slice(0, 3)
    .map(([key, value]) => `${key}: ${textValue(value, "")}`);
  return short(parts.join(" | "), limit);
}

function statusLabel(claim: RecordValue) {
  return textValue(claim.status ?? claim.verdict ?? claim.label ?? claim.support_label, "Unlabeled");
}

function wordCount(text: string) {
  return text.trim() ? text.trim().split(/\s+/).length : 0;
}

function claimScore(claim: RecordValue) {
  const raw = claim.score ?? claim.risk_score ?? claim.confidence;
  if (typeof raw === "number") return normalizeRiskScore(raw);
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? normalizeRiskScore(parsed) : null;
}

function statusVisual(status: string) {
  const value = status.toLowerCase();
  if (/support|verified|low_risk|low|pass/.test(value)) {
    return { icon: CheckCircle2, cls: "border-emerald-300/30 bg-emerald-400/12 text-emerald-100", accent: "border-l-emerald-300", glow: "hover:shadow-[0_0_34px_rgba(16,185,129,0.18)]" };
  }
  if (/contradict|high_risk|high|fail|false|error/.test(value)) {
    return { icon: AlertTriangle, cls: "border-rose-300/35 bg-rose-500/12 text-rose-100", accent: "border-l-rose-400", glow: "hover:shadow-[0_0_38px_rgba(244,63,94,0.22)]" };
  }
  if (/mixed|weakly_supported|weak|unclear|medium_risk|medium|partial/.test(value)) {
    return { icon: Info, cls: "border-amber-300/35 bg-amber-400/14 text-yellow-100", accent: "border-l-amber-300", glow: "hover:shadow-[0_0_34px_rgba(251,191,36,0.20)]" };
  }
  return { icon: FileSearch, cls: "border-violet-300/25 bg-violet-400/12 text-violet-100", accent: "border-l-violet-300", glow: "hover:shadow-[0_0_30px_rgba(167,139,250,0.18)]" };
}

function HeaderIcon({ theme }: { theme: RiskTheme }) {
  const Icon = theme.level === "high" ? ShieldAlert : theme.level === "medium" ? Radar : ShieldCheck;
  return (
    <motion.div
      initial={{ scale: 0.86, opacity: 0, rotate: -6 }}
      animate={{ scale: 1, opacity: 1, rotate: 0 }}
      transition={{ duration: 0.48, ease: "easeOut" }}
      className={`relative grid h-16 w-16 shrink-0 place-items-center rounded-3xl ${theme.iconBg} ${theme.strongText} ring-1 ${theme.ring} ${theme.level === "high" ? "animate-pulse" : ""}`}
    >
      <div className={`absolute inset-0 rounded-3xl blur-xl ${theme.aura}`} />
      <Icon className="relative drop-shadow-2xl" size={31} />
    </motion.div>
  );
}

function RiskGauge({ score, theme }: { score: number; theme: RiskTheme }) {
  const radius = 78;
  const circumference = 2 * Math.PI * radius;
  const dash = circumference * (score / 100);
  const markerAngle = (score / 100) * 360 - 90;
  const markerX = 110 + radius * Math.cos((markerAngle * Math.PI) / 180);
  const markerY = 110 + radius * Math.sin((markerAngle * Math.PI) / 180);

  return (
    <div className={`glass-panel-ai ai-scan-bg ai-risk-bg-${theme.level} relative mx-auto overflow-hidden rounded-[2rem] border ${theme.border} bg-[radial-gradient(circle_at_center,rgba(255,255,255,0.07),rgba(15,23,42,0.20)_45%,rgba(15,23,42,0.64))] p-6 shadow-2xl`}>
      <div className={`ai-decor h-64 w-64 rounded-full blur-3xl ${theme.aura} ${theme.level === "high" ? "animate-pulse" : ""}`} />
      <motion.div aria-hidden animate={{ opacity: [0.15, 0.55, 0.15], scale: [0.85, 1.08, 0.85] }} transition={{ duration: 4, repeat: Infinity }} className={`ai-decor h-32 w-32 rounded-full ${theme.aura} blur-2xl`} />
      <div className="ai-decor inset-0 opacity-20 [background-image:linear-gradient(rgba(255,255,255,.08)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,.08)_1px,transparent_1px)] [background-size:28px_28px]" />
      <div className="relative mx-auto flex min-h-[304px] w-full flex-col items-center justify-center">
        {[0, 1, 2].map((i) => (
          <motion.span
            key={i}
            className="ai-decor h-1.5 w-1.5 rounded-full bg-white/70"
            style={{ left: `${26 + i * 22}%`, top: `${22 + (i % 2) * 34}%` }}
            animate={{ opacity: [0.15, 0.8, 0.15], y: [0, -8, 0] }}
            transition={{ duration: 2.4 + i * 0.4, repeat: Infinity, delay: i * 0.35 }}
          />
        ))}
        <svg viewBox="0 0 220 220" className="relative h-64 w-64 -rotate-90">
          <circle cx="110" cy="110" r={radius} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="18" />
          <circle cx="110" cy="110" r={radius} fill="none" stroke="rgba(255,255,255,0.04)" strokeWidth="34" />
          <motion.circle
            cx="110"
            cy="110"
            r={radius}
            fill="none"
            stroke={theme.stroke}
            strokeLinecap="round"
            strokeWidth="18"
            strokeDasharray={`${dash} ${circumference - dash}`}
            initial={{ strokeDasharray: `0 ${circumference}` }}
            animate={{ strokeDasharray: `${dash} ${circumference - dash}` }}
            transition={{ duration: 1.15, ease: "easeOut" }}
            filter="drop-shadow(0 0 14px currentColor)"
          />
          <motion.circle
            cx={markerX}
            cy={markerY}
            r="7"
            fill={theme.stroke}
            stroke="rgba(255,255,255,0.82)"
            strokeWidth="3"
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.9, duration: 0.28 }}
          />
        </svg>
        <div className="ai-content-absolute inset-0 grid place-items-center text-center">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.28em] text-slate-400">Hallucination Risk</p>
            <motion.p initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }} className={`mt-2 text-6xl font-black tracking-tight ${theme.strongText}`}>{Math.round(score)}</motion.p>
            <p className={`mt-2 px-4 text-sm font-semibold leading-6 ${theme.text}`}>{theme.levelText}</p>
          </div>
        </div>
      </div>
      <div className="relative z-10 mx-auto mt-5 w-[88%]">
        <div className="relative h-2 rounded-full bg-gradient-to-r from-emerald-400 via-yellow-300 to-rose-500 shadow-[0_0_24px_rgba(255,255,255,0.08)]">
          <motion.div initial={{ left: "0%" }} animate={{ left: `${score}%` }} transition={{ duration: 1.05, ease: "easeOut" }} className="absolute top-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white shadow-2xl" style={{ backgroundColor: theme.stroke }} />
        </div>
        <div className="mt-3 grid grid-cols-3 text-center text-[11px] font-bold uppercase tracking-[0.14em]">
          <span className={theme.level === "low" ? "text-emerald-200" : "text-slate-500"}>Low</span>
          <span className={theme.level === "medium" ? "text-yellow-200" : "text-slate-500"}>Medium</span>
          <span className={theme.level === "high" ? "text-rose-200" : "text-slate-500"}>High</span>
        </div>
      </div>
    </div>
  );
}

function MetricTile({ icon: Icon, label, value, detail, delay, theme, accented = false }: { icon: typeof Gauge; label: string; value: string | number; detail?: string; delay: number; theme?: RiskTheme; accented?: boolean }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.35, ease: "easeOut" }}
      whileHover={{ y: -3 }}
      className={`glass-panel-ai rounded-3xl border ${accented && theme ? theme.border : "border-white/10"} bg-white/[0.055] p-5 shadow-2xl backdrop-blur-xl transition hover:bg-white/[0.075] ${accented && theme ? theme.glow : "hover:border-cyan-300/35"}`}
    >
      <p className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400"><Icon size={15} /> {label}</p>
      <p className={`mt-3 text-4xl font-black tracking-tight ${accented && theme ? theme.strongText : "text-white"}`}>{value}</p>
      {detail && <p className="mt-2 text-sm leading-6 text-slate-400">{detail}</p>}
    </motion.div>
  );
}

function ClaimFindingCard({ claim, index }: { claim: RecordValue; index: number }) {
  const status = statusLabel(claim);
  const visual = statusVisual(status);
  const Icon = visual.icon;
  const score = claimScore(claim);
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-60px" }}
      transition={{ delay: index * 0.05, duration: 0.32 }}
      whileHover={{ y: -3 }}
      className={`glass-panel-ai rounded-3xl border border-l-4 border-white/10 ${visual.accent} bg-slate-950/45 p-5 shadow-xl backdrop-blur-xl transition hover:bg-white/[0.055] ${visual.glow}`}
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <span className="rounded-full bg-white/8 px-3 py-1 text-xs font-semibold text-slate-300">Claim {index + 1}</span>
        <span className={`inline-flex items-center gap-1 rounded-full border px-3 py-1 text-xs font-semibold ${visual.cls}`}><Icon size={13} /> {status}</span>
      </div>
      <p className="mt-4 text-base leading-7 text-slate-100">{readableRecord(claim, ["claim", "text", "statement"], 360)}</p>
      <div className="mt-4 grid gap-3 text-sm md:grid-cols-2">
        <div className="rounded-2xl bg-white/[0.045] p-3"><p className="text-xs uppercase tracking-[0.14em] text-slate-500">Score</p><p className="mt-1 font-semibold text-white">{textValue(claim.score ?? claim.risk_score ?? claim.confidence, "N/A")}</p></div>
        <div className="rounded-2xl bg-white/[0.045] p-3"><p className="text-xs uppercase tracking-[0.14em] text-slate-500">Evidence</p><p className="mt-1 line-clamp-3 leading-6 text-slate-300">{readableRecord(claim, ["evidence", "source_snippet", "matched_text", "best_match"], 220)}</p></div>
      </div>
      {score != null && (
        <div className="mt-4">
          <div className="mb-1 flex justify-between text-[11px] uppercase tracking-[0.14em] text-slate-500"><span>Claim signal</span><span>{Math.round(score)}%</span></div>
          <div className="h-2 overflow-hidden rounded-full bg-white/10">
            <motion.div initial={{ width: 0 }} whileInView={{ width: `${score}%` }} viewport={{ once: true }} transition={{ duration: 0.75, ease: "easeOut" }} className={`h-full rounded-full ${/rose/.test(visual.cls) ? "bg-rose-400" : /amber/.test(visual.cls) ? "bg-yellow-300" : /emerald/.test(visual.cls) ? "bg-emerald-300" : "bg-violet-300"}`} />
          </div>
        </div>
      )}
      <p className="mt-4 text-sm leading-6 text-slate-400"><span className="font-semibold text-slate-200">Best match:</span> {readableRecord(claim, ["best_match", "matched_text", "source_snippet"], 220)}</p>
      <p className="mt-2 text-sm leading-6 text-slate-500"><span className="font-semibold text-slate-300">Reason:</span> {readableRecord(claim, ["reason", "explanation", "rationale", "support_reason"], 260)}</p>
    </motion.div>
  );
}

function uniqueEvidence(records: RecordValue[]) {
  const seen = new Set<string>();
  return records.filter((item) => {
    const body = short(item.text ?? item.snippet ?? item.content ?? item.passage ?? item.quote ?? item, 420).toLowerCase();
    const source = textValue(item.source ?? item.source_id ?? item.id ?? item.title, "");
    const key = `${source}:${body.replace(/\W+/g, " ").slice(0, 180)}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function EvidencePanel({ result }: { result: DetectorResult }) {
  const evidence = (result.evidence ?? []).slice(0, 4);
  const citations = (result.citations ?? []).slice(0, 4);
  const records = uniqueEvidence([...evidence, ...citations]).slice(0, 6);
  if (!records.length) {
    return (
      <motion.div initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }} className="glass-panel-ai rounded-3xl border border-white/10 bg-white/[0.045] p-6 shadow-xl backdrop-blur-xl">
        <div className="flex items-start gap-4">
          <div className="grid h-12 w-12 shrink-0 place-items-center rounded-2xl bg-cyan-300/10 text-cyan-100"><BrainCircuit size={24} /></div>
          <div>
            <h3 className="text-xl font-bold text-white">Evidence and citations</h3>
            <p className="mt-2 text-base leading-7 text-slate-400">{result.evidence_used || "No external evidence was used for this method. Internal-signal methods rely on model uncertainty rather than retrieved documents."}</p>
          </div>
        </div>
      </motion.div>
    );
  }
  return (
    <motion.div initial={{ opacity: 0, y: 14 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} className="glass-panel-ai rounded-3xl border border-white/10 bg-white/[0.045] p-6 shadow-xl backdrop-blur-xl">
      <h3 className="flex items-center gap-2 text-xl font-bold text-white"><Database size={20} /> Evidence and citations</h3>
      <div className="mt-5 grid gap-3">
        {records.map((item, index) => {
          const verdict = textValue(item.status ?? item.verdict ?? item.support ?? item.label, "neutral");
          const visual = statusVisual(verdict);
          return (
          <div key={index} className="rounded-2xl border border-white/10 bg-slate-950/45 p-4">
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <span className="rounded-full bg-cyan-300/10 px-2 py-1 text-[11px] font-bold text-cyan-100">S{index + 1}</span>
              <span className={`rounded-full border px-2 py-1 text-[11px] font-semibold ${visual.cls}`}>{verdict}</span>
              <p className="flex items-center gap-2 text-sm font-semibold text-cyan-100"><Quote size={15} /> {textValue(item.title ?? item.source ?? item.source_id ?? item.id, `Evidence ${index + 1}`)}</p>
            </div>
            <p className="text-sm leading-7 text-slate-300">{readableRecord(item, ["text", "snippet", "content", "passage", "quote"], 320)}</p>
          </div>
        );})}
      </div>
    </motion.div>
  );
}

function humanTraceSummary(step: RecordValue, index: number, result: DetectorResult) {
  const title = textValue(step.stage ?? step.name ?? step.step, "").toLowerCase();
  const count = result.claim_findings?.length ?? 0;
  if (/claim/.test(title) && /extract/.test(title)) return `Extracted ${count || "the"} factual claim${count === 1 ? "" : "s"} from the evaluated answer.`;
  if (/chunk|source|evidence/.test(title)) return "Prepared source or evidence snippets for matching.";
  if (/retriev|match|ground/.test(title)) return "Matched answer claims against the available evidence context.";
  if (/score|risk|aggregate|final/.test(title)) return "Aggregated method signals into the final hallucination risk score.";
  if (/verify|question/.test(title)) return "Ran a verification step over the extracted claims.";
  return readableRecord(step, ["summary", "description", "message", "output", "result", "verdict"], 320);
}

function TraceTimelinePanel({ result }: { result: DetectorResult }) {
  const steps = (result.intermediate_steps ?? []).slice(0, 7);
  return (
    <div className="glass-panel-ai rounded-3xl border border-white/10 bg-white/[0.045] p-6 shadow-xl backdrop-blur-xl">
      <h3 className="flex items-center gap-2 text-xl font-bold text-white"><GitBranch size={20} /> Trace timeline</h3>
      <div className="relative mt-5 space-y-4">
        <div className="absolute bottom-4 left-[18px] top-4 w-px bg-gradient-to-b from-cyan-300/70 via-teal-300/35 to-transparent" />
        {steps.map((step, index) => (
          <motion.div key={index} initial={{ opacity: 0, x: -12 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }} transition={{ delay: index * 0.06 }} className="relative flex gap-4">
            <motion.div initial={{ scale: 0.6 }} whileInView={{ scale: 1 }} viewport={{ once: true }} className="z-10 grid h-9 w-9 shrink-0 place-items-center rounded-full border border-cyan-300/30 bg-cyan-300/15 text-sm font-bold text-cyan-100 shadow-[0_0_22px_rgba(34,211,238,0.18)]">{index + 1}</motion.div>
            <div className="min-w-0 rounded-2xl border border-white/10 bg-slate-950/45 p-4">
              <p className="flex items-center gap-2 text-base font-semibold text-white"><Network size={15} /> {textValue(step.stage ?? step.name ?? step.step, `Step ${index + 1}`)}</p>
              <p className="mt-2 text-sm leading-7 text-slate-400">{humanTraceSummary(step, index, result)}</p>
              <details className="mt-3">
                <summary className="cursor-pointer text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">Raw step details</summary>
                <pre className="mt-2 max-h-48 overflow-auto rounded-xl bg-slate-950/70 p-3 text-xs text-slate-300">{JSON.stringify(step, null, 2)}</pre>
              </details>
            </div>
          </motion.div>
        ))}
        {!steps.length && <p className="rounded-2xl bg-slate-950/45 p-4 text-sm text-slate-500">No intermediate trace steps were returned.</p>}
      </div>
    </div>
  );
}

function DecisionSummary({ theme, result, statusCounts, evidenceMode }: { theme: RiskTheme; result: DetectorResult; statusCounts: { supported: number; mixed: number; high: number; insufficient: number }; evidenceMode: string }) {
  const priority = theme.level === "high" ? "High priority" : theme.level === "medium" ? "Review recommended" : "Low priority";
  const claimTotal = result.claim_findings?.length ?? 0;
  const evidenceAvailable = Boolean((result.evidence?.length ?? 0) + (result.citations?.length ?? 0));
  const message = theme.level === "high"
    ? `This answer should be reviewed carefully. The detector returned ${statusCounts.high} high-risk claim${statusCounts.high === 1 ? "" : "s"} out of ${claimTotal}.`
    : theme.level === "medium"
      ? `This answer needs review. The detector returned ${statusCounts.mixed} mixed or unclear claim${statusCounts.mixed === 1 ? "" : "s"} out of ${claimTotal}.`
      : `This answer appears lower risk for this method. The detector returned ${statusCounts.supported} supported claim${statusCounts.supported === 1 ? "" : "s"} out of ${claimTotal}.`;
  return (
    <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.22 }} className={`glass-panel-ai rounded-3xl border ${theme.border} ${theme.bg} p-5 shadow-xl backdrop-blur-xl`}>
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className={`flex items-center gap-2 text-xs font-bold uppercase tracking-[0.2em] ${theme.text}`}><ScanSearch size={15} /> Review Recommendation</p>
          <h3 className="mt-2 text-2xl font-black text-white">Decision Summary</h3>
        </div>
        <span className={`rounded-full px-3 py-1 text-xs font-bold ${theme.badge}`}>{priority}</span>
      </div>
      <p className="mt-4 max-w-4xl text-base leading-8 text-slate-200">{message}</p>
      <div className="mt-4 flex flex-wrap gap-2 text-xs">
        <span className="rounded-full bg-white/8 px-3 py-1 text-slate-200">Risk {Math.round(theme.score)}</span>
        <span className="rounded-full bg-white/8 px-3 py-1 text-slate-200">{evidenceMode}</span>
        <span className="rounded-full bg-white/8 px-3 py-1 text-slate-200">{evidenceAvailable ? "Evidence records available" : "No external evidence records"}</span>
      </div>
    </motion.div>
  );
}

function TechnicalMetadata({ result }: { result: DetectorResult }) {
  return (
    <details className="glass-panel-ai rounded-3xl border border-white/10 bg-white/[0.035] p-6 shadow-xl backdrop-blur-xl">
      <summary className="flex cursor-pointer items-center gap-2 text-base font-bold text-white"><Braces size={18} /> Technical metadata</summary>
      <div className="mt-5 grid gap-3 md:grid-cols-3">
        <div className="rounded-2xl bg-slate-950/45 p-4 text-sm text-slate-300"><Clock3 size={16} className="mb-2 text-cyan-100" /> Runtime: {result.latency_ms == null ? "N/A" : `${result.latency_ms} ms`}</div>
        <div className="rounded-2xl bg-slate-950/45 p-4 text-sm text-slate-300">Available: {String(result.available)}</div>
        <div className="rounded-2xl bg-slate-950/45 p-4 text-sm text-slate-300">Runtime status: {result.runtime_status ?? "N/A"}</div>
      </div>
      <pre className="mt-4 max-h-80 overflow-auto rounded-2xl bg-slate-950/75 p-4 text-xs leading-5 text-slate-300">{JSON.stringify(result.metadata, null, 2)}</pre>
    </details>
  );
}

export function VisualResultCard({ result, state }: { result: DetectorResult; state: DashboardState }) {
  const rawScore = result.risk_score ?? result.score;
  const riskScore = normalizeRiskScore(rawScore);
  const theme = getRiskTheme(rawScore, result.risk_label || result.label || riskLabelFromScore(riskScore));
  const confidence = result.confidence == null ? null : Math.round(normalizeRiskScore(result.confidence));
  const claims = (result.claim_findings ?? []).slice(0, 8);
  const HeaderMark = theme.level === "high" ? ShieldAlert : theme.level === "medium" ? Activity : ShieldCheck;
  const family = `${result.family} ${result.method_name}`.toLowerCase();
  const EvidenceIcon = /critic|tool/.test(family) ? Wrench : /retrieval|rag/.test(family) ? Database : /source|evidence|ground/.test(family) ? FileSearch : /verification|cove/.test(family) ? GitBranch : BrainCircuit;
  const evidenceMode = /critic|tool/.test(family) ? "Tool-assisted" : /retrieval|rag/.test(family) ? "Retrieval grounded" : /source|evidence|ground/.test(family) ? "Source grounded" : /verification|cove/.test(family) ? "Verification based" : "Internal signal only";
  const statusCounts = (result.claim_findings ?? []).reduce<{ supported: number; mixed: number; high: number; insufficient: number }>(
    (acc, claim) => {
      const value = statusLabel(claim).toLowerCase();
      if (/support|verified|low_risk|low|pass/.test(value)) acc.supported += 1;
      else if (/contradict|high_risk|high|fail|false|error/.test(value)) acc.high += 1;
      else if (/mixed|weakly_supported|weak|unclear|medium_risk|medium|partial/.test(value)) acc.mixed += 1;
      else acc.insufficient += 1;
      return acc;
    },
    { supported: 0, mixed: 0, high: 0, insufficient: 0 }
  );
  const why = [
    result.summary || result.explanation,
    result.claim_findings?.length ? `${result.claim_findings.length} claim finding(s) returned by ${result.method_name}.` : "",
    result.evidence?.length || result.citations?.length ? "Evidence or citation records were returned for inspection." : ""
  ].filter(Boolean);

  return (
    <motion.section
      initial={{ opacity: 0, y: 24, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className={`glass-panel-ai ai-scan-bg ai-risk-bg-${theme.level} relative w-full overflow-hidden rounded-[2.25rem] bg-gradient-to-br ${theme.gradient} p-px ${theme.glow}`}
    >
      <div className={`ai-decor inset-0 ${theme.aura} opacity-40 blur-3xl`} />
      <div className="ai-decor inset-0 bg-[radial-gradient(circle_at_18%_8%,rgba(255,255,255,0.12),transparent_28%),radial-gradient(circle_at_82%_12%,rgba(168,85,247,0.10),transparent_28%)]" />
      <div className="ai-decor inset-0 opacity-[0.12] [background-image:linear-gradient(rgba(255,255,255,.18)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,.18)_1px,transparent_1px)] [background-size:36px_36px]" />
      <div className="relative rounded-[2.25rem] border border-white/10 bg-slate-950/80 p-5 shadow-2xl backdrop-blur-2xl md:p-7">
        <div className="flex flex-wrap items-start justify-between gap-6">
          <div className="flex min-w-0 items-start gap-4">
            <HeaderIcon theme={theme} />
            <div className="min-w-0">
              <p className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.3em] text-cyan-200"><Sparkles size={14} /> Detector Result</p>
              <h2 className="mt-3 text-3xl font-black tracking-tight text-white md:text-4xl">{result.method_name}</h2>
              <p className={`mt-2 flex items-center gap-2 text-base font-semibold ${theme.text}`}><HeaderMark size={18} /> AI safety scan complete | {theme.levelText}</p>
              <div className="mt-4 flex flex-wrap gap-2">
                <RiskBadge label={theme.label} />
                <span className="rounded-full bg-cyan-300/10 px-3 py-1 text-xs font-semibold text-cyan-100">{result.family}</span>
                <span className="rounded-full bg-white/8 px-3 py-1 text-xs font-semibold text-slate-200">{result.implementation_status}</span>
                <span className={`inline-flex items-center gap-1 rounded-full px-3 py-1 text-xs font-semibold ${theme.badge}`}><EvidenceIcon size={13} /> {evidenceMode}</span>
              </div>
            </div>
          </div>
          <div className="grid w-full gap-3 sm:grid-cols-3 xl:w-[520px]">
            <MetricTile icon={Gauge} label="Risk" value={Math.round(riskScore)} detail="Normalized score" delay={0.08} theme={theme} accented />
            <MetricTile icon={BadgeCheck} label="Confidence" value={confidence == null ? "N/A" : `${confidence}%`} detail="Backend confidence" delay={0.14} />
            <MetricTile icon={Activity} label="Claims" value={result.claim_findings?.length ?? 0} detail="Checked findings" delay={0.2} />
          </div>
        </div>

        <div className="mt-7 h-px bg-gradient-to-r from-transparent via-white/20 to-transparent" />

        <div className="mt-5 flex w-full flex-wrap gap-2 rounded-2xl border border-white/10 bg-slate-950/55 p-2 shadow-xl backdrop-blur-xl">
          {[
            ["Overview", "result-overview"],
            ["Claims", "result-claims"],
            ["Evidence", "result-evidence"],
            ["Trace", "result-trace"],
            ["Metadata", "result-metadata"]
          ].map(([label, target]) => (
            <button key={target} onClick={() => document.getElementById(target)?.scrollIntoView({ behavior: "smooth", block: "start" })} className="rounded-xl px-3 py-2 text-xs font-semibold text-slate-300 transition hover:bg-white/10 hover:text-white">{label}</button>
          ))}
        </div>

        <div id="result-overview" className="mt-7 scroll-mt-32 grid gap-6 xl:grid-cols-[420px_minmax(0,1fr)] 2xl:grid-cols-[440px_minmax(0,1fr)]">
          <div className="space-y-5">
            <RiskGauge score={riskScore} theme={theme} />
            <div className="glass-panel-ai rounded-3xl border border-white/10 bg-white/[0.045] p-5 shadow-xl backdrop-blur-xl">
              <h3 className="flex items-center gap-2 text-xl font-bold text-white"><ScanSearch size={20} /> Why this score?</h3>
              <ul className="mt-4 space-y-3">
                {why.map((item, index) => <li key={index} className="flex gap-3 text-base leading-7 text-slate-300"><span className={`mt-2 h-2 w-2 shrink-0 rounded-full bg-gradient-to-r ${theme.gradient}`} />{short(item, 300)}</li>)}
              </ul>
            </div>
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <div className="glass-panel-ai rounded-3xl border border-white/10 bg-white/[0.045] p-6 shadow-xl backdrop-blur-xl transition hover:border-cyan-300/25">
              <p className="mb-3 flex items-center gap-2 text-xs font-bold uppercase tracking-[0.2em] text-cyan-200"><FileSearch size={15} /> Question</p>
              <p className="text-base leading-8 text-slate-200">{short(state.question, 560)}</p>
              <div className="mt-4 flex flex-wrap gap-2 text-xs"><span className="rounded-full bg-cyan-300/10 px-3 py-1 text-cyan-100">Input prompt</span><span className="rounded-full bg-white/8 px-3 py-1 text-slate-300">{wordCount(state.question)} words</span><span className="rounded-full bg-white/8 px-3 py-1 text-slate-300">{result.method_name}</span></div>
            </div>
            <div className="glass-panel-ai rounded-3xl border border-white/10 bg-white/[0.045] p-6 shadow-xl backdrop-blur-xl transition hover:border-fuchsia-300/25">
              <p className="mb-3 flex items-center gap-2 text-xs font-bold uppercase tracking-[0.2em] text-fuchsia-200"><ShieldAlert size={15} /> Evaluated answer</p>
              <p className="text-base leading-8 text-slate-200">{short(state.answer, 560)}</p>
              <div className="mt-4 flex flex-wrap gap-2 text-xs"><span className="rounded-full bg-fuchsia-300/10 px-3 py-1 text-fuchsia-100">Evaluated response</span><span className="rounded-full bg-white/8 px-3 py-1 text-slate-300">{wordCount(state.answer)} words</span><span className="rounded-full bg-white/8 px-3 py-1 text-slate-300">{result.claim_findings?.length ?? 0} claims</span></div>
            </div>
            <div className="glass-panel-ai rounded-3xl border border-white/10 bg-white/[0.045] p-6 shadow-xl backdrop-blur-xl lg:col-span-2">
              <h3 className="text-xl font-bold text-white">Explanation</h3>
              <p className="mt-3 text-base leading-8 text-slate-300">{result.explanation || result.summary || "The backend returned no explanation text."}</p>
              {result.limitations && <p className="mt-4 rounded-2xl border border-amber-300/20 bg-amber-300/8 p-4 text-sm leading-7 text-amber-100"><Info className="mr-2 inline" size={16} />{result.limitations}</p>}
            </div>
          </div>
        </div>

        <div className="mt-6">
          <DecisionSummary theme={theme} result={result} statusCounts={statusCounts} evidenceMode={evidenceMode} />
        </div>

        <div className="mt-7 h-px bg-gradient-to-r from-transparent via-white/20 to-transparent" />

        <div id="result-claims" className="glass-panel-ai mt-7 scroll-mt-32 rounded-3xl border border-white/10 bg-white/[0.035] p-6 shadow-xl backdrop-blur-xl">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <h3 className="flex items-center gap-2 text-2xl font-black text-white"><BookOpenCheck size={22} /> Claim findings</h3>
            <div className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
              <span className="rounded-full border border-emerald-300/25 bg-emerald-400/10 px-3 py-1 text-emerald-100">{statusCounts.supported} supported</span>
              <span className="rounded-full border border-amber-300/30 bg-amber-400/10 px-3 py-1 text-yellow-100">{statusCounts.mixed} mixed</span>
              <span className="rounded-full border border-rose-300/30 bg-rose-500/10 px-3 py-1 text-rose-100">{statusCounts.high} high-risk</span>
              <span className="rounded-full border border-violet-300/25 bg-violet-400/10 px-3 py-1 text-violet-100">{statusCounts.insufficient} insufficient</span>
            </div>
          </div>
          <div className="mt-5 grid gap-4 xl:grid-cols-2 2xl:grid-cols-3">
            {claims.map((claim, index) => <ClaimFindingCard key={index} claim={claim} index={index} />)}
            {!claims.length && <div className="rounded-3xl border border-white/10 bg-slate-950/45 p-5 text-base text-slate-400">No claim-level records were returned by this method.</div>}
          </div>
        </div>

        <div className="mt-7 h-px bg-gradient-to-r from-transparent via-white/20 to-transparent" />

        <div className="mt-7 grid gap-6 xl:grid-cols-[0.95fr_1.05fr]">
          <div id="result-evidence" className="scroll-mt-32"><EvidencePanel result={result} /></div>
          <div id="result-trace" className="scroll-mt-32"><TraceTimelinePanel result={result} /></div>
        </div>

        {result.revised_answer && (
          <motion.div initial={{ opacity: 0, y: 14 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} className="mt-7 rounded-3xl border border-emerald-300/20 bg-emerald-300/8 p-6 shadow-xl backdrop-blur-xl">
            <h3 className="flex items-center gap-2 text-2xl font-black text-emerald-100"><CheckCircle2 size={22} /> Revised answer</h3>
            <p className="mt-4 text-base leading-8 text-emerald-50">{result.revised_answer}</p>
          </motion.div>
        )}

        <div id="result-metadata" className="mt-7 scroll-mt-32">
          <TechnicalMetadata result={result} />
        </div>
      </div>
    </motion.section>
  );
}
