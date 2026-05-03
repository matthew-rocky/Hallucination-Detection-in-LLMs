export function normalizeRiskLabel(value?: string | null) {
  const compact = String(value ?? "").trim().toLowerCase().replace(/[\s-]+/g, "_");
  if (compact === "low" || compact === "low_risk") return "Low";
  if (compact === "medium" || compact === "medium_risk") return "Medium";
  if (compact === "high" || compact === "high_risk") return "High";
  return value ? String(value) : "Not Available";
}

export function riskTone(label?: string | null) {
  const normalized = normalizeRiskLabel(label).toLowerCase();
  if (normalized === "low") {
    return {
      border: "from-emerald-300/70 via-teal-300/30 to-cyan-300/20",
      glow: "shadow-[0_0_46px_rgba(16,185,129,0.18)]",
      fill: "bg-gradient-to-r from-emerald-300 to-teal-300",
      text: "text-emerald-100",
      soft: "bg-emerald-300/10 text-emerald-100",
      score: 28
    };
  }
  if (normalized === "high") {
    return {
      border: "from-rose-300/80 via-pink-400/35 to-cyan-300/20",
      glow: "shadow-[0_0_50px_rgba(244,63,94,0.20)]",
      fill: "bg-gradient-to-r from-rose-300 to-pink-400",
      text: "text-rose-100",
      soft: "bg-rose-300/10 text-rose-100",
      score: 88
    };
  }
  return {
    border: "from-amber-300/70 via-orange-300/30 to-cyan-300/20",
    glow: "shadow-[0_0_46px_rgba(251,191,36,0.18)]",
    fill: "bg-gradient-to-r from-amber-300 to-orange-300",
    text: "text-amber-100",
    soft: "bg-amber-300/10 text-amber-100",
    score: 58
  };
}

export function normalizeRiskScore(value?: number | null) {
  if (value == null || Number.isNaN(value)) return 0;
  const scaled = value <= 1 ? value * 100 : value;
  return Math.min(100, Math.max(0, scaled));
}

export function riskLabelFromScore(score: number) {
  if (score >= 70) return "High";
  if (score >= 40) return "Medium";
  return "Low";
}

export function getRiskTheme(score?: number | null, label?: string | null) {
  const normalizedScore = normalizeRiskScore(score);
  const cleanLabel = normalizeRiskLabel(label);
  const respectedLabel = cleanLabel === "Low" || cleanLabel === "Medium" || cleanLabel === "High" ? cleanLabel : riskLabelFromScore(normalizedScore);
  if (respectedLabel === "High") {
    return {
      level: "high" as const,
      label: "High",
      text: "text-rose-100",
      strongText: "text-rose-300",
      bg: "bg-rose-500/10",
      border: "border-rose-300/30",
      glow: "shadow-[0_0_70px_rgba(244,63,94,0.28)]",
      ring: "ring-rose-300/35",
      gradient: "from-rose-500 via-red-400 to-pink-400",
      badge: "border border-rose-300/30 bg-rose-500/15 text-rose-100 shadow-[0_0_24px_rgba(244,63,94,0.22)]",
      iconBg: "bg-rose-500/15",
      aura: "bg-rose-500/24",
      stroke: "#fb7185",
      levelText: "High hallucination risk",
      score: normalizedScore
    };
  }
  if (respectedLabel === "Medium") {
    return {
      level: "medium" as const,
      label: "Medium",
      text: "text-amber-100",
      strongText: "text-yellow-300",
      bg: "bg-amber-400/10",
      border: "border-amber-300/35",
      glow: "shadow-[0_0_70px_rgba(251,191,36,0.26)]",
      ring: "ring-amber-300/35",
      gradient: "from-yellow-300 via-amber-300 to-orange-400",
      badge: "border border-amber-300/35 bg-amber-400/15 text-yellow-100 shadow-[0_0_24px_rgba(251,191,36,0.22)]",
      iconBg: "bg-amber-400/15",
      aura: "bg-amber-400/22",
      stroke: "#facc15",
      levelText: "Needs review",
      score: normalizedScore
    };
  }
  return {
    level: "low" as const,
    label: "Low",
    text: "text-emerald-100",
    strongText: "text-emerald-300",
    bg: "bg-emerald-400/10",
    border: "border-emerald-300/30",
    glow: "shadow-[0_0_70px_rgba(16,185,129,0.24)]",
    ring: "ring-emerald-300/35",
    gradient: "from-emerald-300 via-teal-300 to-cyan-300",
    badge: "border border-emerald-300/30 bg-emerald-400/15 text-emerald-100 shadow-[0_0_24px_rgba(16,185,129,0.18)]",
    iconBg: "bg-emerald-400/15",
    aura: "bg-emerald-400/20",
    stroke: "#34d399",
    levelText: "Low hallucination risk",
    score: normalizedScore
  };
}
