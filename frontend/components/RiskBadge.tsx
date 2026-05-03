import { getRiskTheme, normalizeRiskLabel } from "@/lib/risk";

export function riskClass(label?: string) {
  const value = normalizeRiskLabel(label).toLowerCase();
  if (value === "low") return "risk-low";
  if (value === "medium") return "risk-medium";
  if (value === "high") return "risk-high";
  return "risk-na";
}

export function RiskBadge({ label }: { label?: string }) {
  const theme = getRiskTheme(null, label);
  return <span className={`${theme.badge} rounded-full px-3 py-1 text-xs font-semibold`}>{normalizeRiskLabel(label)}</span>;
}
