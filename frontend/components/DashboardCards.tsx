"use client";

import { motion } from "framer-motion";
import { Activity, Gauge, Layers3, Server, ShieldAlert, ShieldCheck, ShieldQuestion } from "lucide-react";
import type { DetectorResult, MethodInfo, Mode } from "@/lib/types";

function riskCounts(results: DetectorResult[]) {
  return {
    low: results.filter((item) => item.risk_label === "Low").length,
    medium: results.filter((item) => item.risk_label === "Medium").length,
    high: results.filter((item) => item.risk_label === "High").length
  };
}

export function DashboardCards({
  methods,
  selectedCount,
  mode,
  backendStatus,
  results
}: {
  methods: MethodInfo[];
  selectedCount: number;
  mode: Mode;
  backendStatus: string;
  results: DetectorResult[];
}) {
  const claims = results.reduce((sum, result) => sum + (result.claim_findings?.length ?? 0), 0);
  const counts = riskCounts(results);
  const cards = [
    { label: "Methods available", value: methods.length, icon: Layers3, color: "from-cyan-300/20 to-blue-400/10" },
    { label: "Selected methods", value: selectedCount, icon: Gauge, color: "from-fuchsia-300/20 to-pink-400/10" },
    { label: "Backend status", value: backendStatus, icon: Server, color: "from-emerald-300/20 to-teal-400/10" },
    { label: "Runtime mode", value: mode === "quick" ? "Quick" : "Compare", icon: Activity, color: "from-amber-300/20 to-orange-400/10" },
    { label: "Claims checked", value: claims, icon: ShieldQuestion, color: "from-violet-300/20 to-cyan-400/10" },
    { label: "Low risk", value: counts.low, icon: ShieldCheck, color: "from-emerald-300/20 to-emerald-500/10" },
    { label: "Medium risk", value: counts.medium, icon: ShieldAlert, color: "from-yellow-300/20 to-amber-500/10" },
    { label: "High risk", value: counts.high, icon: ShieldAlert, color: "from-rose-300/20 to-pink-500/10" }
  ];

  return (
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
      {cards.map((card, index) => {
        const Icon = card.icon;
        return (
          <motion.div
            key={card.label}
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.035 }}
            className={`rounded-2xl border border-white/10 bg-gradient-to-br ${card.color} p-4`}
          >
            <div className="flex items-center justify-between gap-3">
              <span className="text-xs uppercase tracking-[0.18em] text-slate-400">{card.label}</span>
              <Icon className="text-cyan-100" size={18} />
            </div>
            <p className="mt-4 text-2xl font-semibold text-white">{card.value}</p>
          </motion.div>
        );
      })}
    </div>
  );
}
