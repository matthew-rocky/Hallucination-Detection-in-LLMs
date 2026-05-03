"use client";

import type { LucideIcon } from "lucide-react";
import { motion } from "framer-motion";

export function MetricCard({
  label,
  value,
  detail,
  icon: Icon,
  tone = "cyan"
}: {
  label: string;
  value: string | number;
  detail?: string;
  icon: LucideIcon;
  tone?: "cyan" | "purple" | "green" | "amber" | "rose" | "indigo";
}) {
  const tones = {
    cyan: "from-cyan-300/22 to-blue-400/10 text-cyan-100",
    purple: "from-fuchsia-300/22 to-violet-400/10 text-fuchsia-100",
    green: "from-emerald-300/22 to-teal-400/10 text-emerald-100",
    amber: "from-yellow-300/22 to-orange-400/10 text-amber-100",
    rose: "from-rose-300/22 to-pink-400/10 text-rose-100",
    indigo: "from-indigo-300/22 to-cyan-400/10 text-indigo-100"
  };
  return (
    <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className={`rounded-2xl border border-white/10 bg-gradient-to-br ${tones[tone]} p-4`}>
      <div className="flex items-center justify-between gap-3">
        <p className="text-xs uppercase tracking-[0.18em] text-slate-400">{label}</p>
        <Icon size={18} />
      </div>
      <p className="mt-4 text-2xl font-semibold text-white">{value}</p>
      {detail && <p className="mt-1 text-xs leading-5 text-slate-400">{detail}</p>}
    </motion.div>
  );
}
