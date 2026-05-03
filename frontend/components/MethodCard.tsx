"use client";

import { motion } from "framer-motion";
import { CheckCircle2, Cpu, UploadCloud } from "lucide-react";
import type { MethodInfo } from "@/lib/types";

export function methodTone(name: string) {
  if (name.includes("SEP")) return "border-fuchsia-300/35 bg-fuchsia-300/8";
  if (name.includes("Source")) return "border-emerald-300/35 bg-emerald-300/8";
  if (name.includes("Retrieval")) return "border-teal-300/35 bg-teal-300/8";
  if (name.includes("RAG")) return "border-amber-300/35 bg-amber-300/8";
  if (name.includes("Verification-Based")) return "border-orange-300/35 bg-orange-300/8";
  if (name.includes("CoVe")) return "border-indigo-300/35 bg-indigo-300/8";
  if (name.includes("CRITIC")) return "border-rose-300/35 bg-rose-300/8";
  return "border-cyan-300/35 bg-cyan-300/8";
}

export function MethodCard({
  method,
  active,
  onClick,
  compact = false
}: {
  method: MethodInfo;
  active?: boolean;
  onClick?: () => void;
  compact?: boolean;
}) {
  const content = (
    <motion.div
      whileHover={{ y: -2 }}
      className={`glass-panel-ai h-full rounded-2xl border p-4 transition ${methodTone(method.name)} ${active ? "shadow-glow ring-1 ring-cyan-300/40" : "hover:bg-white/8"}`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="grid h-11 w-11 place-items-center rounded-xl bg-white/8 text-cyan-100">
          <Cpu size={20} />
        </div>
        <div className="flex gap-2">
          {method.supports_uploads && <UploadCloud className="text-slate-300" size={16} />}
          {active && <CheckCircle2 className="text-cyan-200" size={18} />}
        </div>
      </div>
      <h3 className="mt-4 text-sm font-semibold leading-5 text-white">{method.name}</h3>
      <p className="mt-2 text-xs leading-5 text-slate-400">{compact ? method.short_purpose : method.how_it_works}</p>
      <div className="mt-4 flex flex-wrap gap-2">
        <span className="rounded-full bg-white/10 px-2 py-1 text-[11px] text-slate-200">{method.implementation}</span>
        <span className="rounded-full bg-cyan-300/10 px-2 py-1 text-[11px] text-cyan-100">{method.family}</span>
      </div>
      {!compact && (
        <div className="mt-4 grid gap-2 text-xs text-slate-400">
          <p><span className="text-slate-200">Required:</span> {method.required_fields.join(", ") || "None"}</p>
          <p><span className="text-slate-200">Optional:</span> {method.optional_fields.join(", ") || "None"}</p>
        </div>
      )}
    </motion.div>
  );
  if (!onClick) return content;
  return <button type="button" onClick={onClick} className="block h-full w-full text-left">{content}</button>;
}
