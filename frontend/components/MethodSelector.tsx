"use client";

import { motion } from "framer-motion";
import { Check, Layers3 } from "lucide-react";
import type { MethodInfo, Mode } from "@/lib/types";

export function MethodSelector({
  methods,
  selected,
  mode,
  onChange
}: {
  methods: MethodInfo[];
  selected: string[];
  mode: Mode;
  onChange: (methods: string[]) => void;
}) {
  const toggle = (name: string) => {
    if (mode === "quick") {
      onChange([name]);
      return;
    }
    onChange(selected.includes(name) ? selected.filter((item) => item !== name) : [...selected, name]);
  };

  return (
    <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
      {methods.map((method, index) => {
        const active = selected.includes(method.name);
        return (
          <motion.button
            key={method.name}
            type="button"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.025 }}
            onClick={() => toggle(method.name)}
            className={`min-h-52 rounded-2xl border p-4 text-left transition ${
              active
                ? "border-cyan-300/60 bg-cyan-300/12 shadow-glow"
                : "border-white/10 bg-white/5 hover:border-white/25 hover:bg-white/8"
            }`}
          >
            <div className="mb-4 flex items-center justify-between gap-3">
              <span className="grid h-10 w-10 place-items-center rounded-xl bg-white/8 text-cyan-100">
                <Layers3 size={18} />
              </span>
              <span className={`grid h-7 w-7 place-items-center rounded-full ${active ? "bg-cyan-300 text-slate-950" : "bg-slate-800 text-slate-500"}`}>
                <Check size={15} />
              </span>
            </div>
            <h3 className="text-sm font-semibold leading-5 text-white">{method.name}</h3>
            <p className="mt-2 text-xs leading-5 text-slate-400">{method.short_purpose}</p>
            <div className="mt-4 flex flex-wrap gap-2">
              <span className="rounded-full bg-fuchsia-300/10 px-2 py-1 text-[11px] text-fuchsia-100">{method.implementation}</span>
              <span className="rounded-full bg-emerald-300/10 px-2 py-1 text-[11px] text-emerald-100">{method.required_fields.length} required</span>
            </div>
          </motion.button>
        );
      })}
    </div>
  );
}
