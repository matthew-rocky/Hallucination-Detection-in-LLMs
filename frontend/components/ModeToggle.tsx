"use client";

import type { Mode } from "@/lib/types";

export function ModeToggle({ mode, onChange }: { mode: Mode; onChange: (mode: Mode) => void }) {
  return (
    <div className="grid grid-cols-2 rounded-2xl border border-white/10 bg-slate-950/40 p-1">
      {(["quick", "compare"] as Mode[]).map((item) => (
        <button
          key={item}
          onClick={() => onChange(item)}
          className={`rounded-xl px-4 py-3 text-sm font-medium transition ${
            mode === item ? "bg-cyan-300 text-slate-950 shadow-glow" : "text-slate-300 hover:bg-white/8"
          }`}
        >
          {item === "quick" ? "Quick Check" : "Compare Methods"}
        </button>
      ))}
    </div>
  );
}
