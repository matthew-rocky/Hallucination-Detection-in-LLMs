import { Server, WifiOff } from "lucide-react";
import type { Mode } from "@/lib/types";
import { ThemeSelector } from "./ThemeSelector";

export function TopStatusBar({
  online,
  methodCount,
  mode,
  selectedCount
}: {
  online: boolean;
  methodCount: number;
  mode: Mode;
  selectedCount: number;
}) {
  return (
    <div className="glass sticky top-3 z-30 flex flex-wrap items-center justify-between gap-3 rounded-3xl px-5 py-4 md:top-5">
      <div>
        <p className="text-xs uppercase tracking-[0.22em] text-cyan-200">Hallucination Detection Studio</p>
        <p className="mt-1 text-sm text-slate-400">AI safety research dashboard</p>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <span className={`inline-flex items-center gap-2 rounded-full px-3 py-2 text-xs font-semibold ${online ? "bg-emerald-300/12 text-emerald-100" : "bg-rose-300/12 text-rose-100"}`}>
          {online ? <Server size={14} /> : <WifiOff size={14} />} Backend {online ? "online" : "offline"}
        </span>
        <span className="rounded-full bg-white/8 px-3 py-2 text-xs text-slate-200">{methodCount} methods</span>
        <span className="rounded-full bg-white/8 px-3 py-2 text-xs text-slate-200">{mode === "quick" ? "ASK Quick Mode" : "Compare Detectors"}</span>
        <span className="rounded-full bg-white/8 px-3 py-2 text-xs text-slate-200">{selectedCount} selected</span>
        <ThemeSelector />
      </div>
    </div>
  );
}
