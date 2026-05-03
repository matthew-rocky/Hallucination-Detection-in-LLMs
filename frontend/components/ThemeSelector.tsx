"use client";

import { Monitor, Moon, Sun } from "lucide-react";
import { useTheme, type ThemeChoice } from "./ThemeProvider";

const options: { value: ThemeChoice; label: string; icon: typeof Moon }[] = [
  { value: "dark", label: "Dark", icon: Moon },
  { value: "light", label: "Light", icon: Sun },
  { value: "system", label: "System", icon: Monitor }
];

export function ThemeSelector() {
  const { theme, setTheme } = useTheme();
  return (
    <div className="inline-flex rounded-full border border-white/10 bg-white/8 p-1 shadow-xl backdrop-blur-xl">
      {options.map(({ value, label, icon: Icon }) => {
        const active = theme === value;
        return (
          <button
            key={value}
            type="button"
            onClick={() => setTheme(value)}
            aria-pressed={active}
            title={`${label} theme`}
            className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-semibold transition ${
              active
                ? "bg-cyan-300 text-slate-950 shadow-glow"
                : "text-slate-300 hover:bg-white/10 hover:text-white"
            }`}
          >
            <Icon size={14} />
            <span className="hidden sm:inline">{label}</span>
          </button>
        );
      })}
    </div>
  );
}
