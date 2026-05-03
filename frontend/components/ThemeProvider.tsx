"use client";

import { createContext, useContext, useEffect, useMemo, useState } from "react";

export type ThemeChoice = "dark" | "light" | "system";

type ThemeContextValue = {
  theme: ThemeChoice;
  resolvedTheme: "dark" | "light";
  setTheme: (theme: ThemeChoice) => void;
};

const ThemeContext = createContext<ThemeContextValue | undefined>(undefined);

function systemTheme() {
  if (typeof window === "undefined") return "dark";
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

function applyTheme(choice: ThemeChoice) {
  const resolved = choice === "system" ? systemTheme() : choice;
  document.documentElement.classList.remove("dark", "light");
  document.documentElement.classList.add(resolved);
  document.documentElement.dataset.theme = choice;
  return resolved;
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setThemeState] = useState<ThemeChoice>("dark");
  const [resolvedTheme, setResolvedTheme] = useState<"dark" | "light">("dark");

  useEffect(() => {
    const stored = window.localStorage.getItem("hds-theme") as ThemeChoice | null;
    const initial = stored === "light" || stored === "dark" || stored === "system" ? stored : "dark";
    setThemeState(initial);
    setResolvedTheme(applyTheme(initial));

    const media = window.matchMedia("(prefers-color-scheme: light)");
    const onChange = () => {
      const current = (window.localStorage.getItem("hds-theme") as ThemeChoice | null) ?? "dark";
      if (current === "system") setResolvedTheme(applyTheme("system"));
    };
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);

  const value = useMemo<ThemeContextValue>(() => ({
    theme,
    resolvedTheme,
    setTheme: (next) => {
      window.localStorage.setItem("hds-theme", next);
      setThemeState(next);
      setResolvedTheme(applyTheme(next));
    }
  }), [theme, resolvedTheme]);

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  const value = useContext(ThemeContext);
  if (!value) throw new Error("useTheme must be used inside ThemeProvider");
  return value;
}
