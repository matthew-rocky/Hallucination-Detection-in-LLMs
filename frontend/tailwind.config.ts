import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: ["./app/**/*.{js,ts,jsx,tsx}", "./components/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        midnight: "#080b17",
        panel: "rgba(15, 23, 42, 0.68)",
        line: "rgba(148, 163, 184, 0.22)"
      },
      boxShadow: {
        glow: "0 0 38px rgba(45, 212, 191, 0.18)",
        risk: "0 0 28px rgba(248, 113, 113, 0.25)"
      }
    }
  },
  plugins: []
};

export default config;
