import type { Metadata } from "next";
import { ThemeProvider } from "@/components/ThemeProvider";
import "./globals.css";

export const metadata: Metadata = {
  title: "Hallucination Detection Studio",
  description: "Full-stack AI safety dashboard for comparing hallucination detector methods"
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="dark">
      <body><ThemeProvider>{children}</ThemeProvider></body>
    </html>
  );
}
