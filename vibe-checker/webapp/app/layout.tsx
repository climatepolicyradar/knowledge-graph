import "./app.css";

import { DM_Mono, Fraunces } from "next/font/google";

import type { Metadata } from "next";

export const metadata: Metadata = {
  title: {
    template: "%s | Vibe Checker",
    default: "Vibe Checker",
  },
  description: "Check the predictions from candidate classifiers",
};

const dmMono = DM_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-dm-mono",
  display: "swap",
});

const fraunces = Fraunces({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800", "900"],
  variable: "--font-fraunces",
  display: "swap",
});

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${fraunces.variable} ${dmMono.variable}`}>
      <body className="bg-bg-primary text-text-primary">
        <div className="min-h-screen">{children}</div>
      </body>
    </html>
  );
}
