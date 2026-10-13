import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Open Text Detector",
  description: "Explore research papers and read model classifications in the context of the original PDF.",
  other: {
    "codex-preview": "development",
  },
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
