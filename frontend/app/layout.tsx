import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ViperTrace - Malware Detection",
  description:
    "Advanced AI-powered malware detection system using deep learning",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
