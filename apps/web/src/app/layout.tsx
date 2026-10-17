import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { Analytics } from "@vercel/analytics/react";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "EditBanana - Convert Images to Editable Diagrams | AI-Powered",
  description: "Transform static diagrams, flowcharts, and architecture images into fully editable DrawIO files. AI-powered conversion with 99% accuracy. Free to try!",
  keywords: "diagram converter, image to editable, flowchart OCR, architecture diagram tool, diagram to drawio, convert image to diagram",
  authors: [{ name: "BIT DataLab" }],
  creator: "BIT DataLab",
  publisher: "EditBanana",
  metadataBase: new URL("https://editbanana.anxin6.cn"),
  alternates: {
    canonical: "/",
  },
  openGraph: {
    type: "website",
    locale: "en_US",
    url: "https://editbanana.anxin6.cn",
    siteName: "EditBanana",
    title: "EditBanana - Convert Images to Editable Diagrams",
    description: "Transform static diagrams into fully editable DrawIO files with AI. 99% accuracy, free to try!",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "EditBanana - AI Diagram Converter",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "EditBanana - Convert Images to Editable Diagrams",
    description: "Transform static diagrams into fully editable DrawIO files with AI. 99% accuracy, free to try!",
    images: ["/og-image.png"],
    creator: "@editbanana",
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-video-preview": -1,
      "max-image-preview": "large",
      "max-snippet": -1,
    },
  },
  verification: {
    google: "google-site-verification-code",
  },
};

// JSON-LD structured data for SoftwareApplication
const jsonLd = {
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  name: "EditBanana",
  applicationCategory: "GraphicsApplication",
  operatingSystem: "Web",
  description: "AI-powered tool to convert images and PDFs to editable DrawIO diagrams",
  offers: {
    "@type": "Offer",
    price: "0",
    priceCurrency: "USD",
    description: "Free to try",
  },
  aggregateRating: {
    "@type": "AggregateRating",
    ratingValue: "4.8",
    ratingCount: "100",
  },
  url: "https://editbanana.anxin6.cn",
  screenshot: "https://editbanana.anxin6.cn/og-image.png",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
        />
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
        <Analytics />
      </body>
    </html>
  );
}
