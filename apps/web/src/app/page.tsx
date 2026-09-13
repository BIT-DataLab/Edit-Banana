import { Navbar } from "./sections/navbar"
import { Hero } from "./sections/hero"
import { UploadSection } from "./sections/upload-section"
import { Features } from "./sections/features"
import { ExampleShowcase } from "./sections/example-showcase"
import { Footer } from "./sections/footer"
import { ReferralBanner } from "@/components/share/referral-banner"

export default function Home() {
  return (
    <div className="min-h-screen bg-white">
      <Navbar />
      <main>
        <Hero />
        <ReferralBanner />
        <UploadSection />
        <Features />
        <ExampleShowcase />
      </main>
      <Footer />
    </div>
  )
}
