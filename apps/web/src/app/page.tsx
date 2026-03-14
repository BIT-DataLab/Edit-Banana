import { Navbar } from "./sections/navbar"
import { Hero } from "./sections/hero"
import { UploadSection } from "./sections/upload-section"
import { Features } from "./sections/features"
import { Footer } from "./sections/footer"

export default function Home() {
  return (
    <div className="min-h-screen bg-white">
      <Navbar />
      <main>
        <Hero />
        <UploadSection />
        <Features />
      </main>
      <Footer />
    </div>
  )
}
