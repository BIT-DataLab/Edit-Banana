import { Navbar } from "./sections/navbar"
import { Hero } from "./sections/hero"
import { UploadSection } from "./sections/upload-section"
import { Features } from "./sections/features"
import { ExampleShowcase } from "./sections/example-showcase"
import { Footer } from "./sections/footer"

export default function Home() {
  return (
    <div className="min-h-screen bg-white">
      <Navbar />
      <main>
        <Hero />
        <UploadSection />
        <Features />
        <ExampleShowcase />
      </main>
      <Footer />
    </div>
  )
}
