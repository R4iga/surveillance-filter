'use client';

import { Surveillance } from '@/components/Surveillance';

export default function Page() {
  return (
    <main className="mx-auto flex min-h-screen max-w-6xl flex-col px-4 py-8 md:py-12">
      <header className="mb-8 flex items-end justify-between">
        <div>
          <h1
            className="text-xl md:text-2xl"
            style={{ fontFamily: 'var(--font-display)', fontWeight: 700, letterSpacing: '-0.01em' }}
          >
            SURVEILLANCE<span style={{ color: '#39ff7a' }}> // </span>FILTER
          </h1>
          <p className="mt-1 text-[10px] tracking-[0.2em] uppercase text-muted">
            upload video · get cctv · motion tracking · blinks · glitches
          </p>
        </div>
        <div className="hidden text-right text-[10px] tracking-widest uppercase text-muted md:block">
          100% local · no upload · no recording
        </div>
      </header>

      <Surveillance />

      <section className="mt-12 grid grid-cols-1 gap-4 md:grid-cols-3">
        <InfoCard title="How it works">
          Each frame is compared against the previous one. Pixels that changed beyond the
          sensitivity threshold get clustered into boxes — that&apos;s the real motion tracking.
          Random &quot;blink&quot; boxes flicker on top for the glitchy CCTV feel.
        </InfoCard>
        <InfoCard title="Make it yours">
          Edit the camera label and overlay text. Change the box color. Tune motion
          sensitivity. Toggle scanlines, glitches, and the targeting crosshair on or off.
        </InfoCard>
        <InfoCard title="Privacy">
          The video you load is read frame-by-frame in your browser and rendered to a canvas.
          It is never uploaded to any server. Closing the tab clears everything.
        </InfoCard>
      </section>

      <footer className="mt-auto pt-12">
        <div className="border-t border-[#13201a] pt-5 text-[10px] tracking-widest uppercase text-muted/50">
          SURVEILLANCE // FILTER · CANVAS 2D · NO TRACKING · {new Date().getFullYear()}
        </div>
      </footer>
    </main>
  );
}

function InfoCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="panel p-5">
      <div className="mb-2 text-[10px] tracking-[0.2em] uppercase" style={{ color: '#39ff7a' }}>
        {title}
      </div>
      <p className="text-xs leading-relaxed text-muted">{children}</p>
    </div>
  );
}
