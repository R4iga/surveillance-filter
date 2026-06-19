'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { useDetector, type Detection } from '@/hooks/useDetector';

/**
 * Surveillance filter — REAL AI object detection edition.
 * - User uploads a video (kept in-browser via object URL, never uploaded)
 * - TensorFlow.js COCO-SSD detects people/cars/animals/etc each frame
 * - Detections are tracked across frames (overlap matching + lerp smoothing)
 *   so boxes stay glued to objects instead of jittering
 * - Optional fake "blink" boxes flicker on top for glitchy CCTV flavor
 * - Overlay: REC dot, timestamp, camera label, custom free-text, scanlines, crosshair, glitches
 * - 100% client-side
 */

interface TrackedBox {
  id: number;
  x: number; // 0..1 normalized
  y: number;
  w: number;
  h: number;
  label: string;
  conf: number;
  missed: number; // frames since last matched
}

interface BlinkBox {
  x: number; y: number; w: number; h: number; // 0..1
  label: string;
  life: number;
  maxLife: number;
}

interface Settings {
  boxColor: string;
  camLabel: string;
  freeText: string;
  showDetection: boolean;
  showBlink: boolean;
  showScanlines: boolean;
  showGlitch: boolean;
  showCrosshair: boolean;
}

const DEFAULTS: Settings = {
  boxColor: '#39ff7a',
  camLabel: 'CAM-01 // SECTOR 7',
  freeText: 'AUTHORIZED PERSONNEL ONLY',
  showDetection: true,
  showBlink: true,
  showScanlines: true,
  showGlitch: true,
  showCrosshair: true,
};

const FAKE_CLASSES = ['PERSON', 'VEHICLE', 'OBJECT', 'ENTITY', 'SUBJECT', 'UNKNOWN'];

export function Surveillance() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const trackedRef = useRef<TrackedBox[]>([]);
  const blinkRef = useRef<BlinkBox[]>([]);
  const nextIdRef = useRef(1);
  const rafRef = useRef<number>(0);
  const glitchRef = useRef(0);
  const startTimeRef = useRef(0);
  const detectingRef = useRef(false);

  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [settings, setSettings] = useState<Settings>(DEFAULTS);
  const settingsRef = useRef(settings);
  settingsRef.current = settings;

  const { detect, status } = useDetector();

  // handle file upload
  const onFile = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
    setPlaying(false);
    trackedRef.current = [];
    blinkRef.current = [];
  }, [videoUrl]);

  const togglePlay = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) { v.play(); setPlaying(true); }
    else { v.pause(); setPlaying(false); }
  }, []);

  /** Match new detections to existing tracked boxes (IoU), smooth with lerp. */
  const updateTracked = useCallback((dets: Detection[], fit: { dx: number; dy: number; dw: number; dh: number }, W: number, H: number) => {
    const vw = videoRef.current?.videoWidth || 1;
    const vh = videoRef.current?.videoHeight || 1;
    const sx = fit.dw / vw;
    const sy = fit.dh / vh;
    const norm = dets.map((d) => {
      const [px, py, pw, ph] = d.bbox;
      return {
        x: (fit.dx + px * sx) / W,
        y: (fit.dy + py * sy) / H,
        w: (pw * sx) / W,
        h: (ph * sy) / H,
        label: d.class.toUpperCase(),
        conf: d.score,
      };
    });

    const tracked = trackedRef.current;
    const usedDet = new Set<number>();
    const SMOOTH = 0.35;

    for (const t of tracked) {
      let bestI = -1;
      let bestIoU = 0.25;
      for (let i = 0; i < norm.length; i++) {
        if (usedDet.has(i)) continue;
        const v = iou(t, norm[i]);
        if (v > bestIoU) { bestIoU = v; bestI = i; }
      }
      if (bestI >= 0) {
        const d = norm[bestI];
        usedDet.add(bestI);
        t.x += (d.x - t.x) * SMOOTH;
        t.y += (d.y - t.y) * SMOOTH;
        t.w += (d.w - t.w) * SMOOTH;
        t.h += (d.h - t.h) * SMOOTH;
        t.label = d.label;
        t.conf = d.conf;
        t.missed = 0;
      }
    }
    for (let i = 0; i < norm.length; i++) {
      if (usedDet.has(i)) continue;
      tracked.push({ id: nextIdRef.current++, ...norm[i], missed: 0 });
    }
    trackedRef.current = tracked;
  }, []);

  // main render loop
  useEffect(() => {
    if (!videoUrl) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const W = canvas.width;
    const H = canvas.height;
    startTimeRef.current = performance.now();

    const draw = () => {
      const v = videoRef.current;
      const s = settingsRef.current;

      if (v && v.readyState >= 2) {
        // --- draw video frame (CCTV grade) ---
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, W, H);
        ctx.save();
        ctx.filter = 'contrast(1.15) saturate(0.7) brightness(0.95)';
        const vr = v.videoWidth / v.videoHeight;
        const cr = W / H;
        let dw = W, dh = H, dx = 0, dy = 0;
        if (vr > cr) { dw = H * vr; dx = (W - dw) / 2; } else { dh = W / vr; dy = (H - dh) / 2; }
        ctx.drawImage(v, dx, dy, dw, dh);
        ctx.restore();
        const fit = { dx, dy, dw, dh };
        ctx.fillStyle = 'rgba(20, 60, 35, 0.08)';
        ctx.fillRect(0, 0, W, H);

        // --- run real detection (throttled, non-blocking) ---
        if (s.showDetection && !detectingRef.current && !v.paused) {
          detectingRef.current = true;
          detect(v).then((dets) => {
            detectingRef.current = false;
            if (dets.length) updateTracked(dets, fit, W, H);
          }).catch(() => { detectingRef.current = false; });
        }

        // age tracked boxes (drop stale ones)
        trackedRef.current = trackedRef.current
          .map((b) => ({ ...b, missed: b.missed + 1 }))
          .filter((b) => b.missed < 8);

        if (s.showDetection) {
          for (const b of trackedRef.current) drawBox(ctx, b, W, H, s.boxColor, false);
        }

        // --- fake blink boxes ---
        if (s.showBlink && Math.random() < 0.1 && blinkRef.current.length < 4) {
          const bw = 0.08 + Math.random() * 0.2;
          const bh = 0.1 + Math.random() * 0.25;
          blinkRef.current.push({
            x: Math.random() * (1 - bw), y: Math.random() * (1 - bh),
            w: bw, h: bh,
            label: FAKE_CLASSES[Math.floor(Math.random() * FAKE_CLASSES.length)] + ' ' + (55 + Math.floor(Math.random() * 44)) + '%',
            life: 3 + Math.floor(Math.random() * 8),
            maxLife: 11,
          });
        }
        blinkRef.current = blinkRef.current.filter((b) => { b.life--; return b.life > 0; });
        if (s.showBlink) {
          for (const b of blinkRef.current) {
            drawBox(ctx, { ...b, conf: 0.6, missed: 0 } as unknown as TrackedBox, W, H, s.boxColor, true, b.life / b.maxLife);
          }
        }

        // --- glitches ---
        if (s.showGlitch) {
          glitchRef.current--;
          if (glitchRef.current <= 0 && Math.random() < 0.04) glitchRef.current = 2 + Math.floor(Math.random() * 6);
          if (glitchRef.current > 0) {
            const sy = Math.random() * H;
            const sh = 10 + Math.random() * 40;
            const off = (Math.random() - 0.5) * 40;
            try {
              const data = ctx.getImageData(0, sy, W, sh);
              ctx.putImageData(data, off, sy);
            } catch { /* ignore */ }
            ctx.fillStyle = 'rgba(255,255,255,0.04)';
            for (let i = 0; i < 5; i++) ctx.fillRect(0, Math.random() * H, W, 1 + Math.random() * 2);
          }
        }

        // --- overlay + scanlines ---
        drawOverlay(ctx, W, H, s, startTimeRef.current);
        if (s.showScanlines) {
          ctx.fillStyle = 'rgba(0,0,0,0.18)';
          for (let y = 0; y < H; y += 3) ctx.fillRect(0, y, W, 1);
        }
      }
      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videoUrl]);

  useEffect(() => {
    return () => {
      cancelAnimationFrame(rafRef.current);
      if (videoUrl) URL.revokeObjectURL(videoUrl);
    };
  }, [videoUrl]);

  return (
    <div>
      {videoUrl && (
        <video
          ref={videoRef}
          src={videoUrl}
          loop
          muted
          playsInline
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          className="hidden"
        />
      )}

      <div className="flex flex-col gap-5 lg:flex-row">
        <div className="flex-1">
          <div className="panel relative overflow-hidden">
            <canvas ref={canvasRef} width={960} height={540} className="crt-flicker block w-full" style={{ aspectRatio: '16 / 9', background: '#000' }} />

            {videoUrl && status === 'loading' && (
              <div className="absolute right-3 top-12 rounded border border-[#13201a] bg-bg/80 px-3 py-1.5 text-[10px] tracking-widest uppercase text-warn">
                ⟳ loading AI model…
              </div>
            )}
            {videoUrl && status === 'ready' && (
              <div className="absolute right-3 top-12 rounded border border-[#13201a] bg-bg/80 px-3 py-1.5 text-[10px] tracking-widest uppercase" style={{ color: '#39ff7a' }}>
                ● AI DETECTION LIVE
              </div>
            )}
            {videoUrl && status === 'error' && (
              <div className="absolute right-3 top-12 rounded border border-[#13201a] bg-bg/80 px-3 py-1.5 text-[10px] tracking-widest uppercase text-danger">
                ⚠ model failed — blink mode only
              </div>
            )}

            {!videoUrl && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-bg/90 p-6 text-center">
                <div>
                  <h2 className="mb-2 text-lg" style={{ fontFamily: 'var(--font-display)', fontWeight: 600 }}>
                    Load a video feed
                  </h2>
                  <p className="mx-auto max-w-sm text-xs text-muted">
                    Real AI object detection (TensorFlow.js) runs in your browser — finds people, vehicles, animals. Plus blink glitches and a full CCTV overlay. Nothing is uploaded.
                  </p>
                </div>
                <label className="btn btn-primary cursor-pointer">
                  ⤓ Choose video
                  <input type="file" accept="video/*" onChange={onFile} className="hidden" />
                </label>
              </div>
            )}
          </div>

          {videoUrl && (
            <div className="mt-3 flex items-center gap-3">
              <button className="btn" onClick={togglePlay}>
                {playing ? '❚❚ Pause' : '▶ Play'}
              </button>
              <label className="btn cursor-pointer">
                ⟲ Change video
                <input type="file" accept="video/*" onChange={onFile} className="hidden" />
              </label>
              <span className="ml-auto text-[10px] tracking-widest text-muted">
                AI · LOCAL · NOT UPLOADED
              </span>
            </div>
          )}
        </div>

        <aside className="w-full space-y-3 lg:w-72">
          <div className="panel p-4">
            <div className="label mb-2">Camera label</div>
            <input type="text" value={settings.camLabel} onChange={(e) => setSettings((s) => ({ ...s, camLabel: e.target.value }))} className="w-full" />
          </div>
          <div className="panel p-4">
            <div className="label mb-2">Overlay text</div>
            <input type="text" value={settings.freeText} onChange={(e) => setSettings((s) => ({ ...s, freeText: e.target.value }))} className="w-full" />
          </div>
          <div className="panel p-4">
            <div className="label mb-2">Box color</div>
            <input type="color" value={settings.boxColor} onChange={(e) => setSettings((s) => ({ ...s, boxColor: e.target.value }))} className="h-8 w-full cursor-pointer rounded border border-[#13201a] bg-transparent" />
          </div>
          <div className="panel p-4">
            <div className="label mb-2">Layers</div>
            <div className="space-y-2">
              <Toggle label="AI detection" on={settings.showDetection} set={(v) => setSettings((s) => ({ ...s, showDetection: v }))} />
              <Toggle label="Blink boxes" on={settings.showBlink} set={(v) => setSettings((s) => ({ ...s, showBlink: v }))} />
              <Toggle label="Scanlines" on={settings.showScanlines} set={(v) => setSettings((s) => ({ ...s, showScanlines: v }))} />
              <Toggle label="Glitches" on={settings.showGlitch} set={(v) => setSettings((s) => ({ ...s, showGlitch: v }))} />
              <Toggle label="Crosshair" on={settings.showCrosshair} set={(v) => setSettings((s) => ({ ...s, showCrosshair: v }))} />
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}

/** Intersection-over-Union of two normalized boxes. */
function iou(a: { x: number; y: number; w: number; h: number }, b: { x: number; y: number; w: number; h: number }): number {
  const ax2 = a.x + a.w, ay2 = a.y + a.h;
  const bx2 = b.x + b.w, by2 = b.y + b.h;
  const ix = Math.max(0, Math.min(ax2, bx2) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(ay2, by2) - Math.max(a.y, b.y));
  const inter = ix * iy;
  const union = a.w * a.h + b.w * b.h - inter;
  return union > 0 ? inter / union : 0;
}

function drawBox(ctx: CanvasRenderingContext2D, b: TrackedBox, W: number, H: number, color: string, isFake: boolean, fakeFade?: number) {
  const x = b.x * W;
  const y = b.y * H;
  const w = b.w * W;
  const h = b.h * H;
  ctx.globalAlpha = isFake ? (fakeFade ?? 1) : (b.missed < 3 ? 1 : 0.4);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  const cl = Math.min(w, h) * 0.25;
  ctx.beginPath();
  ctx.moveTo(x, y + cl); ctx.lineTo(x, y); ctx.lineTo(x + cl, y);
  ctx.moveTo(x + w - cl, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + cl);
  ctx.moveTo(x + w, y + h - cl); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w - cl, y + h);
  ctx.moveTo(x + cl, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + h - cl);
  ctx.stroke();
  ctx.fillStyle = color;
  ctx.font = '11px "JetBrains Mono", monospace';
  ctx.fillText(`${b.label} ${Math.round(b.conf * 100)}%`, x, y - 4);
  ctx.globalAlpha = 1;
}

function drawOverlay(ctx: CanvasRenderingContext2D, W: number, H: number, s: Settings, startTime: number) {
  const elapsed = (performance.now() - startTime) / 1000;
  const now = new Date();
  const ts = now.toISOString().replace('T', ' ').substring(0, 19);

  ctx.font = '13px "JetBrains Mono", monospace';
  ctx.fillStyle = s.boxColor;
  ctx.globalAlpha = 0.9;
  ctx.fillStyle = Math.floor(elapsed * 2) % 2 === 0 ? '#ff3b3b' : '#440000';
  ctx.beginPath(); ctx.arc(16, 18, 5, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = s.boxColor;
  ctx.fillText('● REC', 26, 22);
  ctx.fillText(s.camLabel, 70, 22);
  ctx.textAlign = 'right';
  ctx.fillText(ts, W - 12, 22);
  ctx.textAlign = 'left';
  ctx.fillStyle = s.boxColor;
  ctx.globalAlpha = 0.85;
  ctx.fillText(s.freeText, 12, H - 14);
  ctx.textAlign = 'right';
  ctx.fillStyle = '#5a6b5f';
  ctx.fillText('FRAME ' + Math.floor(elapsed * 30).toString().padStart(6, '0'), W - 12, H - 14);
  ctx.textAlign = 'left';
  if (s.showCrosshair) {
    ctx.strokeStyle = s.boxColor;
    ctx.globalAlpha = 0.4;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(W / 2 - 14, H / 2); ctx.lineTo(W / 2 - 4, H / 2);
    ctx.moveTo(W / 2 + 4, H / 2); ctx.lineTo(W / 2 + 14, H / 2);
    ctx.moveTo(W / 2, H / 2 - 14); ctx.lineTo(W / 2, H / 2 - 4);
    ctx.moveTo(W / 2, H / 2 + 4); ctx.lineTo(W / 2, H / 2 + 14);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  const grad = ctx.createRadialGradient(W / 2, H / 2, H * 0.3, W / 2, H / 2, H * 0.85);
  grad.addColorStop(0, 'rgba(0,0,0,0)');
  grad.addColorStop(1, 'rgba(0,0,0,0.55)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, W, H);
}

function Toggle({ label, on, set }: { label: string; on: boolean; set: (v: boolean) => void }) {
  return (
    <button onClick={() => set(!on)} className="flex w-full items-center justify-between text-xs" style={{ color: on ? '#cfe8d4' : '#5a6b5f' }}>
      <span>{label}</span>
      <span className="rounded-full border px-2 py-0.5 text-[9px] tracking-widest" style={{ borderColor: on ? '#39ff7a' : '#13201a', color: on ? '#39ff7a' : '#5a6b5f', background: on ? 'rgba(57,255,122,0.1)' : 'transparent' }}>
        {on ? 'ON' : 'OFF'}
      </span>
    </button>
  );
}
