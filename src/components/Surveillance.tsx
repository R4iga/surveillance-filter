'use client';

import { useEffect, useRef, useState, useCallback } from 'react';

/**
 * Surveillance filter.
 * - User uploads a video (kept in-browser via object URL, never uploaded)
 * - Each frame is drawn to a canvas, then motion detection runs:
 *   frame-difference against the previous frame into a coarse grid → clusters → bounding boxes
 * - Additional "fake" blink boxes flicker on/off randomly for the glitchy CCTV feel
 * - Overlay: REC dot, timestamp, camera label, custom free-text, classification tags, scanlines, crosshair, glitches
 * - 100% client-side, no network
 */

const CLASSES = ['PERSON', 'VEHICLE', 'OBJECT', 'MOTION', 'ENTITY', 'SUBJECT', 'UNKNOWN'];

interface Box {
  x: number; // 0..1 normalized
  y: number;
  w: number;
  h: number;
  label: string;
  conf: number;
  life: number; // frames remaining
  maxLife: number;
  fake: boolean;
}

interface Settings {
  sensitivity: number; // motion threshold 0..1
  boxColor: string;
  camLabel: string;
  freeText: string;
  showMotion: boolean;
  showBlink: boolean;
  showScanlines: boolean;
  showGlitch: boolean;
  showCrosshair: boolean;
}

const DEFAULTS: Settings = {
  sensitivity: 0.18,
  boxColor: '#39ff7a',
  camLabel: 'CAM-01 // SECTOR 7',
  freeText: 'AUTHORIZED PERSONNEL ONLY',
  showMotion: true,
  showBlink: true,
  showScanlines: true,
  showGlitch: true,
  showCrosshair: true,
};

export function Surveillance() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const prevFrameRef = useRef<Uint8ClampedArray | null>(null);
  const boxesRef = useRef<Box[]>([]);
  const rafRef = useRef<number>(0);
  const glitchRef = useRef(0);
  const startTimeRef = useRef(0);

  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [settings, setSettings] = useState<Settings>(DEFAULTS);
  const settingsRef = useRef(settings);
  settingsRef.current = settings;

  // offscreen canvases for motion detection
  const sampleRef = useRef<HTMLCanvasElement | null>(null);
  const GRID_W = 32;
  const GRID_H = 18;

  useEffect(() => {
    const c = document.createElement('canvas');
    c.width = GRID_W;
    c.height = GRID_H;
    sampleRef.current = c;
    return () => c.remove();
  }, []);

  // handle file upload
  const onFile = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
    setPlaying(false);
    boxesRef.current = [];
    prevFrameRef.current = null;
  }, [videoUrl]);

  // toggle play/pause
  const togglePlay = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) {
      v.play();
      setPlaying(true);
    } else {
      v.pause();
      setPlaying(false);
    }
  }, []);

  // main render loop
  useEffect(() => {
    if (!videoUrl) return;
    const canvas = canvasRef.current;
    const sample = sampleRef.current;
    if (!canvas || !sample) return;
    const ctx = canvas.getContext('2d');
    const sCtx = sample.getContext('2d', { willReadFrequently: true });
    if (!ctx || !sCtx) return;

    const W = canvas.width;
    const H = canvas.height;
    startTimeRef.current = performance.now();

    const draw = () => {
      const v = videoRef.current;
      const s = settingsRef.current;
      if (v && v.readyState >= 2) {
        // --- draw video frame (slightly desaturated, contrast bumped for CCTV look) ---
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, W, H);
        ctx.save();
        ctx.filter = 'contrast(1.15) saturate(0.7) brightness(0.95)';
        // cover-fit
        const vr = v.videoWidth / v.videoHeight;
        const cr = W / H;
        let dw = W, dh = H, dx = 0, dy = 0;
        if (vr > cr) { dw = H * vr; dx = (W - dw) / 2; } else { dh = W / vr; dy = (H - dh) / 2; }
        ctx.drawImage(v, dx, dy, dw, dh);
        ctx.restore();

        // slight green tint
        ctx.fillStyle = 'rgba(20, 60, 35, 0.08)';
        ctx.fillRect(0, 0, W, H);

        // --- motion detection ---
        if (s.showMotion) {
          sCtx.drawImage(v, 0, 0, GRID_W, GRID_H);
          const cur = sCtx.getImageData(0, 0, GRID_W, GRID_H).data;
          const prev = prevFrameRef.current;
          if (prev) {
            const grid = new Uint8Array(GRID_W * GRID_H);
            for (let i = 0; i < grid.length; i++) {
              const di = i * 4;
              const d = Math.abs(cur[di] - prev[di]) + Math.abs(cur[di + 1] - prev[di + 1]) + Math.abs(cur[di + 2] - prev[di + 2]);
              grid[i] = d > s.sensitivity * 255 * 3 ? 1 : 0;
            }
            // cluster active cells into boxes (simple connected-component-ish via rows)
            const motionBoxes = clusterBoxes(grid, GRID_W, GRID_H);
            // convert normalized boxes, add as tracked boxes
            for (const mb of motionBoxes) {
              // throttle: only add if few boxes
              if (boxesRef.current.filter((b) => !b.fake).length < 8) {
                boxesRef.current.push({
                  x: mb.x / GRID_W,
                  y: mb.y / GRID_H,
                  w: mb.w / GRID_W,
                  h: mb.h / GRID_H,
                  label: CLASSES[Math.floor(Math.random() * 3)] + ' ' + (70 + Math.floor(Math.random() * 29)) + '%',
                  conf: 0.7 + Math.random() * 0.29,
                  life: 8 + Math.floor(Math.random() * 10),
                  maxLife: 18,
                  fake: false,
                });
              }
            }
          }
          prevFrameRef.current = cur;
        }

        // --- fake blink boxes ---
        if (s.showBlink && Math.random() < 0.12 && boxesRef.current.filter((b) => b.fake).length < 4) {
          const bw = 0.08 + Math.random() * 0.2;
          const bh = 0.1 + Math.random() * 0.25;
          boxesRef.current.push({
            x: Math.random() * (1 - bw),
            y: Math.random() * (1 - bh),
            w: bw,
            h: bh,
            label: CLASSES[Math.floor(Math.random() * CLASSES.length)] + ' ' + (55 + Math.floor(Math.random() * 44)) + '%',
            conf: 0.55 + Math.random() * 0.44,
            life: 3 + Math.floor(Math.random() * 8),
            maxLife: 11,
            fake: true,
          });
        }

        // --- draw + age boxes ---
        const next: Box[] = [];
        for (const b of boxesRef.current) {
          b.life--;
          if (b.life > 0) {
            drawBox(ctx, b, W, H, s.boxColor);
            next.push(b);
          }
        }
        boxesRef.current = next;

        // --- glitches ---
        if (s.showGlitch) {
          glitchRef.current--;
          if (glitchRef.current <= 0 && Math.random() < 0.04) {
            glitchRef.current = 2 + Math.floor(Math.random() * 6);
          }
          if (glitchRef.current > 0) {
            // RGB slice shift
            const sy = Math.random() * H;
            const sh = 10 + Math.random() * 40;
            const off = (Math.random() - 0.5) * 40;
            try {
              const data = ctx.getImageData(0, sy, W, sh);
              ctx.putImageData(data, off, sy);
            } catch { /* ignore */ }
            // noise bars
            ctx.fillStyle = 'rgba(255,255,255,0.04)';
            for (let i = 0; i < 5; i++) {
              ctx.fillRect(0, Math.random() * H, W, 1 + Math.random() * 2);
            }
          }
        }

        // --- overlay ---
        drawOverlay(ctx, W, H, s, startTimeRef.current);

        // --- scanlines ---
        if (s.showScanlines) {
          ctx.fillStyle = 'rgba(0,0,0,0.18)';
          for (let y = 0; y < H; y += 3) ctx.fillRect(0, y, W, 1);
        }
      }
      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [videoUrl]);

  // cleanup
  useEffect(() => {
    return () => {
      cancelAnimationFrame(rafRef.current);
      if (videoUrl) URL.revokeObjectURL(videoUrl);
    };
  }, [videoUrl]);

  return (
    <div>
      {/* hidden video element sourcing the frames */}
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
        {/* viewport */}
        <div className="flex-1">
          <div className="panel relative overflow-hidden">
            <canvas
              ref={canvasRef}
              width={960}
              height={540}
              className="crt-flicker block w-full"
              style={{ aspectRatio: '16 / 9', background: '#000' }}
            />
            {!videoUrl && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-bg/90 p-6 text-center">
                <div>
                  <h2 className="mb-2 text-lg" style={{ fontFamily: 'var(--font-display)', fontWeight: 600 }}>
                    Load a video feed
                  </h2>
                  <p className="mx-auto max-w-sm text-xs text-muted">
                    Drop any video file — it&apos;s processed locally in your browser. Nothing is uploaded. You&apos;ll get motion-tracking boxes, blink glitches, and a full CCTV overlay.
                  </p>
                </div>
                <label className="btn btn-primary cursor-pointer">
                  ⤓ Choose video
                  <input type="file" accept="video/*" onChange={onFile} className="hidden" />
                </label>
              </div>
            )}
          </div>

          {/* transport */}
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
                PROCESSED LOCALLY · NOT UPLOADED
              </span>
            </div>
          )}
        </div>

        {/* controls */}
        <aside className="w-full space-y-3 lg:w-72">
          <div className="panel p-4">
            <div className="label mb-2">Camera label</div>
            <input
              type="text"
              value={settings.camLabel}
              onChange={(e) => setSettings((s) => ({ ...s, camLabel: e.target.value }))}
              className="w-full"
            />
          </div>
          <div className="panel p-4">
            <div className="label mb-2">Overlay text</div>
            <input
              type="text"
              value={settings.freeText}
              onChange={(e) => setSettings((s) => ({ ...s, freeText: e.target.value }))}
              className="w-full"
            />
          </div>
          <div className="panel p-4">
            <div className="label mb-2">Box color</div>
            <input
              type="color"
              value={settings.boxColor}
              onChange={(e) => setSettings((s) => ({ ...s, boxColor: e.target.value }))}
              className="h-8 w-full cursor-pointer rounded border border-[#13201a] bg-transparent"
            />
          </div>
          <div className="panel p-4">
            <div className="label mb-2">Motion sensitivity · {Math.round(settings.sensitivity * 100)}%</div>
            <input
              type="range"
              min={2}
              max={50}
              value={Math.round(settings.sensitivity * 100)}
              onChange={(e) => setSettings((s) => ({ ...s, sensitivity: Number(e.target.value) / 100 }))}
              className="w-full"
            />
          </div>
          <div className="panel p-4">
            <div className="label mb-2">Layers</div>
            <div className="space-y-2">
              <Toggle label="Motion tracking" on={settings.showMotion} set={(v) => setSettings((s) => ({ ...s, showMotion: v }))} />
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

function Toggle({ label, on, set }: { label: string; on: boolean; set: (v: boolean) => void }) {
  return (
    <button
      onClick={() => set(!on)}
      className="flex w-full items-center justify-between text-xs"
      style={{ color: on ? '#cfe8d4' : '#5a6b5f' }}
    >
      <span>{label}</span>
      <span
        className="rounded-full border px-2 py-0.5 text-[9px] tracking-widest"
        style={{
          borderColor: on ? '#39ff7a' : '#13201a',
          color: on ? '#39ff7a' : '#5a6b5f',
          background: on ? 'rgba(57,255,122,0.1)' : 'transparent',
        }}
      >
        {on ? 'ON' : 'OFF'}
      </span>
    </button>
  );
}

/** Cluster active grid cells into bounding boxes. */
function clusterBoxes(grid: Uint8Array, gw: number, gh: number): { x: number; y: number; w: number; h: number }[] {
  const boxes: { x: number; y: number; w: number; h: number }[] = [];
  const visited = new Uint8Array(grid.length);
  for (let y = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++) {
      const idx = y * gw + x;
      if (!grid[idx] || visited[idx]) continue;
      // BFS flood fill
      let minX = x, maxX = x, minY = y, maxY = y, count = 0;
      const stack = [idx];
      while (stack.length) {
        const i = stack.pop()!;
        if (visited[i] || !grid[i]) continue;
        visited[i] = 1;
        count++;
        const cx = i % gw;
        const cy = (i / gw) | 0;
        if (cx < minX) minX = cx;
        if (cx > maxX) maxX = cx;
        if (cy < minY) minY = cy;
        if (cy > maxY) maxY = cy;
        if (cx > 0) stack.push(i - 1);
        if (cx < gw - 1) stack.push(i + 1);
        if (cy > 0) stack.push(i - gw);
        if (cy < gh - 1) stack.push(i + gw);
      }
      if (count >= 3) {
        boxes.push({ x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 });
      }
    }
  }
  return boxes;
}

function drawBox(ctx: CanvasRenderingContext2D, b: Box, W: number, H: number, color: string) {
  const x = b.x * W;
  const y = b.y * H;
  const w = b.w * W;
  const h = b.h * H;
  const fade = b.life / b.maxLife;
  ctx.globalAlpha = Math.min(1, fade * 1.5);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  // corner-bracket style
  const cl = Math.min(w, h) * 0.25;
  ctx.beginPath();
  // top-left
  ctx.moveTo(x, y + cl); ctx.lineTo(x, y); ctx.lineTo(x + cl, y);
  // top-right
  ctx.moveTo(x + w - cl, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + cl);
  // bottom-right
  ctx.moveTo(x + w, y + h - cl); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w - cl, y + h);
  // bottom-left
  ctx.moveTo(x + cl, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + h - cl);
  ctx.stroke();

  // label
  ctx.fillStyle = color;
  ctx.font = '11px "JetBrains Mono", monospace';
  ctx.fillText(b.label, x, y - 4);
  ctx.globalAlpha = 1;
}

function drawOverlay(ctx: CanvasRenderingContext2D, W: number, H: number, s: Settings, startTime: number) {
  const elapsed = (performance.now() - startTime) / 1000;
  const now = new Date(Date.now()); // live wall clock
  const ts = now.toISOString().replace('T', ' ').substring(0, 19);

  ctx.font = '13px "JetBrains Mono", monospace';
  ctx.fillStyle = s.boxColor;
  ctx.globalAlpha = 0.9;

  // REC dot top-left
  ctx.fillStyle = Math.floor(elapsed * 2) % 2 === 0 ? '#ff3b3b' : '#440000';
  ctx.beginPath();
  ctx.arc(16, 18, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = s.boxColor;
  ctx.fillText('● REC', 26, 22);

  // cam label
  ctx.fillText(s.camLabel, 70, 22);

  // timestamp top-right
  const tsText = ts;
  ctx.textAlign = 'right';
  ctx.fillText(tsText, W - 12, 22);
  ctx.textAlign = 'left';

  // free text bottom
  ctx.fillStyle = s.boxColor;
  ctx.globalAlpha = 0.85;
  ctx.fillText(s.freeText, 12, H - 14);

  // frame counter bottom-right
  ctx.textAlign = 'right';
  ctx.fillStyle = '#5a6b5f';
  ctx.fillText('FRAME ' + Math.floor(elapsed * 30).toString().padStart(6, '0'), W - 12, H - 14);
  ctx.textAlign = 'left';

  // crosshair center
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

  // vignette
  const grad = ctx.createRadialGradient(W / 2, H / 2, H * 0.3, W / 2, H / 2, H * 0.85);
  grad.addColorStop(0, 'rgba(0,0,0,0)');
  grad.addColorStop(1, 'rgba(0,0,0,0.55)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, W, H);
}
