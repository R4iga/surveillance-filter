'use client';

import { useEffect, useRef, useState, useCallback } from 'react';

/**
 * Loads the COCO-SSD model (TensorFlow.js) once and exposes a `detect` function.
 * Real AI object detection — finds people, cars, animals, etc.
 */

export interface Detection {
  bbox: [number, number, number, number]; // [x, y, w, h] in source pixels
  class: string;
  score: number;
}

export type DetectorStatus = 'idle' | 'loading' | 'ready' | 'error';

interface DetectedObject {
  bbox: [number, number, number, number];
  class: string;
  score: number;
}

interface ObjectDetection {
  detect(
    img: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement | ImageData,
    maxNumBoxes?: number,
    minScore?: number
  ): Promise<DetectedObject[]>;
  dispose(): void;
}

// Cache the loaded model at module scope so it's shared across hook instances
// and only downloaded once.
let cachedModel: ObjectDetection | null = null;
let loadingPromise: Promise<ObjectDetection | null> | null = null;

async function loadModel(): Promise<ObjectDetection | null> {
  if (cachedModel) return cachedModel;
  if (loadingPromise) return loadingPromise;
  loadingPromise = (async () => {
    const tf = await import('@tensorflow/tfjs');
    await tf.ready();
    const coco = await import('@tensorflow-models/coco-ssd');
    const model = (await coco.load({ base: 'lite_mobilenet_v2' })) as unknown as ObjectDetection;
    cachedModel = model;
    return model;
  })();
  try {
    return await loadingPromise;
  } finally {
    loadingPromise = null;
  }
}

export function useDetector() {
  const [status, setStatus] = useState<DetectorStatus>(cachedModel ? 'ready' : 'idle');
  const modelRef = useRef<ObjectDetection | null>(cachedModel);

  const ensureModel = useCallback(async (): Promise<ObjectDetection | null> => {
    if (modelRef.current) return modelRef.current;
    setStatus('loading');
    try {
      const m = await loadModel();
      modelRef.current = m;
      setStatus(m ? 'ready' : 'error');
      return m;
    } catch (e) {
      console.error('Model load failed', e);
      setStatus('error');
      return null;
    }
  }, []);

  const detect = useCallback(
    async (video: HTMLVideoElement): Promise<Detection[]> => {
      const model = await ensureModel();
      if (!model || video.readyState < 2) return [];
      try {
        const raw = await model.detect(video, 10, 0.45);
        return raw.map((d: DetectedObject) => ({
          bbox: d.bbox,
          class: d.class,
          score: d.score,
        }));
      } catch {
        return [];
      }
    },
    [ensureModel]
  );

  useEffect(() => {
    return () => {
      modelRef.current = null;
    };
  }, []);

  return { detect, status };
}
