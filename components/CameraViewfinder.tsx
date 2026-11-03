"use client";

import React, { useEffect, useRef, useState } from "react";
import { Camera, CameraOff, RefreshCw } from "lucide-react";
import { computeLaplacianVarianceFromCanvas } from "../lib/laplacian-focus";
import { canvasFromSource } from "../lib/image-tools";

interface CameraViewfinderProps {
  onCapture: (frame: HTMLCanvasElement) => void;
  onFallbackFile: (file: File) => void;
  disabled?: boolean;
}

const SHARP_THRESHOLD = 110;

function describeCameraError(err: unknown): string {
  const name = (err as DOMException)?.name;
  if (typeof window !== "undefined" && !window.isSecureContext) return "The camera only works over HTTPS.";
  if (name === "NotAllowedError") return "Camera permission was denied. Allow camera access in your browser's site settings and retry.";
  if (name === "NotFoundError" || name === "OverconstrainedError") return "No camera was found on this device.";
  if (name === "NotReadableError") return "The camera is being used by another app.";
  return "The camera could not be started.";
}

export const CameraViewfinder: React.FC<CameraViewfinderProps> = ({ onCapture, onFallbackFile, disabled }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const [status, setStatus] = useState<"starting" | "live" | "error">("starting");
  const [errorMsg, setErrorMsg] = useState("");
  const [variance, setVariance] = useState<number | null>(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let stream: MediaStream | null = null;
    let timer: ReturnType<typeof setInterval> | null = null;
    let cancelled = false;
    setStatus("starting");
    setVariance(null);

    (async () => {
      try {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error("unsupported");
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 } },
          audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        const video = videoRef.current!;
        video.srcObject = stream;
        await video.play().catch(() => undefined);
        setStatus("live");

        // Sharpness meter on a 256x256 centre crop of the live frame.
        const probe = document.createElement("canvas");
        probe.width = probe.height = 256;
        const pctx = probe.getContext("2d", { willReadFrequently: true })!;
        timer = setInterval(() => {
          const v = videoRef.current;
          if (!v || !v.videoWidth) return;
          const side = Math.min(v.videoWidth, v.videoHeight) / 2;
          pctx.drawImage(v, (v.videoWidth - side) / 2, (v.videoHeight - side) / 2, side, side, 0, 0, 256, 256);
          setVariance(Math.round(computeLaplacianVarianceFromCanvas(pctx, 256, 256, SHARP_THRESHOLD).variance));
        }, 250);
      } catch (err) {
        if (cancelled) return;
        setErrorMsg(describeCameraError(err));
        setStatus("error");
      }
    })();

    return () => {
      cancelled = true;
      stream?.getTracks().forEach((t) => t.stop());
      if (timer) clearInterval(timer);
    };
  }, [attempt]);

  const handleCapture = () => {
    const v = videoRef.current;
    if (!v || !v.videoWidth) return;
    onCapture(canvasFromSource(v, v.videoWidth, v.videoHeight));
  };

  const isSharp = variance !== null && variance >= SHARP_THRESHOLD;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 lg:p-6">
      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        capture="environment"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFallbackFile(f);
          e.target.value = "";
        }}
      />
      <div className="flex items-center gap-3 mb-4">
        <div className={`w-3 h-3 rounded-full ${status === "live" ? "bg-red-500 animate-pulse" : "bg-slate-600"}`} />
        <h3 className="text-sm font-semibold text-white">Camera</h3>
        <span className="text-xs text-slate-400">Fill the frame with the printed text and hold steady.</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="md:col-span-2 relative aspect-[4/3] bg-black rounded-xl overflow-hidden border border-slate-800 flex items-center justify-center">
          <video ref={videoRef} playsInline autoPlay muted className={`w-full h-full object-contain ${status === "live" ? "" : "hidden"}`} />
          {status === "live" && (
            <div className="absolute inset-6 border-2 border-cyan-400/50 rounded-lg pointer-events-none flex items-end justify-center p-3">
              <span
                className={`text-xs font-semibold px-3 py-1 rounded-full border ${
                  isSharp ? "bg-emerald-950/90 text-emerald-300 border-emerald-500/40" : "bg-slate-900/90 text-amber-300 border-amber-500/40"
                }`}
              >
                {variance === null ? "Measuring focus…" : isSharp ? "✓ Sharp — ready to capture" : "Hold steady / move closer…"}
              </span>
            </div>
          )}
          {status === "starting" && <p className="text-sm text-slate-400">Starting camera… allow access if your browser asks.</p>}
          {status === "error" && (
            <div className="text-center p-6 space-y-3">
              <CameraOff className="w-10 h-10 mx-auto text-slate-500" />
              <p className="text-sm text-amber-200">{errorMsg}</p>
              <div className="flex flex-wrap justify-center gap-2">
                <button onClick={() => setAttempt((a) => a + 1)} className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs flex items-center gap-1.5">
                  <RefreshCw className="w-3.5 h-3.5" /> Try again
                </button>
                <button onClick={() => fileRef.current?.click()} className="px-3 py-1.5 rounded-lg bg-cyan-700 hover:bg-cyan-600 text-xs text-white">
                  Take / choose a photo instead
                </button>
              </div>
            </div>
          )}
        </div>

        <div className="flex flex-col justify-between gap-4">
          <div>
            <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider block mb-1">Focus meter</span>
            <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-slate-400">Sharpness (Laplacian variance):</span>
                <span className={`font-mono font-bold ${isSharp ? "text-emerald-400" : "text-amber-400"}`}>{variance ?? "—"}</span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                <div
                  className={`h-full transition-all duration-300 ${isSharp ? "bg-emerald-500" : "bg-amber-500"}`}
                  style={{ width: `${Math.min(100, ((variance ?? 0) / (SHARP_THRESHOLD * 2)) * 100)}%` }}
                />
              </div>
              <p className="text-[11px] text-slate-400">
                Above {SHARP_THRESHOLD} the small Kashmiri diacritics are usually crisp enough to read.
              </p>
            </div>
          </div>

          <div className="space-y-2">
            <button
              onClick={handleCapture}
              disabled={status !== "live" || disabled}
              className="w-full py-3.5 px-4 rounded-xl bg-gradient-to-r from-cyan-500 to-emerald-500 text-slate-950 font-bold hover:brightness-110 active:scale-[0.98] transition flex items-center justify-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <Camera className="w-5 h-5" />
              Capture & OCR
            </button>
            <button onClick={() => fileRef.current?.click()} disabled={disabled} className="w-full text-xs text-slate-400 hover:text-cyan-300 disabled:opacity-40">
              or take a photo with your phone&apos;s camera app
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
