"use client";

import React, { useState, useRef, useEffect } from "react";
import { Sliders, Camera, ArrowRightCircle } from "lucide-react";

interface SynthCameraSimulatorProps {
  onSendToStudio: (simulatedLines: string[]) => void;
}

export const SynthCameraSimulator: React.FC<SynthCameraSimulatorProps> = ({ onSendToStudio }) => {
  const [text, setText] = useState<string>(
    `کٲشِر زَبانہِ ہٕنٛز تَوٲریخ چھےٚ واریاہ پرٛٲنؠ۔\nاَتھ مَنٛز چھےٚ شیٚنٛٹھ، شیو مَت تہٕ صُوفی شٲعِری ہٕنٛز خوٗشبوٗ۔\nآزاد کٔشیٖر تہٕ وادیِ کٔشیٖر کؠن لوٗکَن ہُنٛد فخر۔`
  );
  const [curve, setCurve] = useState<number>(28);
  const [shadow, setShadow] = useState<number>(45);
  const [moire, setMoire] = useState<number>(35);
  const [kerning, setKerning] = useState<number>(1.1);
  const [noise, setNoise] = useState<number>(15);

  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;

    // 1. Paper base texture
    ctx.fillStyle = "#fefce8";
    ctx.fillRect(0, 0, w, h);

    const lines = text.trim().split("\n");

    // 2. Render Kashmiri Text with 3D Cylindrical Spine Sine-Wave Curvature
    ctx.save();
    ctx.fillStyle = "#1e293b";
    ctx.font = 'bold 24px "Noto Nastaliq Urdu", serif';
    ctx.direction = "rtl";
    ctx.textAlign = "right";

    lines.forEach((lineStr, lineIdx) => {
      const baseY = 100 + lineIdx * 90;
      const words = lineStr.trim().split(/\s+/).filter(Boolean);
      let currX = w - 80;

      words.forEach((word) => {
        // Spine curve calculation: y' = y + A * sin(pi * x / W)
        const curveOffset = Math.sin((currX / w) * Math.PI) * curve;
        ctx.fillText(word, currX, baseY + curveOffset);
        currX -= ctx.measureText(word).width + 20 * kerning;
      });
    });
    ctx.restore();

    // 3. Parabolic Hand Shadow Gradient
    if (shadow > 0) {
      const shadowGrad = ctx.createRadialGradient(w * 0.25, h * 0.3, 10, w * 0.25, h * 0.3, 350);
      shadowGrad.addColorStop(0, `rgba(15, 23, 42, ${shadow / 130})`);
      shadowGrad.addColorStop(1, "rgba(15, 23, 42, 0)");
      ctx.fillStyle = shadowGrad;
      ctx.fillRect(0, 0, w, h);
    }

    // 4. Screen LCD Moiré Grid
    if (moire > 0) {
      ctx.save();
      ctx.fillStyle = `rgba(6, 182, 212, ${moire / 350})`;
      for (let x = 0; x < w; x += 3) {
        ctx.fillRect(x, 0, 1, h);
      }
      ctx.restore();
    }

    // 5. ISO Sensor Noise
    if (noise > 0) {
      const imgData = ctx.getImageData(0, 0, w, h);
      const d = imgData.data;
      const noiseFactor = noise * 0.4;
      for (let i = 0; i < d.length; i += 4) {
        const n = (Math.random() - 0.5) * noiseFactor;
        d[i] = Math.min(255, Math.max(0, d[i] + n));
        d[i + 1] = Math.min(255, Math.max(0, d[i + 1] + n));
        d[i + 2] = Math.min(255, Math.max(0, d[i + 2] + n));
      }
      ctx.putImageData(imgData, 0, 0);
    }
  }, [text, curve, shadow, moire, kerning, noise]);

  const handleSend = () => {
    const lines = text.trim().split("\n").filter(Boolean);
    onSendToStudio(lines);
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-4 gap-4">
        <div>
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <Camera className="w-5 h-5 text-cyan-400" />
            SynthCamera-KS v2.0 Real-Time Optical Distortion Simulator
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Simulate 3D physical book spine curvature, smartphone hand shadows, and screen Moiré on Kashmiri Nastaliq.
          </p>
        </div>
        <button
          onClick={handleSend}
          className="px-4 py-2 rounded-xl bg-gradient-to-r from-cyan-500 to-emerald-500 text-slate-950 font-bold hover:brightness-110 transition flex items-center gap-2 text-xs shadow-lg shadow-cyan-500/20"
        >
          <ArrowRightCircle className="w-4 h-4" />
          <span>Send Simulated Scan to OCR Studio</span>
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls Column */}
        <div className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1.5">
              Input Kashmiri Text
            </label>
            <textarea
              rows={4}
              dir="rtl"
              lang="ks"
              value={text}
              onChange={(e) => setText(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-white font-nastaliq text-base focus:border-cyan-500 focus:outline-none"
            />
          </div>

          <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-4">
            <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider block">
              Physical & Sensor Parameters
            </span>

            {/* Spine curve */}
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-400">Book Spine 3D Curvature:</span>
                <span className="font-mono text-cyan-400">{curve}px</span>
              </div>
              <input
                type="range"
                min="0"
                max="60"
                value={curve}
                onChange={(e) => setCurve(parseInt(e.target.value))}
                className="w-full accent-cyan-500"
              />
            </div>

            {/* Shadow */}
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-400">Hand Shadow Gradient:</span>
                <span className="font-mono text-emerald-400">{shadow}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                value={shadow}
                onChange={(e) => setShadow(parseInt(e.target.value))}
                className="w-full accent-emerald-500"
              />
            </div>

            {/* Moiré */}
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-400">Screen LCD Subpixel Moiré:</span>
                <span className="font-mono text-amber-400">{moire}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                value={moire}
                onChange={(e) => setMoire(parseInt(e.target.value))}
                className="w-full accent-amber-500"
              />
            </div>

            {/* Kerning */}
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-400">Inter-Word Kerning Ratio:</span>
                <span className="font-mono text-purple-400">{kerning.toFixed(1)}x</span>
              </div>
              <input
                type="range"
                min="6"
                max="15"
                value={Math.round(kerning * 10)}
                onChange={(e) => setKerning(parseInt(e.target.value) / 10)}
                className="w-full accent-purple-500"
              />
            </div>

            {/* Noise */}
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-400">ISO Low-Light Noise:</span>
                <span className="font-mono text-pink-400">{noise}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="50"
                value={noise}
                onChange={(e) => setNoise(parseInt(e.target.value))}
                className="w-full accent-pink-500"
              />
            </div>
          </div>
        </div>

        {/* Canvas Preview Column */}
        <div className="lg:col-span-2 flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
              Simulated Camera Sensor Frame (300 DPI)
            </span>
            <span className="text-xs font-mono text-slate-400">800 × 520 px</span>
          </div>

          <div className="relative w-full aspect-[16/10] bg-slate-950 rounded-xl overflow-hidden border border-slate-800 flex items-center justify-center">
            <canvas ref={canvasRef} width={800} height={520} className="max-w-full max-h-full shadow-2xl rounded" />
          </div>

          <div className="p-3 bg-slate-950 rounded-xl border border-slate-800 text-xs font-mono text-slate-400 space-y-1">
            <span className="text-cyan-400 font-bold block">// SynthCamera-KS v2.0 Mathematical Model:</span>
            <p>
              Spine Warp: <span className="text-slate-200">y&apos; = y + A * sin(π * x / W)</span> | Illumination:{" "}
              <span className="text-slate-200">I_norm = clip((I / (GaussianBlur(I, 45) + ε)) * 240, 0, 255)</span>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
