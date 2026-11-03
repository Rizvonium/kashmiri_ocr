import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export async function GET() {
  return NextResponse.json({
    status: "healthy",
    in_browser_engine: "Tesseract.js (Urdu LSTM model)",
    cloud_gpu_configured: Boolean(process.env.GPU_WORKER_URL),
  });
}
