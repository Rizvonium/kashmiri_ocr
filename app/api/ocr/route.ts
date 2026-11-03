import { NextRequest, NextResponse } from "next/server";

export const maxDuration = 60;

/**
 * Forwards an image to the GPU OCR worker configured in GPU_WORKER_URL
 * (for example the Modal app in python_engine/modal_app.py).
 * Body: { image: <base64 JPEG/PNG without data: prefix>, dewarp?: boolean, descreen?: boolean }
 */
export async function POST(req: NextRequest) {
  const gpuWorkerUrl = process.env.GPU_WORKER_URL;
  if (!gpuWorkerUrl) {
    return NextResponse.json(
      { status: "error", message: "Cloud OCR is not configured. Set GPU_WORKER_URL in your Vercel project settings." },
      { status: 503 }
    );
  }

  let body: { image?: unknown; dewarp?: unknown; descreen?: unknown };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ status: "error", message: "Send JSON with an 'image' field." }, { status: 400 });
  }
  if (typeof body.image !== "string" || !body.image) {
    return NextResponse.json({ status: "error", message: "Missing 'image' (base64)." }, { status: 400 });
  }

  try {
    const remoteRes = await fetch(gpuWorkerUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_base64: body.image, dewarp: body.dewarp !== false, descreen: body.descreen !== false }),
      signal: AbortSignal.timeout(55_000),
    });
    const json = await remoteRes.json().catch(() => null);
    if (!remoteRes.ok || !json) {
      return NextResponse.json(
        { status: "error", message: `GPU worker returned HTTP ${remoteRes.status}.` },
        { status: 502 }
      );
    }
    return NextResponse.json(json);
  } catch (err) {
    return NextResponse.json(
      { status: "error", message: `GPU worker unreachable: ${(err as Error).message}` },
      { status: 502 }
    );
  }
}
