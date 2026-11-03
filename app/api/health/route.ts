import { NextResponse } from "next/server";

export async function GET() {
  return NextResponse.json({
    status: "healthy",
    version: "2.0.0",
    engine: "UTRNet-CoordAtt + KenLM",
    features: [
      "Client-side 300 DPI PDF rendering",
      "Division illumination normalization",
      "2D TPS book spine dewarping",
      "Bilateral screen moiré descreening",
      "Real-time Laplacian camera focus gate",
      "Canonical NFC Kashmiri orthography normalization"
    ],
    supported_scripts: ["Kashmiri Nastaliq", "Gulmarg Nastaleeq", "SIL Awami Nastaliq", "Afan Koshur Naksh"]
  });
}
