"""
Modal GPU Worker Deployment Definition for KoshurOCR v2.0
Deploys UTRNet-CoordAtt + KenLM pipeline on Modal.com with an A10G GPU.
Runs 50+ lines/second at minimal cost with automatic scale-to-zero.
"""

# Note: This file is used when deploying with `modal deploy python_engine/modal_app.py`
try:
    import modal

    app = modal.App("koshur-ocr-v2")

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch",
            "torchvision",
            "numpy",
            "pillow",
            "pydantic",
            "fastapi",
            "pyctcdecode",
            "huggingface_hub"
        )
        .apt_install("fonts-noto-core")
    )

    @app.cls(gpu="A10G", image=image, scaledown_window=60)
    class ModalKoshurOCR:
        @modal.enter()
        def load_models(self):
            print("Modal container started: loading UTRNet-CoordAtt weights & KenLM 5-gram...")
            from python_engine.api_server import pipeline
            self.pipeline = pipeline

        @modal.web_endpoint(method="POST")
        def ocr_endpoint(self, payload: dict):
            import base64
            from PIL import Image
            import io

            img_b64 = payload.get("image_base64", "")
            img_bytes = base64.b64decode(img_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            result = self.pipeline.process_page_image(
                img,
                apply_dewarp=payload.get("dewarp", True),
                apply_descreen=payload.get("descreen", True)
            )
            return result

except ImportError:
    pass
