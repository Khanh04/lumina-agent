from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env into os.environ so provider SDKs that read env directly (e.g. the Gemini
# provider's GOOGLE_API_KEY) work in local dev. No-op in containers where env is set.
load_dotenv()





class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # gemini-2.5-flash 404s for new keys and flash-latest is often 503-throttled;
    # flash-lite-latest is the available, fast, cost-efficient tier (the doc's original pick).
    model_name: str = "google:gemini-flash-lite-latest"
    redis_url: str = "redis://localhost:6379/0"
    session_ttl_seconds: int = 3600
    # Comma-separated origins allowed via CORS. Only set when hosting the frontend on a
    # separate origin; dev (Vite proxy) and the default deploy (UI served by FastAPI) are same-origin.
    cors_allow_origins: str = ""
    # Segmentation models. Downloaded in the Dockerfile; for local dev fetch them once
    # (see README/verification) into these paths.
    segmenter_model_path: str = "app/cv/models/selfie_segmenter.tflite"  # MediaPipe preset regions
    # MobileSAM (ONNX) for click-to-select — much sharper masks than the MediaPipe interactive model.
    sam_encoder_path: str = "app/cv/models/mobile_sam.encoder.onnx"
    sam_decoder_path: str = "app/cv/models/sam_vit_h_4b8939.decoder.onnx"
    # YOLO-World open-vocab detector for text->region ("select by name").
    grounder_model_path: str = "app/cv/models/yolov8s-world.pt"


settings = Settings()
