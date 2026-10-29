import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import AliasChoices, BaseModel, Field, field_validator

# ==============================================================================
# KONFIGURASI DAN PATH MODEL
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "lightgbm_liver_model.pkl"

# State global penyimpanan model & metadata
model_bundle: Dict[str, Any] = {
    "model": None,
    "feature_columns": [],
    "target_column": "Result",
    "is_loaded": False,
    "load_time": None,
}


def load_ml_model() -> None:
    """Memuat model LightGBM dari file pickle bundle."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"File model '{MODEL_PATH.name}' tidak ditemukan di {BASE_DIR}. "
            "Pastikan file pickle telah tersedia."
        )

    bundle = joblib.load(MODEL_PATH)
    if isinstance(bundle, dict) and "model" in bundle:
        model_bundle["model"] = bundle["model"]
        model_bundle["feature_columns"] = bundle.get(
            "feature_columns",
            [
                "Age of the patient",
                "Gender of the patient",
                "Total Bilirubin",
                "Direct Bilirubin",
                "Alkphos Alkaline Phosphotase",
                "Sgpt Alamine Aminotransferase",
                "Sgot Aspartate Aminotransferase",
                "Total Protiens",
                "ALB Albumin",
                "A/G Ratio Albumin and Globulin Ratio",
            ],
        )
        model_bundle["target_column"] = bundle.get("target_column", "Result")
    else:
        # Fallback jika file pickle langsung berupa model object
        model_bundle["model"] = bundle
        model_bundle["feature_columns"] = getattr(
            bundle,
            "feature_name_",
            [
                "Age of the patient",
                "Gender of the patient",
                "Total Bilirubin",
                "Direct Bilirubin",
                "Alkphos Alkaline Phosphotase",
                "Sgpt Alamine Aminotransferase",
                "Sgot Aspartate Aminotransferase",
                "Total Protiens",
                "ALB Albumin",
                "A/G Ratio Albumin and Globulin Ratio",
            ],
        )

    model_bundle["is_loaded"] = True
    model_bundle["load_time"] = datetime.now(timezone.utc).isoformat()
    print(f"Model berhasil dimuat dari: {MODEL_PATH}")
    print(f"Fitur yang digunakan ({len(model_bundle['feature_columns'])}): {model_bundle['feature_columns']}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event handler untuk inisialisasi dan cleanup FastAPI."""
    try:
        load_ml_model()
    except Exception as exc:
        print(f"Peringatan saat memuat model: {exc}")
    yield


# ==============================================================================
# INISIALISASI APLIKASI FASTAPI
# ==============================================================================
app = FastAPI(
    title="Liver Disease Prediction API",
    description=(
        "API untuk mendeteksi risiko penyakit liver (hati) berdasarkan hasil uji laboratorium klinis "
        "menggunakan model Machine Learning LightGBM Classifier.\n\n"
        "- **Kelas 1**: Indikasi Penyakit Hati (Liver Patient)\n"
        "- **Kelas 2**: Normal / Bebas Penyakit Hati (Non-Liver Patient)"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Konfigurasi CORS agar API dapat diakses oleh frontend (Streamlit, React, Vue, Web Apps)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# PYDANTIC SCHEMAS (REQUEST & RESPONSE)
# ==============================================================================
class PatientInput(BaseModel):
    """Skema data input pasien dengan dukungan berbagai alias nama kolom."""

    age: int = Field(
        ...,
        ge=1,
        le=120,
        description="Usia pasien dalam tahun (1 - 120)",
        examples=[45],
        validation_alias=AliasChoices("age", "Age", "Age of the patient", "age_of_the_patient"),
    )
    gender: str = Field(
        ...,
        description="Jenis kelamin: 'Male' / 'Female' (bisa juga 'Pria'/'Wanita' atau 'Laki-laki'/'Perempuan')",
        examples=["Male"],
        validation_alias=AliasChoices("gender", "Gender", "Gender of the patient", "gender_of_the_patient", "sex"),
    )
    total_bilirubin: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Kadar Total Bilirubin dalam darah (mg/dL)",
        examples=[0.8],
        validation_alias=AliasChoices("total_bilirubin", "Total Bilirubin", "total_bilirubin", "tb", "TB"),
    )
    direct_bilirubin: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Kadar Direct (Conjugated) Bilirubin dalam darah (mg/dL)",
        examples=[0.2],
        validation_alias=AliasChoices("direct_bilirubin", "Direct Bilirubin", "direct_bilirubin", "db", "DB"),
    )
    alkaline_phosphotase: float = Field(
        ...,
        ge=0.0,
        le=3000.0,
        description="Kadar Alkphos Alkaline Phosphotase (IU/L)",
        examples=[200.0],
        validation_alias=AliasChoices(
            "alkaline_phosphotase",
            "Alkphos Alkaline Phosphotase",
            "alkphos",
            "alp",
            "ALP",
            "alkaline_phosphatase",
        ),
    )
    alamine_aminotransferase: float = Field(
        ...,
        ge=0.0,
        le=3000.0,
        description="Kadar Alamine Aminotransferase / SGPT (IU/L)",
        examples=[30.0],
        validation_alias=AliasChoices(
            "alamine_aminotransferase",
            "Sgpt Alamine Aminotransferase",
            "sgpt",
            "SGPT",
            "alt",
            "ALT",
        ),
    )
    aspartate_aminotransferase: float = Field(
        ...,
        ge=0.0,
        le=5000.0,
        description="Kadar Aspartate Aminotransferase / SGOT (IU/L)",
        examples=[35.0],
        validation_alias=AliasChoices(
            "aspartate_aminotransferase",
            "Sgot Aspartate Aminotransferase",
            "sgot",
            "SGOT",
            "ast",
            "AST",
        ),
    )
    total_proteins: float = Field(
        ...,
        ge=0.0,
        le=20.0,
        description="Kadar Total Protein serum (g/dL)",
        examples=[6.5],
        validation_alias=AliasChoices(
            "total_proteins",
            "Total Protiens",
            "Total Proteins",
            "total_protiens",
            "total_proteins",
            "tp",
            "TP",
        ),
    )
    albumin: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Kadar Albumin serum (g/dL)",
        examples=[3.2],
        validation_alias=AliasChoices("albumin", "ALB Albumin", "alb", "ALB", "Albumin"),
    )
    albumin_globulin_ratio: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="Rasio Albumin dan Globulin (A/G Ratio). Jika tidak diisi, akan dihitung otomatis dari protein & albumin.",
        examples=[0.9],
        validation_alias=AliasChoices(
            "albumin_globulin_ratio",
            "A/G Ratio Albumin and Globulin Ratio",
            "ag_ratio",
            "agr",
            "AGR",
        ),
    )

    @field_validator("gender", mode="before")
    @classmethod
    def normalize_gender(cls, value: Any) -> str:
        """Normalisasi format gender ke 'Male' atau 'Female' sesuai training data."""
        if not isinstance(value, str):
            raise ValueError("Gender harus berupa teks ('Male' atau 'Female').")
        val = value.strip().lower()
        if val in ["male", "m", "pria", "laki-laki", "lakilaki", "l"]:
            return "Male"
        if val in ["female", "f", "wanita", "perempuan", "p", "w"]:
            return "Female"
        raise ValueError(
            f"Nilai gender '{value}' tidak valid. Gunakan 'Male'/'Female' atau 'Pria'/'Wanita'."
        )


class PredictionResult(BaseModel):
    """Hasil prediksi dan analisa risiko untuk satu pasien."""

    prediction_raw: int = Field(..., description="Kelas prediksi model mentah: 1 (Sakit) atau 2 (Sehat)")
    diagnosis: str = Field(..., description="Deskripsi diagnosis: 'Indikasi Penyakit Hati' atau 'Normal / Sehat'")
    has_liver_disease: bool = Field(..., description="True jika terindikasi penyakit hati, False jika sehat")
    confidence: float = Field(..., description="Skor probabilitas keyakinan model (0.0 - 1.0)")
    risk_level: str = Field(..., description="Tingkat risiko klinis: 'Tinggi', 'Sedang', atau 'Rendah'")
    probabilities: Dict[str, float] = Field(
        ...,
        description="Probabilitas per kelas: 'liver_disease' (kelas 1) dan 'healthy' (kelas 2)",
    )
    recommendation: str = Field(..., description="Rekomendasi medis awal untuk pasien")


class SinglePredictionResponse(BaseModel):
    """Respons lengkap untuk prediksi pasien tunggal."""

    status: str = "success"
    data: PredictionResult
    patient_input: Dict[str, Any]
    timestamp: str


class BatchPredictionRequest(BaseModel):
    """Permintaan prediksi batch untuk beberapa pasien sekaligus."""

    patients: List[PatientInput] = Field(..., min_length=1, max_length=500)


class BatchPredictionResponse(BaseModel):
    """Respons untuk prediksi batch."""

    status: str = "success"
    total_samples: int
    summary: Dict[str, int]
    results: List[PredictionResult]
    timestamp: str


# ==============================================================================
# FUNGSI HELPER PREDIKSI
# ==============================================================================
def prepare_dataframe(patients: List[PatientInput]) -> pd.DataFrame:
    """Mengonversi data input pasien menjadi DataFrame yang cocok dengan ekspektasi LightGBM."""
    rows = []
    for p in patients:
        ag_ratio = p.albumin_globulin_ratio
        if ag_ratio is None or np.isnan(ag_ratio):
            globulin = max(p.total_proteins - p.albumin, 0.1)
            ag_ratio = round(p.albumin / globulin, 2)

        # Susun baris sesuai nama kolom saat training
        row = {
            "Age of the patient": p.age,
            "Gender of the patient": p.gender,
            "Total Bilirubin": p.total_bilirubin,
            "Direct Bilirubin": p.direct_bilirubin,
            "Alkphos Alkaline Phosphotase": p.alkaline_phosphotase,
            "Sgpt Alamine Aminotransferase": p.alamine_aminotransferase,
            "Sgot Aspartate Aminotransferase": p.aspartate_aminotransferase,
            "Total Protiens": p.total_proteins,
            "ALB Albumin": p.albumin,
            "A/G Ratio Albumin and Globulin Ratio": ag_ratio,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    # Ubah gender menjadi categorical dengan kategori ['Female', 'Male']
    df["Gender of the patient"] = pd.Categorical(
        df["Gender of the patient"], categories=["Female", "Male"]
    )
    return df


def generate_prediction_result(
    pred_raw: int, probas: np.ndarray
) -> PredictionResult:
    """Menghitung diagnosis, probabilitas, tingkat risiko, dan rekomendasi."""
    # Model classes_: [1, 2] -> index 0: kelas 1 (Liver Disease), index 1: kelas 2 (Healthy)
    prob_liver = float(probas[0])
    prob_healthy = float(probas[1])

    is_liver_patient = int(pred_raw) == 1

    if is_liver_patient:
        diagnosis = "Indikasi Penyakit Hati (Liver Disease Detected)"
        confidence = round(prob_liver, 4)
        if prob_liver >= 0.75:
            risk_level = "Tinggi (High Risk)"
            recommendation = (
                "Hasil tes menunjukkan indikasi kuat gangguan fungsi hati. Disarankan segera "
                "berkonsultasi dengan dokter spesialis gastroenterologi-hepatologi untuk pemeriksaan lanjutan (USG Abdomen, panel hepatitis, atau biopsi)."
            )
        else:
            risk_level = "Sedang (Moderate Risk)"
            recommendation = (
                "Hasil tes menunjukkan potensi risiko penyakit hati. Disarankan mengulang pemeriksaan laboratorium "
                "fungsi hati dalam 2-4 minggu dan konsultasi dengan dokter."
            )
    else:
        diagnosis = "Normal / Bebas Penyakit Hati (Healthy)"
        confidence = round(prob_healthy, 4)
        if prob_liver < 0.25:
            risk_level = "Rendah (Low Risk)"
            recommendation = (
                "Profil fungsi hati berada dalam rentang normal dan sehat. Pertahankan pola hidup sehat, "
                "konsumsi makanan bergizi seimbang, dan lakukan skrining rutin tahunan."
            )
        else:
            risk_level = "Sedang (Borderline Risk)"
            recommendation = (
                "Hasil tes secara keseluruhan belum mengindikasikan penyakit hati, namun beberapa parameter mendekati ambang batas. "
                "Jaga pola makan, hindari alkohol/obat-obatan hepatotoksik, dan lakukan evaluasi berkala."
            )

    return PredictionResult(
        prediction_raw=int(pred_raw),
        diagnosis=diagnosis,
        has_liver_disease=is_liver_patient,
        confidence=confidence,
        risk_level=risk_level,
        probabilities={
            "liver_disease": round(prob_liver, 4),
            "healthy": round(prob_healthy, 4),
        },
        recommendation=recommendation,
    )


# ==============================================================================
# ROUTE ENDPOINTS
# ==============================================================================
@app.get("/", tags=["General"])
async def root():
    """Halaman beranda API yang menampilkan ringkasan layanan dan dokumentasi."""
    return {
        "app_name": "Liver Disease Prediction API",
        "status": "online",
        "model_loaded": model_bundle["is_loaded"],
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "endpoints": {
            "health_check": "GET /health",
            "model_info": "GET /model-info",
            "predict_single": "POST /predict",
            "predict_batch": "POST /predict/batch",
        },
        "description": "API inferensi Machine Learning berbasis LightGBM untuk prediksi risiko penyakit hati.",
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Memeriksa kesehatan server dan status model machine learning."""
    if not model_bundle["is_loaded"] or model_bundle["model"] is None:
        try:
            load_ml_model()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model belum berhasil dimuat: {str(exc)}",
            )

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_path": str(MODEL_PATH),
        "load_time": model_bundle["load_time"],
        "server_time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/model-info", tags=["Model"])
async def get_model_info():
    """Mengembalikan informasi detail arsitektur model dan daftar fitur."""
    if not model_bundle["is_loaded"] or model_bundle["model"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model belum dimuat di memori.",
        )

    model = model_bundle["model"]
    classes = getattr(model, "classes_", [1, 2])
    params = getattr(model, "get_params", lambda: {})()

    return {
        "model_type": type(model).__name__,
        "target_column": model_bundle["target_column"],
        "classes": [int(c) for c in classes],
        "class_mapping": {
            1: "Liver Disease Patient (Positif Penyakit Hati)",
            2: "Non-Liver Patient (Negatif / Sehat)",
        },
        "feature_count": len(model_bundle["feature_columns"]),
        "features": model_bundle["feature_columns"],
        "hyperparameters": {
            k: v for k, v in params.items() if k in ["n_estimators", "learning_rate", "num_leaves", "max_depth", "random_state"]
        },
    }


@app.post(
    "/predict",
    response_model=SinglePredictionResponse,
    tags=["Prediction"],
    summary="Prediksi risiko penyakit hati untuk 1 pasien",
)
async def predict_single(patient: PatientInput):
    """Menerima parameter uji lab seorang pasien dan mengembalikan hasil prediksi diagnosis dan risiko."""
    if not model_bundle["is_loaded"] or model_bundle["model"] is None:
        try:
            load_ml_model()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model tidak siap: {str(exc)}",
            )

    try:
        model = model_bundle["model"]
        df_input = prepare_dataframe([patient])
        pred = model.predict(df_input)[0]
        prob = model.predict_proba(df_input)[0]

        result = generate_prediction_result(pred, prob)

        return SinglePredictionResponse(
            status="success",
            data=result,
            patient_input=patient.model_dump(),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Terjadi kesalahan saat inferensi: {str(exc)}",
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Prediksi risiko penyakit hati untuk banyak pasien sekaligus (Batch)",
)
async def predict_batch(payload: BatchPredictionRequest):
    """Menerima daftar data beberapa pasien dan memproses prediksi secara batch."""
    if not model_bundle["is_loaded"] or model_bundle["model"] is None:
        try:
            load_ml_model()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model tidak siap: {str(exc)}",
            )

    try:
        model = model_bundle["model"]
        df_input = prepare_dataframe(payload.patients)
        preds = model.predict(df_input)
        probs = model.predict_proba(df_input)

        results: List[PredictionResult] = []
        positive_count = 0
        negative_count = 0

        for pred, prob in zip(preds, probs):
            res = generate_prediction_result(pred, prob)
            results.append(res)
            if res.has_liver_disease:
                positive_count += 1
            else:
                negative_count += 1

        return BatchPredictionResponse(
            status="success",
            total_samples=len(results),
            summary={
                "liver_disease_cases": positive_count,
                "healthy_cases": negative_count,
            },
            results=results,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Terjadi kesalahan saat pemrosesan batch: {str(exc)}",
        )


# ==============================================================================
# ENTRYPOINT SERVER
# ==============================================================================
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
