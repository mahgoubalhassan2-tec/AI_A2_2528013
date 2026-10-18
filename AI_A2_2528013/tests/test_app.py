
from fastapi.testclient import TestClient
from app import app

# إنشاء عميل لاختبار API دون فتح المتصفح
client = TestClient(app)

# بيانات صحيحة نستخدمها في الاختبارات
VALID_DATA = {
    "rainfall_mm": 1000,
    "avg_temp_c": 20,
    "soil_moisture_pct": 50,
    "fertilizer_kg": 100,
    "field_size_ha": 2,
    "season": "A",
    "cooperative_support": 1
}


# الاختبار 1: التأكد من أن الخدمة تعمل
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# الاختبار 2: التأكد من معلومات النموذج
def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json()["verification_code"] == "AI-A2-8013"


# الاختبار 3: التنبؤ باستخدام بيانات صحيحة
def test_valid_prediction():
    response = client.post("/predict", json=VALID_DATA)

    assert response.status_code == 200
    result = response.json()

    assert "estimated_yield_tons" in result
    assert "high_risk" in result
    assert "cluster" in result


# الاختبار 4: رفض البيانات التي ينقصها حقل
def test_missing_field():
    data = VALID_DATA.copy()
    del data["rainfall_mm"]

    response = client.post("/predict", json=data)
    assert response.status_code == 422


# الاختبار 5: رفض نوع بيانات خاطئ
def test_wrong_type():
    data = VALID_DATA.copy()
    data["rainfall_mm"] = "hello"

    response = client.post("/predict", json=data)
    assert response.status_code == 422


# الاختبار 6: رفض قيمة خارج النطاق
def test_out_of_range():
    data = VALID_DATA.copy()
    data["rainfall_mm"] = 2000

    response = client.post("/predict", json=data)
    assert response.status_code == 422


# الاختبار 7: التأكد من ثبات نتيجة التنبؤ
def test_reproducibility():
    first = client.post("/predict", json=VALID_DATA)
    second = client.post("/predict", json=VALID_DATA)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()


def test_extra_field_rejected():
    data = VALID_DATA.copy()
    data["unexpected"] = 1
    response = client.post("/predict", json=data)
    assert response.status_code == 422


def test_model_info_discloses_synthetic_data():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert "synthetic" in response.json()["data_status"]
