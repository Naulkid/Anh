"""
test_m1_forecasting.py — Unit tests cho Module M1
===================================================
Kiểm tra tính đúng đắn của hàm Cobb-Douglas, ước lượng TFP,
phân rã tăng trưởng và dự báo kịch bản.

Chạy: pytest tests/test_m1_forecasting.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.m1_forecasting import CobbDouglasModel, ForecastResult
from src.config import COBB_DOUGLAS_PARAMS, SCENARIOS, INITIAL_CONDITIONS_2026


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def sample_macro_df():
    """DataFrame macro giả định cho test."""
    return pd.DataFrame({
        "year": [2020, 2021, 2022, 2023, 2024, 2025],
        "GDP_trillion_VND": [8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6],
        "K_capital_trillion": [16500, 17800, 19600, 21300, 23500, 25900],
        "L_labor_million": [53.6, 50.5, 51.7, 52.4, 52.9, 53.4],
        "D_digital_pct": [12.0, 12.7, 14.3, 16.5, 18.3, 19.5],
        "AI_tech_firms_thousand": [55.6, 60.2, 65.4, 67.0, 73.8, 80.1],
        "H_trained_labor_pct": [24.1, 26.1, 26.2, 27.0, 28.4, 29.2],
        "inflation_pct": [3.23, 1.84, 3.15, 3.25, 3.63, 3.20],
        "labor_productivity_million_VND": [117.4, 121.6, 148.1, 195.3, 221.9, 245.0],
    })


@pytest.fixture
def fitted_model(sample_macro_df):
    """Model đã được fit."""
    model = CobbDouglasModel()
    model.fit(sample_macro_df)
    return model


# ─────────────────────────────────────────────────────────────
# Test 1: Tham số Cobb-Douglas
# ─────────────────────────────────────────────────────────────

class TestCobbDouglasParams:
    """Kiểm tra điều kiện CRS và tính hợp lệ tham số."""

    def test_params_sum_to_one(self):
        """Tổng hệ số phải bằng 1 (CRS)."""
        total = sum(COBB_DOUGLAS_PARAMS.values())
        assert abs(total - 1.0) < 1e-9, f"CRS vi phạm: sum={total}"

    def test_all_params_positive(self):
        """Tất cả hệ số phải dương."""
        for name, val in COBB_DOUGLAS_PARAMS.items():
            assert val > 0, f"Hệ số {name}={val} phải > 0"

    def test_invalid_params_raise_error(self):
        """Tham số tổng ≠ 1 phải raise ValueError."""
        bad_params = {"alpha": 0.5, "beta": 0.5, "gamma": 0.1,
                      "delta": 0.1, "theta": 0.1}
        with pytest.raises(ValueError, match="CRS"):
            CobbDouglasModel(params=bad_params)

    def test_default_params_match_config(self):
        """Tham số mặc định phải khớp config."""
        model = CobbDouglasModel()
        assert model.alpha == COBB_DOUGLAS_PARAMS["alpha"]
        assert model.beta  == COBB_DOUGLAS_PARAMS["beta"]


# ─────────────────────────────────────────────────────────────
# Test 2: Tính sản lượng
# ─────────────────────────────────────────────────────────────

class TestComputeOutput:
    """Kiểm tra hàm compute_output."""

    def test_output_positive(self):
        """Sản lượng phải luôn dương với đầu vào dương."""
        model = CobbDouglasModel()
        y = model.compute_output(A=2.0, K=20000, L=52.0,
                                 D=15.0, AI=65.0, H=26.0)
        assert y > 0, "Sản lượng phải dương"

    def test_output_scales_with_A(self):
        """Tăng A gấp đôi → Y tăng gấp đôi (tuyến tính với A)."""
        model = CobbDouglasModel()
        kwargs = dict(K=20000, L=52.0, D=15.0, AI=65.0, H=26.0)
        y1 = model.compute_output(A=1.0, **kwargs)
        y2 = model.compute_output(A=2.0, **kwargs)
        assert abs(y2 / y1 - 2.0) < 1e-6, "Y không tuyến tính với A"

    def test_crs_property(self):
        """Tăng tất cả đầu vào λ lần → Y tăng λ lần (CRS)."""
        model = CobbDouglasModel()
        lam = 1.5
        y1 = model.compute_output(A=2.0, K=20000, L=52.0,
                                   D=15.0, AI=65.0, H=26.0)
        y2 = model.compute_output(A=2.0, K=20000*lam, L=52.0*lam,
                                   D=15.0*lam, AI=65.0*lam, H=26.0*lam)
        assert abs(y2 / y1 - lam) < 1e-4, f"CRS vi phạm: ratio={y2/y1:.4f} ≠ {lam}"

    def test_numpy_array_input(self):
        """Hàm phải hoạt động với numpy array đầu vào."""
        model = CobbDouglasModel()
        A   = np.array([2.0, 2.1, 2.2])
        K   = np.array([20000, 21000, 22000])
        L   = np.full(3, 52.0)
        D   = np.full(3, 15.0)
        AI  = np.full(3, 65.0)
        H   = np.full(3, 26.0)
        Y = model.compute_output(A, K, L, D, AI, H)
        assert Y.shape == (3,)
        assert np.all(Y > 0)


# ─────────────────────────────────────────────────────────────
# Test 3: Ước lượng TFP
# ─────────────────────────────────────────────────────────────

class TestFitTFP:
    """Kiểm tra quá trình fit và ước lượng TFP."""

    def test_fit_returns_self(self, sample_macro_df):
        """fit() phải trả về self để chain."""
        model = CobbDouglasModel()
        result = model.fit(sample_macro_df)
        assert result is model

    def test_tfp_shape(self, fitted_model, sample_macro_df):
        """TFP phải có đúng số phần tử."""
        assert len(fitted_model.tfp_hist) == len(sample_macro_df)

    def test_tfp_all_positive(self, fitted_model):
        """TFP phải dương tất cả."""
        assert np.all(fitted_model.tfp_hist > 0), "TFP phải > 0"

    def test_mape_reasonable(self, fitted_model):
        """MAPE fit không nên quá 10%."""
        assert fitted_model.mape_fit < 10.0, \
            f"MAPE quá cao: {fitted_model.mape_fit:.2f}%"

    def test_forecast_before_fit_raises(self):
        """forecast() trước fit() phải raise RuntimeError."""
        model = CobbDouglasModel()
        with pytest.raises(RuntimeError, match="fit"):
            model.forecast()


# ─────────────────────────────────────────────────────────────
# Test 4: Dự báo
# ─────────────────────────────────────────────────────────────

class TestForecast:
    """Kiểm tra chất lượng dự báo GDP."""

    def test_forecast_returns_correct_type(self, fitted_model):
        """forecast() phải trả về ForecastResult."""
        result = fitted_model.forecast(2026, 2030, "S1")
        assert isinstance(result, ForecastResult)

    def test_forecast_year_count(self, fitted_model):
        """Số năm dự báo phải đúng."""
        result = fitted_model.forecast(2026, 2035, "S5")
        assert len(result.years) == 10
        assert len(result.gdp) == 10

    def test_forecast_gdp_positive(self, fitted_model):
        """GDP dự báo phải dương."""
        for sid in ["S1", "S2", "S3", "S4", "S5"]:
            result = fitted_model.forecast(2026, 2030, sid)
            assert np.all(result.gdp > 0), f"GDP âm ở kịch bản {sid}"

    def test_scenario_s3_outperforms_s1_ai_capacity(self, fitted_model):
        """S3 (AI dẫn dắt) phải có AI capacity cao hơn S1 (Truyền thống).
        Lưu ý kinh tế: S1 có thể cao hơn GDP ngắn hạn do α_K=0.33 > δ_AI=0.08,
        nhưng S3 đạt AI capacity vượt trội nhờ ưu tiên 45% cho AI."""
        r1 = fitted_model.forecast(2026, 2030, "S1")
        r3 = fitted_model.forecast(2026, 2030, "S3")
        assert r3.ai_cap[-1] > r1.ai_cap[-1], \
            "S3 phải có AI capacity cao hơn S1 (45% vs 10% AI allocation)"

    def test_forecast_digital_bounded(self, fitted_model):
        """Chỉ số số hóa D không được vượt 45% (trần thiết kế)."""
        result = fitted_model.forecast(2026, 2035, "S2")
        assert np.all(result.digital <= 45.0), "D vượt trần 45%"


# ─────────────────────────────────────────────────────────────
# Test 5: Phân rã tăng trưởng
# ─────────────────────────────────────────────────────────────

class TestGrowthAccounting:
    """Kiểm tra tính đúng đắn của phân rã tăng trưởng."""

    def test_growth_accounting_row_count(self, fitted_model, sample_macro_df):
        """Số hàng = số năm - 1."""
        df = fitted_model.growth_accounting()
        assert len(df) == len(sample_macro_df) - 1

    def test_contributions_sum_approximately(self, fitted_model):
        """Tổng đóng góp phải ≈ Δln(Y) (sai số < 0.5 pp)."""
        df = fitted_model.growth_accounting()
        for _, row in df.iterrows():
            total_contrib = (
                row["contrib_K_pp"] + row["contrib_L_pp"]
                + row["contrib_D_pp"] + row["contrib_AI_pp"]
                + row["contrib_H_pp"] + row["contrib_TFP_pp"]
            )
            diff = abs(total_contrib - row["gdp_growth_pct"])
            assert diff < 0.5, \
                f"Phân rã tăng trưởng sai: year={row['year']}, diff={diff:.4f}"

    def test_compare_scenarios_returns_dataframe(self, fitted_model):
        """compare_scenarios() phải trả về DataFrame."""
        df = fitted_model.compare_scenarios(year_end=2030)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5  # 5 kịch bản

    def test_compare_scenarios_has_required_columns(self, fitted_model):
        """DataFrame so sánh phải có cột bắt buộc."""
        df = fitted_model.compare_scenarios(year_end=2030)
        assert "scenario_id" in df.columns
        assert "avg_growth_pct" in df.columns
        assert "gdp_gain_vs_S1_pct" in df.columns
