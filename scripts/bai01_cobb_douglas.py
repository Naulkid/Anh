"""
Bài 1: Hàm sản xuất Cobb-Douglas mở rộng với AI và số hóa
=============================================================
Mô hình: Y_t = A_t * K^α * L^β * D^γ * AI^δ * H^θ
Với α+β+γ+δ+θ = 1 (lợi suất không đổi theo quy mô)

Yêu cầu:
    1.4.1 - Ước lượng TFP (A_t) từng năm, vẽ đồ thị
    1.4.2 - Dự báo Ŷ_t bằng A trung bình, tính MAPE
    1.4.3 - Phân rã tăng trưởng GDP 2020-2025
    1.4.4 - Dự báo GDP năm 2030 theo kịch bản cho trước
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────
# DỮ LIỆU VIỆT NAM 2020-2025
# ─────────────────────────────────────────
years = np.array([2020, 2021, 2022, 2023, 2024, 2025])

Y   = np.array([8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6])  # GDP (nghìn tỷ VND)
K   = np.array([16500,  17800,  19600,  21300,   23500,   25900])     # Vốn vật chất (nghìn tỷ)
L   = np.array([53.6,   50.5,   51.7,   52.4,    52.9,    53.4])      # Lao động (triệu người)
D   = np.array([12.0,   12.7,   14.3,   16.5,    18.3,    19.5])      # Kinh tế số/GDP (%)
AI  = np.array([55.6,   60.2,   65.4,   67.0,    73.8,    80.1])      # Doanh nghiệp số (nghìn DN)
H   = np.array([24.1,   26.1,   26.2,   27.0,    28.4,    29.2])      # Lao động qua đào tạo (%)

# Hệ số đề xuất (tổng = 1.00)
alpha = 0.33   # vốn vật chất
beta  = 0.42   # lao động
gamma = 0.10   # số hóa
delta = 0.08   # AI
theta = 0.07   # vốn nhân lực số

print("=" * 60)
print("BÀI 1: HÀM SẢN XUẤT COBB-DOUGLAS MỞ RỘNG")
print("=" * 60)
print(f"Tổng hệ số: α+β+γ+δ+θ = {alpha+beta+gamma+delta+theta:.2f}")

# ─────────────────────────────────────────
# CÂU 1.4.1: Ước lượng TFP (A_t) từng năm
# ─────────────────────────────────────────
print("\n--- Câu 1.4.1: Ước lượng TFP A_t ---")

# A_t = Y / (K^α * L^β * D^γ * AI^δ * H^θ)
denominator = (K**alpha) * (L**beta) * (D**gamma) * (AI**delta) * (H**theta)
A = Y / denominator

for i, yr in enumerate(years):
    print(f"  {yr}: A_t = {A[i]:.4f}")

# Nhận xét xu hướng TFP
A_mean = A.mean()
A_trend = np.polyfit(years, A, 1)
print(f"\nTFP trung bình 2020-2025: {A_mean:.4f}")
print(f"Xu hướng TFP (hệ số góc): {A_trend[0]:.6f}/năm")
if A_trend[0] > 0:
    print("→ TFP tăng: tăng trưởng dựa trên cải thiện năng suất (tích cực)")
else:
    print("→ TFP giảm: tăng trưởng chủ yếu từ đầu vào, chưa bền vững")

# ─────────────────────────────────────────
# CÂU 1.4.2: Dự báo Ŷ_t bằng A trung bình, tính MAPE
# ─────────────────────────────────────────
print("\n--- Câu 1.4.2: Dự báo GDP & MAPE ---")

Y_hat = A_mean * denominator  # Dự báo với A cố định = trung bình

MAPE = np.mean(np.abs((Y - Y_hat) / Y)) * 100
print(f"\nMAPE = {MAPE:.2f}%")
print(f"\n{'Năm':>6} {'Y thực':>12} {'Ŷ dự báo':>12} {'Sai số (%)':>12}")
print("-" * 46)
for i, yr in enumerate(years):
    err = abs(Y[i] - Y_hat[i]) / Y[i] * 100
    print(f"{yr:>6} {Y[i]:>12,.1f} {Y_hat[i]:>12,.1f} {err:>11.2f}%")

# ─────────────────────────────────────────
# CÂU 1.4.3: Phân rã tăng trưởng (Growth Accounting)
# ─────────────────────────────────────────
print("\n--- Câu 1.4.3: Phân rã tăng trưởng 2020-2025 ---")

# Tăng trưởng bình quân mỗi năm theo log-difference
# Δln(Y) = Δln(A) + α·Δln(K) + β·Δln(L) + γ·Δln(D) + δ·Δln(AI) + θ·Δln(H)
n = len(years) - 1  # số khoảng thời gian

dln_Y  = np.diff(np.log(Y))
dln_K  = np.diff(np.log(K))
dln_L  = np.diff(np.log(L))
dln_D  = np.diff(np.log(D))
dln_AI = np.diff(np.log(AI))
dln_H  = np.diff(np.log(H))
dln_A  = np.diff(np.log(A))  # TFP residual

# Đóng góp của từng yếu tố
contrib_K  = alpha * dln_K
contrib_L  = beta  * dln_L
contrib_D  = gamma * dln_D
contrib_AI = delta * dln_AI
contrib_H  = theta * dln_H
contrib_A  = dln_A  # TFP (phần dư)

# Bình quân toàn giai đoạn
avg_Y  = dln_Y.mean() * 100
avg_K  = contrib_K.mean()  * 100
avg_L  = contrib_L.mean()  * 100
avg_D  = contrib_D.mean()  * 100
avg_AI = contrib_AI.mean() * 100
avg_H  = contrib_H.mean()  * 100
avg_A  = contrib_A.mean()  * 100

print(f"\nTăng trưởng GDP bình quân/năm: {avg_Y:.2f}%")
print(f"\n{'Yếu tố':<20} {'Đóng góp (%/năm)':>18} {'Tỷ trọng (%)':>14}")
print("-" * 54)
factors = [
    ("Vốn vật chất (K)",     avg_K),
    ("Lao động (L)",          avg_L),
    ("Số hóa (D)",            avg_D),
    ("Năng lực AI (AI)",      avg_AI),
    ("Nhân lực số (H)",       avg_H),
    ("TFP (A)",               avg_A),
]
for name, val in factors:
    share = val / avg_Y * 100
    print(f"{name:<20} {val:>17.3f}%  {share:>12.1f}%")
print("-" * 54)
total = sum(v for _, v in factors)
print(f"{'Tổng':<20} {total:>17.3f}%  {total/avg_Y*100:>12.1f}%")

# ─────────────────────────────────────────
# CÂU 1.4.4: Dự báo GDP năm 2030
# ─────────────────────────────────────────
print("\n--- Câu 1.4.4: Dự báo GDP năm 2030 ---")

# Kịch bản: K và L tăng 6%/năm từ 2025, TFP tăng 1.2%/năm
horizon = 5  # 2025 → 2030

K_2025   = K[-1]
L_2025   = L[-1]
A_2025   = A[-1]

K_2030   = K_2025  * (1 + 0.06) ** horizon
L_2030   = L_2025  * (1 + 0.06) ** horizon
D_2030   = 30.0    # % GDP
AI_2030  = 100.0   # nghìn DN số
H_2030   = 35.0    # % lao động qua đào tạo
A_2030   = A_2025  * (1 + 0.012) ** horizon

Y_2030 = A_2030 * (K_2030**alpha) * (L_2030**beta) * (D_2030**gamma) * (AI_2030**delta) * (H_2030**theta)

print(f"\nGiả định kịch bản 2030:")
print(f"  K = {K_2030:,.0f} nghìn tỷ VND (+6%/năm)")
print(f"  L = {L_2030:.2f} triệu người (+6%/năm)")
print(f"  D = {D_2030}% GDP")
print(f"  AI = {AI_2030} nghìn DN số")
print(f"  H = {H_2030}% lao động qua đào tạo")
print(f"  A = {A_2030:.4f} (TFP tăng 1.2%/năm)")
print(f"\n→ GDP dự báo năm 2030: {Y_2030:,.1f} nghìn tỷ VND")
print(f"→ Tăng trưởng bình quân 2025-2030: {((Y_2030/Y[-1])**(1/horizon)-1)*100:.2f}%/năm")

# ─────────────────────────────────────────
# BIỂU ĐỒ TỔNG HỢP
# ─────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.suptitle("Bài 1: Hàm Sản xuất Cobb-Douglas Mở rộng — Việt Nam 2020-2025",
             fontsize=14, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# --- Plot 1: TFP theo năm ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(years, A, 'o-', color='#1565C0', linewidth=2.5, markersize=8, label='TFP thực tế')
trend_line = np.polyval(A_trend, years)
ax1.plot(years, trend_line, '--', color='#E53935', linewidth=1.5, alpha=0.7, label='Xu hướng')
ax1.fill_between(years, A, alpha=0.1, color='#1565C0')
ax1.set_title("TFP (A_t) theo năm", fontweight='bold')
ax1.set_xlabel("Năm")
ax1.set_ylabel("TFP")
ax1.legend()
ax1.grid(True, alpha=0.3)
for i, (yr, a_val) in enumerate(zip(years, A)):
    ax1.annotate(f'{a_val:.3f}', (yr, a_val), textcoords="offset points",
                 xytext=(0, 10), ha='center', fontsize=8)

# --- Plot 2: GDP thực tế vs Dự báo ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(years, Y,     'o-', color='#2E7D32', linewidth=2.5, markersize=8, label='GDP thực tế')
ax2.plot(years, Y_hat, 's--', color='#F57F17', linewidth=2,   markersize=7, label=f'GDP dự báo (MAPE={MAPE:.2f}%)')
ax2.scatter([2030], [Y_2030], s=150, marker='*', color='#C62828', zorder=5, label=f'Dự báo 2030: {Y_2030:,.0f}')
ax2.set_title("GDP: Thực tế vs Dự báo", fontweight='bold')
ax2.set_xlabel("Năm")
ax2.set_ylabel("GDP (nghìn tỷ VND)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# --- Plot 3: Phân rã tăng trưởng (biểu đồ cột xếp chồng theo năm) ---
ax3 = fig.add_subplot(gs[1, 0])
bar_years = years[1:]  # 2021-2025
colors_bar = ['#1565C0', '#7B1FA2', '#00838F', '#E65100', '#2E7D32', '#78909C']
labels_bar = ['K (vốn)', 'L (lao động)', 'D (số hóa)', 'AI', 'H (nhân lực)', 'TFP']
contribs   = [contrib_K, contrib_L, contrib_D, contrib_AI, contrib_H, contrib_A]

bottoms_pos = np.zeros(n)
bottoms_neg = np.zeros(n)
for contrib, color, label in zip(contribs, colors_bar, labels_bar):
    vals = contrib * 100
    pos_vals = np.where(vals > 0, vals, 0)
    neg_vals = np.where(vals < 0, vals, 0)
    ax3.bar(bar_years, pos_vals, bottom=bottoms_pos, color=color, label=label, alpha=0.85, width=0.6)
    ax3.bar(bar_years, neg_vals, bottom=bottoms_neg, color=color, alpha=0.85, width=0.6)
    bottoms_pos += pos_vals
    bottoms_neg += neg_vals

ax3.plot(bar_years, dln_Y * 100, 'ko-', linewidth=2, markersize=6, label='GDP growth thực')
ax3.axhline(0, color='black', linewidth=0.8)
ax3.set_title("Phân rã Tăng trưởng GDP (%/năm)", fontweight='bold')
ax3.set_xlabel("Năm")
ax3.set_ylabel("Đóng góp (%)")
ax3.legend(fontsize=7, loc='upper left')
ax3.grid(True, alpha=0.3, axis='y')

# --- Plot 4: Tỷ trọng đóng góp trung bình (pie/bar ngang) ---
ax4 = fig.add_subplot(gs[1, 1])
avg_contribs = [avg_K, avg_L, avg_D, avg_AI, avg_H, avg_A]
shares = [v / avg_Y * 100 for v in avg_contribs]
bars = ax4.barh(labels_bar, shares, color=colors_bar, alpha=0.85, edgecolor='white', linewidth=0.5)
ax4.axvline(0, color='black', linewidth=0.8)
ax4.set_title(f"Tỷ trọng đóng góp TB/năm\n(Tổng tăng trưởng = {avg_Y:.2f}%/năm)", fontweight='bold')
ax4.set_xlabel("Tỷ trọng (%)")
for bar, val in zip(bars, shares):
    xpos = bar.get_width() + 0.5 if bar.get_width() >= 0 else bar.get_width() - 0.5
    ax4.text(xpos, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', ha='left' if bar.get_width() >= 0 else 'right', fontsize=9)
ax4.grid(True, alpha=0.3, axis='x')

plt.savefig('/mnt/user-data/outputs/bai01_cobb_douglas_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ Biểu đồ đã lưu: bai01_cobb_douglas_results.png")

# ─────────────────────────────────────────
# THẢO LUẬN CHÍNH SÁCH (ngắn gọn)
# ─────────────────────────────────────────
print("\n--- Thảo luận chính sách ---")
print(f"""
a) TFP xu hướng {'TĂNG' if A_trend[0] > 0 else 'GIẢM'}: hệ số góc = {A_trend[0]:.6f}/năm
   → Chất lượng tăng trưởng {'đang cải thiện' if A_trend[0] > 0 else 'cần cải thiện'}, 
     năng suất nội sinh đóng góp tích cực.

b) Yếu tố đóng góp lớn nhất: {factors[np.argmax([abs(v) for _,v in factors])][0]}
   D (số hóa) đóng góp {avg_D:.3f}%/năm, AI đóng góp {avg_AI:.3f}%/năm.

c) Mục tiêu D=30% vào 2030: {'KHẢ THI' if Y_2030 > 15000 else 'CẦN XEM XÉT'}
   GDP 2030 dự báo ≈ {Y_2030:,.0f} nghìn tỷ, tăng trưởng ≈ {((Y_2030/Y[-1])**(1/5)-1)*100:.2f}%/năm
""")

print("=" * 60)
print("HOÀN THÀNH BÀI 1")
print("=" * 60)
