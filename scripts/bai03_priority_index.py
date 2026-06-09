"""
Bài 3: Tính chỉ số ưu tiên ngành Priority_i cho 10 ngành Việt Nam
===================================================================
Priority_i = a1*Growth + a2*Productivity + a3*Spillover + a4*Export
           + a5*Employment + a6*AIReadiness - a7*Risk
Chuẩn hóa min-max trước khi tính.

Yêu cầu:
    3.4.1 - Chuẩn hóa min-max, in ma trận đã chuẩn hóa
    3.4.2 - Tính Priority, xếp hạng với bộ trọng số mặc định
    3.4.3 - Phân tích độ nhạy theo a6 (AI Readiness)
    3.4.4 - So sánh "Định hướng tăng trưởng" vs "Định hướng bao trùm"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

print("=" * 60)
print("BÀI 3: CHỈ SỐ ƯU TIÊN NGÀNH — 10 NGÀNH VIỆT NAM 2024")
print("=" * 60)

# ─────────────────────────────────────────
# DỮ LIỆU 10 NGÀNH (vietnam_sectors_2024)
# ─────────────────────────────────────────
sectors = [
    "Nông-Lâm-Thủy sản",
    "CN chế biến chế tạo",
    "Xây dựng",
    "Khai khoáng",
    "Bán buôn-bán lẻ",
    "Tài chính-Ngân hàng",
    "Logistics-Vận tải",
    "CNTT-Truyền thông",
    "Giáo dục-Đào tạo",
    "Y tế",
]

# [Growth(%), Productivity(tr.VND/LĐ), Spillover(0-1), Export(tỷUSD),
#  Employment(tr.LĐ), AIReadiness(0-100), Risk(%)]
raw_data = np.array([
    [ 3.27,   103.4, 0.35,   40.5, 13.20, 15, 18],
    [ 9.64,   241.2, 0.78,  290.9, 11.50, 55, 42],
    [ 7.45,   168.8, 0.42,    2.5,  4.80, 20, 25],
    [-1.20,  1290.5, 0.30,    8.2,  0.30, 30, 55],
    [ 7.10,   145.3, 0.55,    5.5,  7.80, 48, 38],
    [ 7.36,  1072.4, 0.85,    1.2,  0.55, 72, 52],
    [ 9.93,   321.4, 0.72,    3.1,  1.95, 42, 35],
    [ 7.85,   713.8, 0.92,  178.0,  0.62, 88, 28],
    [ 6.42,   205.7, 0.65,    0.0,  2.15, 38, 22],
    [ 6.85,   437.1, 0.60,    0.0,  0.75, 45, 18],
])

col_names = ['Growth(%)', 'Productivity', 'Spillover', 'Export',
             'Employment', 'AIReadiness', 'Risk(%)']

df = pd.DataFrame(raw_data, index=sectors, columns=col_names)

# ─────────────────────────────────────────
# CÂU 3.4.1: Chuẩn hóa min-max
# ─────────────────────────────────────────
print("\n--- Câu 3.4.1: Chuẩn hóa Min-Max ---")

def norm_good(col):
    """Chuẩn hóa tiêu chí tốt (cao hơn = tốt hơn)"""
    return (col - col.min()) / (col.max() - col.min())

def norm_bad(col):
    """Chuẩn hóa tiêu chí xấu (thấp hơn = tốt hơn) — đảo dấu"""
    return (col.max() - col) / (col.max() - col.min())

df_norm = df.copy()
good_cols = ['Growth(%)', 'Productivity', 'Spillover', 'Export', 'Employment', 'AIReadiness']
for c in good_cols:
    df_norm[c] = norm_good(df[c])
df_norm['Risk(%)'] = norm_bad(df['Risk(%)'])

print("\nMa trận đã chuẩn hóa [0, 1]:")
print(df_norm.round(4).to_string())

# ─────────────────────────────────────────
# CÂU 3.4.2: Tính Priority với trọng số mặc định
# ─────────────────────────────────────────
print("\n--- Câu 3.4.2: Tính Priority (trọng số mặc định) ---")

# Trọng số: a1=Growth, a2=Productivity, a3=Spillover, a4=Export,
#           a5=Employment, a6=AIReadiness, a7=Risk
w_default = np.array([0.15, 0.15, 0.20, 0.15, 0.10, 0.20, 0.15])
print(f"Trọng số: {dict(zip(col_names, w_default))}")

# Priority = sum(w_good * X_norm) - w_risk * Risk_norm
X_good = df_norm[good_cols].values   # shape (10, 6)
X_risk = df_norm['Risk(%)'].values   # shape (10,)

priority = X_good @ w_default[:6] - w_default[6] * X_risk

df['Priority'] = priority
df_sorted = df.sort_values('Priority', ascending=False)

print("\nXếp hạng 10 ngành theo Priority (giảm dần):")
print(f"\n{'Hạng':>5} {'Ngành':<25} {'Priority':>10}")
print("-" * 43)
for rank, (sector, row) in enumerate(df_sorted.iterrows(), 1):
    print(f"{rank:>5}  {sector:<25} {row['Priority']:>9.4f}")

# ─────────────────────────────────────────
# CÂU 3.4.3: Phân tích độ nhạy theo a6 (AI Readiness)
# ─────────────────────────────────────────
print("\n--- Câu 3.4.3: Phân tích độ nhạy theo a6 (AIReadiness) ---")

a6_range = np.arange(0.05, 0.45, 0.05)
rankings_matrix = []   # lưu ranking của từng ngành theo a6

for a6 in a6_range:
    # Tái chuẩn hóa tổng trọng số = 1 (điều chỉnh a3 - Spillover)
    remaining = 1.0 - a6 - 0.15  # bỏ a7=0.15 cố định
    # Phân bổ phần còn lại theo tỷ lệ: a1:a2:a3:a4:a5 = 0.15:0.15:0.20:0.15:0.10
    base = np.array([0.15, 0.15, 0.20, 0.15, 0.10])
    scale = remaining / base.sum()
    w_new = np.concatenate([base * scale, [a6]])
    # w_new[0..4] = 5 yếu tố tốt (không phải AI), w_new[5] = AI
    # Ghép lại đúng thứ tự: Growth, Productivity, Spillover, Export, Employment, AIReadiness
    w_good_new = np.array([w_new[0], w_new[1], w_new[2], w_new[3], w_new[4], a6])
    w_risk_new = 0.15
    p = X_good @ w_good_new - w_risk_new * X_risk
    rankings_matrix.append(p)

rankings_df = pd.DataFrame(rankings_matrix, index=a6_range.round(2), columns=sectors)

print("\nTop-3 ngành theo a6:")
print(f"\n{'a6':>6}", end="")
for i in range(3):
    print(f"  {'Top'+str(i+1):>23}", end="")
print()
print("-" * 80)
for a6, row in rankings_df.iterrows():
    top3 = row.nlargest(3).index.tolist()
    print(f"{a6:>6.2f}  {top3[0]:<23}  {top3[1]:<23}  {top3[2]:<23}")

# ─────────────────────────────────────────
# CÂU 3.4.4: So sánh 2 bộ trọng số
# ─────────────────────────────────────────
print("\n--- Câu 3.4.4: So sánh 2 bộ trọng số ---")

# Bộ 1: Định hướng tăng trưởng (Growth, Productivity, Export cao)
w_growth = np.array([0.25, 0.25, 0.10, 0.25, 0.05, 0.10, 0.10])
# Bộ 2: Định hướng bao trùm (Employment, Spillover, Risk thấp)
w_inclusive = np.array([0.10, 0.10, 0.25, 0.05, 0.25, 0.10, 0.15])

p_growth    = X_good @ w_growth[:6]    - w_growth[6]    * X_risk
p_inclusive = X_good @ w_inclusive[:6] - w_inclusive[6] * X_risk

df['Priority_Growth']    = p_growth
df['Priority_Inclusive'] = p_inclusive

top3_growth    = df.nlargest(3, 'Priority_Growth').index.tolist()
top3_inclusive = df.nlargest(3, 'Priority_Inclusive').index.tolist()

print("\nBộ trọng số 'Định hướng Tăng trưởng':")
print(f"  Growth=0.25, Productivity=0.25, Export=0.25, AI=0.10")
print(f"  Top-3: {top3_growth}")

print("\nBộ trọng số 'Định hướng Bao trùm':")
print(f"  Employment=0.25, Spillover=0.25, Risk=0.15")
print(f"  Top-3: {top3_inclusive}")

different = set(top3_growth) != set(top3_inclusive)
print(f"\n→ Top-3 {'KHÁC NHAU' if different else 'GIỐNG NHAU'} giữa 2 bộ trọng số")

# ─────────────────────────────────────────
# BIỂU ĐỒ
# ─────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.suptitle("Bài 3: Chỉ số Ưu tiên Ngành — 10 Ngành Việt Nam 2024",
             fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.35)

# Plot 1: Ranking Priority mặc định
ax1 = fig.add_subplot(gs[0, 0])
sorted_idx = np.argsort(priority)
colors_rank = ['#1565C0' if i >= 7 else '#78909C' for i in range(10)]
colors_rank_sorted = [colors_rank[i] for i in sorted_idx]
bars = ax1.barh([sectors[i] for i in sorted_idx],
                [priority[i] for i in sorted_idx],
                color=colors_rank_sorted, alpha=0.85, edgecolor='white')
ax1.set_title("Priority Index (trọng số mặc định)", fontweight='bold')
ax1.set_xlabel("Priority Score")
for bar, val in zip(bars, [priority[i] for i in sorted_idx]):
    ax1.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=8)
ax1.grid(True, alpha=0.3, axis='x')

# Plot 2: Heatmap ma trận chuẩn hóa
ax2 = fig.add_subplot(gs[0, 1])
sns.heatmap(df_norm.values, annot=True, fmt='.2f', cmap='RdYlGn',
            xticklabels=col_names, yticklabels=[s[:15] for s in sectors],
            ax=ax2, linewidths=0.5, linecolor='white', annot_kws={'size': 7})
ax2.set_title("Ma trận chuẩn hóa Min-Max", fontweight='bold')
ax2.tick_params(axis='x', rotation=30, labelsize=8)
ax2.tick_params(axis='y', rotation=0, labelsize=7)

# Plot 3: Độ nhạy a6 — priority scores
ax3 = fig.add_subplot(gs[1, 0])
highlight = ["CNTT-Truyền thông", "CN chế biến chế tạo", "Tài chính-Ngân hàng",
             "Logistics-Vận tải", "Nông-Lâm-Thủy sản"]
cmap_lines = plt.cm.tab10
for idx, sector in enumerate(sectors):
    alpha_val = 0.9 if sector in highlight else 0.2
    lw = 2.0 if sector in highlight else 0.8
    label = sector if sector in highlight else None
    ax3.plot(a6_range, rankings_df[sector], alpha=alpha_val,
             linewidth=lw, label=label, color=cmap_lines(idx/10))
ax3.set_title("Độ nhạy Priority theo a6 (AIReadiness)", fontweight='bold')
ax3.set_xlabel("Trọng số a6 (AIReadiness)")
ax3.set_ylabel("Priority Score")
ax3.legend(fontsize=7, loc='upper left')
ax3.grid(True, alpha=0.3)

# Plot 4: So sánh 2 bộ trọng số
ax4 = fig.add_subplot(gs[1, 1])
x_pos = np.arange(10)
width = 0.35
bars1 = ax4.bar(x_pos - width/2, p_growth, width, label='Tăng trưởng',
                color='#1565C0', alpha=0.8, edgecolor='white')
bars2 = ax4.bar(x_pos + width/2, p_inclusive, width, label='Bao trùm',
                color='#E65100', alpha=0.8, edgecolor='white')
ax4.set_xticks(x_pos)
ax4.set_xticklabels([s[:10] for s in sectors], rotation=45, ha='right', fontsize=7)
ax4.set_title("So sánh 2 bộ trọng số chính sách", fontweight='bold')
ax4.set_ylabel("Priority Score")
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.savefig('/mnt/user-data/outputs/bai03_priority_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ Biểu đồ đã lưu: bai03_priority_results.png")

print("\n--- Thảo luận chính sách ---")
top3_default = df.nlargest(3, 'Priority').index.tolist()
print(f"""
a) Ba ngành ưu tiên đẩy mạnh CĐS & AI: {top3_default}
   → Phù hợp Nghị quyết 57-NQ/TW: ưu tiên CNTT, chế biến chế tạo, 
     tài chính số.

b) Khai khoáng: năng suất cao (1290 tr.VND/LĐ) nhưng:
   - Tăng trưởng âm (-1.20%), rủi ro tự động hóa cao (55%)
   - AI Readiness thấp (30), xuất khẩu nội địa chủ yếu
   → Không ưu tiên CĐS: khó hấp thụ công nghệ mới

c) Trọng số nên do Hội đồng chính sách đa bên quyết định
   (kỹ thuật + xã hội + governance) để bảo đảm tính chính danh
""")

print("=" * 60)
print("HOÀN THÀNH BÀI 3")
print("=" * 60)
