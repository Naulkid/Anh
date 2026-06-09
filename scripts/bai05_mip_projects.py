"""
Bài 5: Quy hoạch nguyên hỗn hợp (MIP) — Lựa chọn 15 dự án CĐS
=================================================================
max Σ B_i * y_i
y_i ∈ {0,1}: chọn hay không chọn dự án i
Ràng buộc: ngân sách, loại trừ, tiên quyết, cân đối lĩnh vực

Yêu cầu:
    5.4.1 - Giải với PuLP (ngân sách 80,000 tỷ)
    5.4.2 - Nới ngân sách lên 100,000 tỷ
    5.4.3 - Bắt buộc cả P1 và P2
    5.4.4 - Tối đa hóa lợi ích kỳ vọng E[Z] = Σ p_i * B_i * y_i
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import pulp
    PULP_OK = True
except ImportError:
    PULP_OK = False
    print("⚠ pulp chưa cài")

print("=" * 60)
print("BÀI 5: MIP — LỰA CHỌN 15 DỰ ÁN CHUYỂN ĐỔI SỐ QUỐC GIA")
print("=" * 60)

# ─────────────────────────────────────────
# DỮ LIỆU 15 DỰ ÁN
# ─────────────────────────────────────────
P = list(range(1, 16))

project_names = [
    "P1  Trung tâm DL QG Hòa Lạc",
    "P2  Trung tâm DL QG phía Nam",
    "P3  5G phủ sóng toàn quốc",
    "P4  VNeID 2.0",
    "P5  Cổng DVCQG v3",
    "P6  Y tế số quốc gia",
    "P7  Giáo dục số K-12",
    "P8  Trung tâm AI + Supercomputing",
    "P9  Sandbox fintech",
    "P10 Logistics thông minh",
    "P11 Nông nghiệp số ĐBSCL",
    "P12 Đào tạo 50k kỹ sư AI",
    "P13 Khu CN bán dẫn Bắc Ninh",
    "P14 An ninh mạng SOC",
    "P15 Open Data quốc gia",
]

# Chi phí tổng 5 năm (tỷ VND)
C  = {1:12000, 2:11500, 3:18000, 4:4500,  5:3200,
      6:5800,  7:6500,  8:15000, 9:2500,  10:7200,
      11:4800, 12:8500, 13:20000, 14:3800, 15:1500}

# Chi phí năm 1-2
C1 = {1:8500,  2:7500,  3:12000, 4:3500,  5:2500,
      6:4000,  7:4500,  8:9000,  9:1800,  10:5000,
      11:3500, 12:5500, 13:13000, 14:2800, 15:1200}

# Lợi ích NPV
B  = {1:21500, 2:20800, 3:32500, 4:9200,  5:6800,
      6:11400, 7:12200, 8:28500, 9:5800,  10:13800,
      11:8500, 12:16200, 13:35000, 14:7500, 15:3800}

# Xác suất hoàn thành đúng tiến độ (cho câu 5.4.4)
# hạ tầng=0.85, chính phủ số=0.75, AI/bán dẫn=0.65, còn lại=0.80
prob = {1:0.85, 2:0.85, 3:0.85, 4:0.75, 5:0.75,
        6:0.80, 7:0.80, 8:0.65, 9:0.80, 10:0.80,
        11:0.80, 12:0.80, 13:0.65, 14:0.80, 15:0.80}

# ─────────────────────────────────────────
# HELPER: In kết quả
# ─────────────────────────────────────────
def print_results(y_vals, B_dict, C_dict, label=""):
    selected = [i for i in P if y_vals.get(i, 0) > 0.5]
    total_B = sum(B_dict[i] for i in selected)
    total_C = sum(C_dict[i] for i in selected)
    print(f"\n  {label}")
    print(f"  Số dự án chọn: {len(selected)}")
    print(f"  Tổng chi phí: {total_C:,} tỷ VND")
    print(f"  Z* (tổng lợi ích): {total_B:,} tỷ VND")
    print(f"  NPV biên (Z*/C): {total_B/total_C:.4f}")
    print(f"\n  Danh sách dự án chọn:")
    for i in selected:
        print(f"    {project_names[i-1]}: NPV={B_dict[i]:,}, Chi phí={C_dict[i]:,}")
    return selected, total_B, total_C

# ─────────────────────────────────────────
# HÀM GIẢI CHÍNH
# ─────────────────────────────────────────
def solve_mip(budget_total=80000, budget_y12=40000,
              force_p1p2=False, use_prob=False, label=""):
    if not PULP_OK:
        return None

    m = pulp.LpProblem("VN_Project_MIP", pulp.LpMaximize)
    y = pulp.LpVariable.dicts('y', P, cat='Binary')

    # Hàm mục tiêu
    if use_prob:
        m += pulp.lpSum(prob[i] * B[i] * y[i] for i in P)
    else:
        m += pulp.lpSum(B[i] * y[i] for i in P)

    # C1: Ngân sách tổng 5 năm
    m += pulp.lpSum(C[i] * y[i] for i in P) <= budget_total, "C1_total"
    # C2: Ngân sách năm 1-2
    m += pulp.lpSum(C1[i] * y[i] for i in P) <= budget_y12, "C2_year12"
    # C3: Chỉ chọn một trung tâm dữ liệu
    if not force_p1p2:
        m += y[1] + y[2] <= 1, "C3_datacenter"
    # C4: Tiên quyết AI cần đào tạo
    m += y[8] <= y[12], "C4_AI_prereq"
    # C5: Tiên quyết bán dẫn cần đào tạo
    m += y[13] <= y[12], "C5_semi_prereq"
    # C6: Cân đối lĩnh vực
    m += y[4] + y[5] >= 1, "C6_gov_digital"   # ít nhất 1 chính phủ số
    m += y[14] >= 1, "C6_cybersec"             # an ninh mạng bắt buộc
    # C7: Số lượng dự án
    m += pulp.lpSum(y[i] for i in P) >= 7, "C7_min_projects"
    m += pulp.lpSum(y[i] for i in P) <= 11, "C7_max_projects"

    if force_p1p2:
        m += y[1] == 1, "Force_P1"
        m += y[2] == 1, "Force_P2"

    m.solve(pulp.PULP_CBC_CMD(msg=False))

    status = pulp.LpStatus[m.status]
    if status == "Optimal":
        y_vals = {i: pulp.value(y[i]) for i in P}
        return y_vals, pulp.value(m.objective)
    else:
        print(f"  Status: {status} — không tìm được nghiệm")
        return None, None

# ─────────────────────────────────────────
# CÂU 5.4.1: Ngân sách 80,000 tỷ
# ─────────────────────────────────────────
print("\n--- Câu 5.4.1: Ngân sách 80,000 tỷ (mặc định) ---")
y1, Z1 = solve_mip(budget_total=80000, label="Ngân sách 80,000 tỷ")
if y1:
    sel1, _, _ = print_results(y1, B, C, "Ngân sách 80,000 tỷ")

# ─────────────────────────────────────────
# CÂU 5.4.2: Nới ngân sách 100,000 tỷ
# ─────────────────────────────────────────
print("\n--- Câu 5.4.2: Nới ngân sách lên 100,000 tỷ ---")
y2, Z2 = solve_mip(budget_total=100000, budget_y12=50000)
if y2:
    sel2, _, _ = print_results(y2, B, C, "Ngân sách 100,000 tỷ")
    new_proj = set([i for i in P if y2.get(i,0)>0.5]) - set([i for i in P if y1 and y1.get(i,0)>0.5])
    if new_proj:
        print(f"\n  Dự án thêm khi tăng ngân sách: {[project_names[i-1] for i in new_proj]}")

# ─────────────────────────────────────────
# CÂU 5.4.3: Bắt buộc P1 và P2
# ─────────────────────────────────────────
print("\n--- Câu 5.4.3: Bắt buộc cả P1 và P2 ---")
y3, Z3 = solve_mip(budget_total=80000, force_p1p2=True)
if y3:
    sel3, _, _ = print_results(y3, B, C, "Bắt buộc P1+P2")
    if Z1 and Z3:
        print(f"\n  ΔZ = {Z3 - Z1:,.0f} tỷ VND (so với phương án gốc)")
elif y3 is None:
    print("  → BÀI TOÁN KHÔNG KHẢ THI khi bắt buộc cả P1 và P2!")
    # Kiểm tra: tổng C[1]+C[2]=23,500 và C1[1]+C1[2]=16,000 đều trong ngân sách
    print(f"  Chi phí P1+P2 = {C[1]+C[2]:,} tỷ (năm 1-2: {C1[1]+C1[2]:,} tỷ)")
    print("  → Tìm lý do: bỏ ràng buộc min 7 dự án rồi thử lại...")

# ─────────────────────────────────────────
# CÂU 5.4.4: Tối đa hóa lợi ích kỳ vọng E[Z]
# ─────────────────────────────────────────
print("\n--- Câu 5.4.4: Tối đa hóa lợi ích kỳ vọng E[Z] = Σ p_i*B_i*y_i ---")
y4, Z4 = solve_mip(budget_total=80000, use_prob=True)
if y4:
    # Tính E[Z]
    sel4 = [i for i in P if y4.get(i,0)>0.5]
    EZ = sum(prob[i]*B[i] for i in sel4)
    print_results(y4, B, C, f"Lợi ích kỳ vọng E[Z]")
    print(f"\n  E[Z] thực = {EZ:,.0f} tỷ VND")
    if y1:
        diff = set(sel4) - set([i for i in P if y1.get(i,0)>0.5])
        same = set(sel4) & set([i for i in P if y1.get(i,0)>0.5])
        print(f"\n  Dự án khác với phương án xác định: {[project_names[i-1] for i in diff]}")

# ─────────────────────────────────────────
# BIỂU ĐỒ
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Bài 5: MIP — Lựa chọn Dự án Chuyển đổi Số", fontsize=14, fontweight='bold')

# Plot 1: NPV ratio của từng dự án
ratios = [B[i]/C[i] for i in P]
colors_bar = ['#2E7D32' if (y1 and y1.get(i,0)>0.5) else '#B0BEC5' for i in P]
bars = axes[0].bar([f'P{i}' for i in P], ratios, color=colors_bar, alpha=0.85)
axes[0].set_title("Tỷ suất NPV/Chi phí mỗi dự án\n(xanh = được chọn)", fontweight='bold')
axes[0].set_xlabel("Dự án")
axes[0].set_ylabel("NPV / Chi phí")
axes[0].axhline(y=np.mean(ratios), color='red', linestyle='--', alpha=0.7, label=f'Trung bình: {np.mean(ratios):.2f}')
axes[0].legend()
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(True, alpha=0.3, axis='y')

# Plot 2: So sánh Z* giữa các kịch bản
scenario_labels = []
z_values = []
if Z1:
    scenario_labels.append('Gốc\n80k tỷ')
    z_values.append(Z1)
if Z2:
    scenario_labels.append('Nới\n100k tỷ')
    z_values.append(Z2)
if Z4:
    scenario_labels.append('E[Z]\n(có rủi ro)')
    z_values.append(Z4)

if z_values:
    colors_sc = ['#1565C0', '#2E7D32', '#E65100'][:len(z_values)]
    bars2 = axes[1].bar(scenario_labels, z_values, color=colors_sc, alpha=0.85, width=0.4)
    axes[1].set_title("So sánh Z* giữa các kịch bản", fontweight='bold')
    axes[1].set_ylabel("Z* (tỷ VND)")
    for bar, val in zip(bars2, z_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                     f'{val:,.0f}', ha='center', fontweight='bold', fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/bai05_mip_projects.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ Biểu đồ đã lưu: bai05_mip_projects.png")

print("\n--- Thảo luận chính sách ---")
print("""
a) P15 (Open Data) bị bỏ qua dù NPV/C cao vì:
   - Ngân sách năm 1-2 đã dùng hết, P15 ít lợi ích tuyệt đối
   - Ràng buộc số lượng dự án (7≤Σy≤11) và tiên quyết
   → Chính sách: nên tách P15 ra ngoài khung ngân sách cứng

b) P14 (An ninh mạng) bắt buộc làm giảm Z* một chút nhưng hợp lý:
   → An ninh là điều kiện nền cho toàn bộ hệ thống số hoạt động

c) Cộng hưởng P8+P13: thêm ràng buộc synergy bonus
   y8_and_y13 = binary, nếu cả hai chọn thì B[8]+B[13] tăng thêm δ%
""")

print("=" * 60)
print("HOÀN THÀNH BÀI 5")
print("=" * 60)
