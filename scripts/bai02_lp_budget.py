"""
Bài 2: Phân bổ ngân sách đơn giản theo 4 hạng mục đầu tư số
===============================================================
max Z = 0.85*x1 + 1.20*x2 + 0.95*x3 + 1.35*x4
Ràng buộc: ngân sách tổng, sàn từng hạng mục, tỷ trọng công nghệ chiến lược

Yêu cầu:
    2.4.1 - Giải bằng scipy.optimize.linprog
    2.4.2 - Giải bằng PuLP, in shadow price
    2.4.3 - Phân tích độ nhạy: tăng ngân sách 100→120→140 nghìn tỷ
    2.4.4 - Thêm ràng buộc x3 >= 30, kiểm tra khả thi
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    from scipy.optimize import linprog
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    print("⚠ scipy chưa cài, bỏ qua câu 2.4.1")

try:
    import pulp
    PULP_OK = True
except ImportError:
    PULP_OK = False
    print("⚠ pulp chưa cài, bỏ qua câu 2.4.2")

print("=" * 60)
print("BÀI 2: QUY HOẠCH TUYẾN TÍNH PHÂN BỔ NGÂN SÁCH SỐ")
print("=" * 60)
print("Biến: x1=Hạ tầng số, x2=AI&Dữ liệu, x3=Nhân lực số, x4=R&D")
print("Đơn vị: nghìn tỷ VND")
print("Hệ số tác động: [0.85, 1.20, 0.95, 1.35]")

# ─────────────────────────────────────────
# CÂU 2.4.1: scipy.optimize.linprog
# ─────────────────────────────────────────
print("\n--- Câu 2.4.1: Giải bằng scipy.optimize.linprog ---")

if SCIPY_OK:
    # linprog tối thiểu hóa → dùng -c để tối đa hóa Z
    c = [-0.85, -1.20, -0.95, -1.35]

    # Ràng buộc bất đẳng thức: A_ub @ x <= b_ub
    A_ub = [
        [ 1,     1,     1,     1   ],  # x1+x2+x3+x4 <= 100  (tổng ngân sách)
        [-1,     0,     0,     0   ],  # -x1 <= -25          (x1 >= 25)
        [ 0,    -1,     0,     0   ],  # -x2 <= -15          (x2 >= 15)
        [ 0,     0,    -1,     0   ],  # -x3 <= -20          (x3 >= 20)
        [ 0,     0,     0,    -1   ],  # -x4 <= -10          (x4 >= 10)
        # x2+x4 >= 0.35*(x1+x2+x3+x4)
        # ↔ 0.35*x1 - 0.65*x2 + 0.35*x3 - 0.65*x4 <= 0
        [ 0.35, -0.65,  0.35, -0.65],
    ]
    b_ub = [100, -25, -15, -20, -10, 0]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(0, None)] * 4,
                  method='highs')

    if res.success:
        x_opt = res.x
        Z_opt = -res.fun
        print(f"\nKết quả tối ưu (scipy):")
        labels = ["x1 - Hạ tầng số", "x2 - AI & Dữ liệu", "x3 - Nhân lực số", "x4 - R&D"]
        for lbl, xi in zip(labels, x_opt):
            print(f"  {lbl}: {xi:.2f} nghìn tỷ VND")
        print(f"  Z* (GDP tăng thêm) = {Z_opt:.2f} nghìn tỷ VND")
    else:
        print(f"  Lỗi: {res.message}")

# ─────────────────────────────────────────
# CÂU 2.4.2: PuLP + Shadow price
# ─────────────────────────────────────────
print("\n--- Câu 2.4.2: Giải bằng PuLP + Shadow Price ---")

def solve_lp_pulp(budget_total=100, x3_min=20, verbose=True):
    """Giải bài toán LP với PuLP, trả về kết quả và dual values"""
    if not PULP_OK:
        return None

    m = pulp.LpProblem("Budget_Allocation", pulp.LpMaximize)

    x1 = pulp.LpVariable("x1_infra",  lowBound=25)    # hạ tầng số >= 25
    x2 = pulp.LpVariable("x2_AI",     lowBound=15)    # AI >= 15
    x3 = pulp.LpVariable("x3_human",  lowBound=x3_min)  # nhân lực >= x3_min
    x4 = pulp.LpVariable("x4_RD",     lowBound=10)    # R&D >= 10

    # Hàm mục tiêu
    m += 0.85*x1 + 1.20*x2 + 0.95*x3 + 1.35*x4, "Z_GDP_gain"

    # Ràng buộc
    m += (x1 + x2 + x3 + x4 <= budget_total, "C1_total_budget")
    m += (x2 + x4 >= 0.35*(x1+x2+x3+x4),    "C2_tech_ratio")

    m.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[m.status] == "Optimal":
        result = {
            'status': 'Optimal',
            'x': [pulp.value(x1), pulp.value(x2), pulp.value(x3), pulp.value(x4)],
            'Z': pulp.value(m.objective),
            'budget': budget_total,
        }
        if verbose:
            print(f"\n  Ngân sách tổng: {budget_total} nghìn tỷ VND")
            labels = ["x1 - Hạ tầng số", "x2 - AI & Dữ liệu", "x3 - Nhân lực số", "x4 - R&D"]
            for lbl, xi in zip(labels, result['x']):
                print(f"    {lbl}: {xi:.2f}")
            print(f"  Z* = {result['Z']:.4f} nghìn tỷ VND GDP gain")

            # Shadow prices (dual values) — đọc từ constraints
            print("\n  Shadow Prices (Giá đối ngẫu):")
            for name, constr in m.constraints.items():
                print(f"    [{name}]: {constr.pi:.4f}")
            print("  → Shadow price C1_total_budget: mỗi tăng 1 nghìn tỷ ngân sách,")
            print(f"    GDP kỳ vọng tăng thêm {abs(m.constraints['C1_total_budget'].pi):.4f} nghìn tỷ VND")
        return result
    else:
        if verbose:
            print(f"  Không tìm được nghiệm tối ưu: {pulp.LpStatus[m.status]}")
        return {'status': pulp.LpStatus[m.status], 'Z': None}

result_base = solve_lp_pulp(budget_total=100, x3_min=20)

# ─────────────────────────────────────────
# CÂU 2.4.3: Phân tích độ nhạy theo ngân sách
# ─────────────────────────────────────────
print("\n--- Câu 2.4.3: Phân tích độ nhạy ngân sách 100→140 nghìn tỷ ---")

budgets  = list(range(100, 145, 5))  # 100, 105, ..., 140
Z_values = []

for B in budgets:
    r = solve_lp_pulp(budget_total=B, verbose=False)
    Z_values.append(r['Z'] if r and r['Z'] else 0)

print(f"\n{'Ngân sách':>12} {'Z* (GDP gain)':>16}")
print("-" * 30)
for B, Z in zip(budgets, Z_values):
    print(f"{B:>10} nghìn tỷ  {Z:>14.2f}")

# ─────────────────────────────────────────
# CÂU 2.4.4: Thêm ràng buộc x3 >= 30
# ─────────────────────────────────────────
print("\n--- Câu 2.4.4: Ràng buộc x3 >= 30 (nhân lực số) ---")
result_x3 = solve_lp_pulp(budget_total=100, x3_min=30)

if result_x3 and result_x3['Z']:
    delta_Z = result_x3['Z'] - result_base['Z']
    print(f"\n  ΔZ = {delta_Z:.4f} nghìn tỷ VND")
    print(f"  → Bài toán VẪN KHẢ THI, Z* {'giảm' if delta_Z < 0 else 'tăng'} "
          f"{abs(delta_Z):.2f} nghìn tỷ")
    print("  → Chi phí cơ hội của chính sách ưu tiên nhân lực số!")
else:
    print("  → Bài toán KHÔNG KHẢ THI với x3 >= 30")

# ─────────────────────────────────────────
# BIỂU ĐỒ
# ─────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.suptitle("Bài 2: Phân bổ Ngân sách Số tối ưu — LP", fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# Plot 1: Phân bổ tối ưu (bar chart)
ax1 = fig.add_subplot(gs[0, 0])
labels   = ['Hạ tầng\nsố (x1)', 'AI &\nDữ liệu (x2)', 'Nhân lực\nsố (x3)', 'R&D\n(x4)']
colors   = ['#1565C0', '#6A1B9A', '#2E7D32', '#E65100']
x_vals   = result_base['x'] if result_base else [25, 15, 20, 40]
bars     = ax1.bar(labels, x_vals, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
coeff    = [0.85, 1.20, 0.95, 1.35]
for bar, val, c in zip(bars, x_vals, coeff):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}\n(β={c})', ha='center', va='bottom', fontsize=9)
ax1.set_title("Phân bổ tối ưu (ngân sách = 100 nghìn tỷ)", fontweight='bold')
ax1.set_ylabel("Nghìn tỷ VND")
ax1.set_ylim(0, max(x_vals) * 1.3)
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Đường cong Z*(B)
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(budgets, Z_values, 'o-', color='#C62828', linewidth=2.5, markersize=7)
ax2.fill_between(budgets, Z_values, alpha=0.1, color='#C62828')
ax2.axvline(100, color='gray', linestyle='--', alpha=0.6, label='Ngân sách gốc 100')
ax2.set_title("Độ nhạy: Z* theo ngân sách tổng", fontweight='bold')
ax2.set_xlabel("Ngân sách tổng (nghìn tỷ VND)")
ax2.set_ylabel("Z* — GDP gain (nghìn tỷ VND)")
ax2.legend()
ax2.grid(True, alpha=0.3)
# Ghi nhãn điểm
for B, Z in zip(budgets[::2], Z_values[::2]):
    ax2.annotate(f'{Z:.1f}', (B, Z), textcoords="offset points",
                 xytext=(0, 8), ha='center', fontsize=8)

# Plot 3: So sánh x3>=20 vs x3>=30
ax3 = fig.add_subplot(gs[1, 0])
x3_scenarios = ['x3 ≥ 20\n(mặc định)', 'x3 ≥ 30\n(ưu tiên NL số)']
z_compare = [result_base['Z'] if result_base else 0,
             result_x3['Z'] if result_x3 and result_x3['Z'] else 0]
bar_colors = ['#1565C0', '#E65100']
bars3 = ax3.bar(x3_scenarios, z_compare, color=bar_colors, alpha=0.85,
                edgecolor='white', width=0.4)
for bar, val in zip(bars3, z_compare):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
ax3.set_title("So sánh Z* theo ràng buộc nhân lực", fontweight='bold')
ax3.set_ylabel("Z* — GDP gain (nghìn tỷ VND)")
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Hệ số tác động (β) và phân bổ
ax4 = fig.add_subplot(gs[1, 1])
hm_data = np.array([[0.85, 1.20, 0.95, 1.35],
                     x_vals if result_base else [25, 15, 20, 40]])
im = ax4.imshow(hm_data, cmap='YlOrRd', aspect='auto')
ax4.set_xticks(range(4))
ax4.set_xticklabels(['Hạ tầng\n(x1)', 'AI\n(x2)', 'Nhân lực\n(x3)', 'R&D\n(x4)'], fontsize=9)
ax4.set_yticks(range(2))
ax4.set_yticklabels(['Hệ số β\n(tác động)', 'Phân bổ\n(nghìn tỷ)'])
for i in range(2):
    for j in range(4):
        ax4.text(j, i, f'{hm_data[i, j]:.2f}', ha='center', va='center',
                 fontsize=11, fontweight='bold', color='black')
ax4.set_title("Ma trận hệ số & phân bổ tối ưu", fontweight='bold')
plt.colorbar(im, ax=ax4, shrink=0.8)

plt.savefig('/mnt/user-data/outputs/bai02_lp_budget_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ Biểu đồ đã lưu: bai02_lp_budget_results.png")

print("\n--- Thảo luận chính sách ---")
print("""
a) Shadow price ngân sách tổng = hệ số β cao nhất (x4: R&D = 1.35)
   → Mỗi 1 nghìn tỷ ngân sách tăng thêm → GDP gain ≈ 1.35 nghìn tỷ
   → Đây là cận trên hợp lý của chi phí cơ hội vốn công

b) R&D hệ số cao (1.35) nhưng ràng buộc tối thiểu thấp (10 nghìn tỷ)
   → Vì R&D có độ trễ dài hạn, khó giải ngân nhanh; thực tế VN
     năng lực hấp thụ R&D còn hạn chế nên đặt sàn thấp hơn

c) Tỷ lệ 35% AI+R&D: mô hình đạt được về mặt toán học
   → Thực tiễn VN 2025: R&D chỉ ~0.5% GDP, cần lộ trình dài
""")

print("=" * 60)
print("HOÀN THÀNH BÀI 2")
print("=" * 60)
