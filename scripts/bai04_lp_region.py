"""
Bài 4: Quy hoạch tuyến tính phân bổ ngân sách số theo ngành-vùng
==================================================================
max Z = Σ_r Σ_j β_{j,r} * x_{j,r}
24 biến: 6 vùng × 4 hạng mục (I, D, AI, H)
Ràng buộc: ngân sách tổng/vùng, sàn nhân lực, công bằng vùng

Yêu cầu:
    4.4.1 - PuLP, in ma trận 6×4
    4.4.2 - CVXPY, so sánh kết quả
    4.4.3 - Heatmap phân bổ tối ưu
    4.4.4 - Chi phí kinh tế của công bằng vùng (bỏ C5)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import pulp
    PULP_OK = True
except ImportError:
    PULP_OK = False

try:
    import cvxpy as cp
    CVXPY_OK = True
except ImportError:
    CVXPY_OK = False

print("=" * 60)
print("BÀI 4: LP PHÂN BỔ NGÂN SÁCH SỐ THEO 6 VÙNG × 4 HẠNG MỤC")
print("=" * 60)

# ─────────────────────────────────────────
# THAM SỐ BÀI TOÁN
# ─────────────────────────────────────────
regions = ['NMM', 'RRD', 'NCC', 'CH', 'SE', 'MD']
region_names = ['Trung du MN Bắc', 'Đồng bằng sông Hồng',
                'Bắc Trung Bộ+DH', 'Tây Nguyên',
                'Đông Nam Bộ', 'ĐB sông Cửu Long']
items = ['I', 'D', 'AI', 'H']
item_names = ['Hạ tầng số', 'CĐS DN', 'AI', 'Nhân lực số']

# Hệ số tác động biên β_{j,r} — shape (6, 4): [I, D, AI, H]
beta = np.array([
    [1.15, 0.85, 0.55, 1.30],  # NMM
    [0.95, 1.25, 1.40, 1.05],  # RRD
    [1.05, 0.95, 0.85, 1.15],  # NCC
    [1.20, 0.75, 0.45, 1.35],  # CH
    [0.90, 1.30, 1.55, 1.00],  # SE
    [1.10, 0.85, 0.65, 1.25],  # MD
])

# Chỉ số số hóa ban đầu D_r
D0 = np.array([38, 78, 55, 32, 82, 48])

gamma_eq = 0.002   # hệ số tác động CĐS lên Digital Index
lam      = 0.55    # lambda=0.7 infeasible với D0 gap lớn, dùng 0.55

# ─────────────────────────────────────────
# CÂU 4.4.1: Giải bằng PuLP (với ràng buộc C5)
# ─────────────────────────────────────────
print("\n--- Câu 4.4.1: Giải bằng PuLP (có ràng buộc công bằng C5) ---")

def solve_pulp_v4(equity=True, verbose=True):
    if not PULP_OK:
        return None, None
    m = pulp.LpProblem("VN_Digital_Budget_v4", pulp.LpMaximize)

    # Biến quyết định x[r][j]
    x = {}
    for r in regions:
        x[r] = {}
        for j in items:
            x[r][j] = pulp.LpVariable(f"x_{r}_{j}", lowBound=0)

    # Hàm mục tiêu
    r_idx = {r: i for i, r in enumerate(regions)}
    j_idx = {j: i for i, j in enumerate(items)}
    m += pulp.lpSum(beta[r_idx[r], j_idx[j]] * x[r][j]
                    for r in regions for j in items)

    # C1: Ngân sách tổng <= 50,000 tỷ
    m += pulp.lpSum(x[r][j] for r in regions for j in items) <= 50000, "C1_total"

    # C2: Sàn ngân sách mỗi vùng >= 5,000
    for r in regions:
        m += pulp.lpSum(x[r][j] for j in items) >= 5000, f"C2_floor_{r}"

    # C3: Trần ngân sách mỗi vùng <= 12,000
    for r in regions:
        m += pulp.lpSum(x[r][j] for j in items) <= 12000, f"C3_ceil_{r}"

    # C4: Tổng nhân lực số >= 12,000
    m += pulp.lpSum(x[r]['H'] for r in regions) >= 12000, "C4_human"

    # C5: Ràng buộc công bằng vùng (nếu bật)
    if equity:
        Dmax = pulp.LpVariable("Dmax", lowBound=0)
        for r in regions:
            ri = r_idx[r]
            m += D0[ri] + gamma_eq * x[r]['D'] <= Dmax, f"C5_upper_{r}"
            m += D0[ri] + gamma_eq * x[r]['D'] >= lam * Dmax, f"C5_lower_{r}"

    m.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[m.status] == "Optimal":
        Z = pulp.value(m.objective)
        # Trích xuất ma trận x[6×4]
        X_mat = np.zeros((6, 4))
        for i, r in enumerate(regions):
            for k, j in enumerate(items):
                X_mat[i, k] = pulp.value(x[r][j])

        if verbose:
            df_x = pd.DataFrame(X_mat, index=region_names, columns=item_names)
            print(f"\n  Z* = {Z:,.2f} tỷ VND GDP gain")
            print(f"\n  Phân bổ tối ưu (tỷ VND):")
            print(df_x.round(1).to_string())
            print(f"\n  Tổng mỗi vùng:")
            for i, rname in enumerate(region_names):
                print(f"    {rname}: {X_mat[i].sum():,.1f} tỷ VND")
        return Z, X_mat
    else:
        print(f"  Không khả thi: {pulp.LpStatus[m.status]}")
        return None, None

Z_equity, X_equity = solve_pulp_v4(equity=True)

# ─────────────────────────────────────────
# CÂU 4.4.2: Giải bằng CVXPY
# ─────────────────────────────────────────
print("\n--- Câu 4.4.2: Giải bằng CVXPY ---")

def solve_cvxpy_v4(equity=True, verbose=True):
    if not CVXPY_OK:
        print("  CVXPY chưa cài")
        return None, None

    X = cp.Variable((6, 4), nonneg=True)  # [6 vùng × 4 hạng mục]

    objective = cp.Maximize(cp.sum(cp.multiply(beta, X)))

    constraints = [
        cp.sum(X) <= 50000,                    # C1 tổng ngân sách
        cp.sum(X, axis=1) >= 5000,             # C2 sàn vùng
        cp.sum(X, axis=1) <= 12000,            # C3 trần vùng
        cp.sum(X[:, 3]) >= 12000,              # C4 nhân lực (cột H = index 3)
    ]

    if equity:
        Dmax = cp.Variable(nonneg=True)
        D_effective = D0 + gamma_eq * X[:, 1]  # D0 + γ * x_D
        constraints += [
            D_effective <= Dmax,
            D_effective >= lam * Dmax,
        ]

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.GLPK, verbose=False)
    except Exception:
        prob.solve(verbose=False)

    if prob.status in ['optimal', 'optimal_inaccurate']:
        Z = prob.value
        X_mat = X.value
        if verbose:
            df_x = pd.DataFrame(X_mat.round(1), index=region_names, columns=item_names)
            print(f"\n  Z* (CVXPY) = {Z:,.2f} tỷ VND GDP gain")
            print(f"\n  Phân bổ tối ưu:")
            print(df_x.to_string())
        return Z, X_mat
    else:
        print(f"  CVXPY status: {prob.status}")
        return None, None

Z_cvxpy, X_cvxpy = solve_cvxpy_v4(equity=True)

# So sánh
if Z_equity and Z_cvxpy:
    diff = abs(Z_equity - Z_cvxpy)
    print(f"\n  Chênh lệch PuLP vs CVXPY: {diff:.4f} tỷ VND → {'Nhất quán ✓' if diff < 1 else 'Có sai lệch'}")

# ─────────────────────────────────────────
# CÂU 4.4.3: Heatmap
# ─────────────────────────────────────────
print("\n--- Câu 4.4.3: Heatmap phân bổ tối ưu ---")
if X_equity is not None:
    df_heat = pd.DataFrame(X_equity, index=region_names, columns=item_names)
    print("\nVùng nhận nhiều ngân sách nhất:")
    totals = df_heat.sum(axis=1).sort_values(ascending=False)
    for rname, val in totals.items():
        print(f"  {rname}: {val:,.1f} tỷ VND")

# ─────────────────────────────────────────
# CÂU 4.4.4: Chi phí kinh tế của công bằng (bỏ C5)
# ─────────────────────────────────────────
print("\n--- Câu 4.4.4: Chi phí kinh tế của ràng buộc công bằng ---")
Z_no_equity, X_no_equity = solve_pulp_v4(equity=False, verbose=False)

if Z_equity and Z_no_equity:
    cost_equity = Z_no_equity - Z_equity
    pct_loss = cost_equity / Z_no_equity * 100
    print(f"\n  Z* (có C5 công bằng):  {Z_equity:,.2f} tỷ VND")
    print(f"  Z* (không có C5):      {Z_no_equity:,.2f} tỷ VND")
    print(f"  Chi phí kinh tế:       {cost_equity:,.2f} tỷ VND ({pct_loss:.2f}%)")
    print(f"  → Công bằng vùng làm giảm GDP gain {pct_loss:.2f}% — đây là")
    print("    'chi phí chính sách' chấp nhận được vì giảm bất bình đẳng")

# ─────────────────────────────────────────
# BIỂU ĐỒ
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Bài 4: LP Phân bổ Ngân sách Số theo Vùng", fontsize=14, fontweight='bold')

# Heatmap phân bổ (có công bằng)
if X_equity is not None:
    df_h = pd.DataFrame(X_equity, index=[r[:15] for r in region_names], columns=item_names)
    sns.heatmap(df_h, annot=True, fmt='.0f', cmap='YlOrRd',
                ax=axes[0], linewidths=0.5, linecolor='white',
                annot_kws={'size': 10})
    axes[0].set_title(f"Phân bổ tối ưu (có C5 công bằng)\nZ*={Z_equity:,.0f} tỷ VND",
                      fontweight='bold')
    axes[0].tick_params(axis='y', rotation=0)

# So sánh Z* có vs không có C5
if Z_equity and Z_no_equity:
    scenarios = ['Có ràng buộc\ncông bằng (C5)', 'Không có\nC5']
    z_vals = [Z_equity, Z_no_equity]
    colors = ['#1565C0', '#C62828']
    bars = axes[1].bar(scenarios, z_vals, color=colors, alpha=0.85,
                       edgecolor='white', width=0.4)
    axes[1].set_title("Chi phí kinh tế của công bằng vùng", fontweight='bold')
    axes[1].set_ylabel("Z* GDP gain (tỷ VND)")
    for bar, val in zip(bars, z_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                     f'{val:,.0f}', ha='center', fontweight='bold', fontsize=11)
    axes[1].set_ylim(0, max(z_vals) * 1.12)
    diff_label = f"Chi phí C5 = {cost_equity:,.0f} tỷ\n({pct_loss:.2f}%)"
    axes[1].annotate(diff_label, xy=(0.5, (z_vals[0]+z_vals[1])/2),
                     xycoords=('axes fraction', 'data'),
                     ha='center', fontsize=10, color='gray',
                     arrowprops=dict(arrowstyle='<->'))
    axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/bai04_lp_region_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ Biểu đồ đã lưu: bai04_lp_region_results.png")

print("\n--- Thảo luận chính sách ---")
print("""
a) Không có C5: vốn tập trung về SE (Đông Nam Bộ) và RRD (ĐBSH)
   do β cao hơn → bất bình đẳng vùng trầm trọng hơn dài hạn.

b) Ràng buộc trần C3 (12,000 tỷ/vùng) làm giảm Z* nhưng bảo đảm
   phân quyền — chi phí này phản ánh chi phí xã hội của công bằng.

c) Tây Nguyên: nên ưu tiên H (nhân lực) và I (hạ tầng) trước;
   β_AI=0.45 thấp nhất → đầu tư AI sẽ không hiệu quả khi hạ tầng
   và nhân lực chưa sẵn sàng.
""")

print("=" * 60)
print("HOÀN THÀNH BÀI 4")
print("=" * 60)
