"""
Bài 9: Tác động AI tới thị trường lao động Việt Nam
=====================================================
NetJob_i = a1_i*x_AI + a2_i*x_D + b1_i*x_H - c1_i*risk_i*x_AI
         = (a1_i - c1_i*risk_i)*x_AI + b1_i*x_H

max Σ_i NetJob_i
s.t. Σ_i (x_AI_i + x_H_i) <= 30,000 tỷ
     NetJob_i >= 0  ∀i
     DisplacedJob_i <= RetrainingCapacity_i
     x_AI_i >= AI_min_i  (sàn đầu tư AI thực tiễn)
     Σ x_AI_i >= 0.35 * budget  (AI chiếm ít nhất 35% ngân sách)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import cvxpy as cp
    CVXPY_OK = True
except ImportError:
    CVXPY_OK = False

print("=" * 60)
print("BÀI 9: TÁC ĐỘNG AI TỚI THỊ TRƯỜNG LAO ĐỘNG VIỆT NAM")
print("=" * 60)

# ─────────────────────────────────────────
# DỮ LIỆU 8 NGÀNH
# ─────────────────────────────────────────
sectors = ["Nông-Lâm-Thủy sản","CN chế biến chế tạo","Xây dựng",
           "Bán buôn-bán lẻ","Tài chính-Ngân hàng","Logistics-Vận tải",
           "CNTT-Truyền thông","Giáo dục-Đào tạo"]
N = 8

labor = np.array([13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15])  # triệu người
risk  = np.array([18, 42, 25, 38, 52, 35, 28, 22]) / 100

a1 = np.array([8.5,  32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])  # NewJob/tỷ AI
b1 = np.array([45.0, 28.0, 35.0, 32.0, 22.0, 30.0, 20.0, 55.0])  # UpgradeJob/tỷ H
c1 = np.array([5.2,  62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])  # Displaced/tỷ AI
d1 = np.array([50.0, 32.0, 42.0, 38.0, 26.0, 36.0, 24.0, 62.0])  # RetrainCap/tỷ H

# Net AI coefficient (việc/tỷ đầu tư AI)
net_ai = a1 - c1 * risk
print(f"\n{'Ngành':<25} {'Risk(%)':>8} {'Net AI/tỷ':>12}")
print("-" * 48)
for i in range(N):
    print(f"  {sectors[i]:<23} {risk[i]*100:>7.0f}% {net_ai[i]:>11.2f}")

BUDGET = 30000

# ─────────────────────────────────────────
# CÂU 9.4.1: Tối ưu CVXPY với ràng buộc thực tế
# ─────────────────────────────────────────
print("\n--- Câu 9.4.1: Tối ưu hóa (CVXPY) ---")

def solve_labor(budget=BUDGET, max_disp_pct=None, ai_min_pct=0.35, verbose=True):
    if not CVXPY_OK:
        return None

    x_AI = cp.Variable(N, nonneg=True)
    x_H  = cp.Variable(N, nonneg=True)

    NewJob     = cp.multiply(a1, x_AI)
    UpgradeJob = cp.multiply(b1, x_H)
    Displaced  = cp.multiply(c1 * risk, x_AI)
    RetrainCap = cp.multiply(d1, x_H)
    NetJob     = NewJob + UpgradeJob - Displaced

    constraints = [
        cp.sum(x_AI + x_H) <= budget,        # ngân sách tổng
        NetJob >= 0,                           # không mất việc ròng
        Displaced <= RetrainCap,               # đào tạo lại đủ
        cp.sum(x_AI) >= ai_min_pct * budget,  # AI chiếm ít nhất 35%
        x_AI >= 200,                           # sàn tối thiểu 200 tỷ/ngành
        x_H  >= 100,                           # sàn H tối thiểu 100 tỷ/ngành
    ]

    if max_disp_pct is not None:
        # DisplacedJob (việc) <= max_disp_pct * Labor (triệu) * 1e6
        for i in range(N):
            constraints.append(
                c1[i] * risk[i] * x_AI[i] <= max_disp_pct * labor[i] * 1e6
            )

    prob = cp.Problem(cp.Maximize(cp.sum(NetJob)), constraints)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        prob.solve(verbose=False)

    if prob.status in ['optimal', 'optimal_inaccurate']:
        xAI = x_AI.value
        xH  = x_H.value
        NJ  = a1*xAI + b1*xH - c1*risk*xAI

        if verbose:
            print(f"\n  Z* Tổng NetJob = {NJ.sum():,.0f} việc làm ròng")
            print(f"  Tổng x_AI = {xAI.sum():,.1f} tỷ  |  Tổng x_H = {xH.sum():,.1f} tỷ\n")
            print(f"  {'Ngành':<25} {'x_AI':>8} {'x_H':>8} {'NewJob':>9} "
                  f"{'Upgrade':>9} {'Displaced':>10} {'NetJob':>9}")
            print("-" * 83)
            for i in range(N):
                print(f"  {sectors[i]:<25} {xAI[i]:>8.0f} {xH[i]:>8.0f} "
                      f"{a1[i]*xAI[i]:>9.0f} {b1[i]*xH[i]:>9.0f} "
                      f"{c1[i]*risk[i]*xAI[i]:>10.0f} {NJ[i]:>9.0f}")
        return {'x_AI':xAI,'x_H':xH,'NetJob':NJ,'Z':NJ.sum(),'status':'optimal'}
    else:
        if verbose:
            print(f"  Status: {prob.status}")
        return {'status':prob.status,'Z':None}

r_base = solve_labor(budget=BUDGET, verbose=True)

# ─────────────────────────────────────────
# CÂU 9.4.2: Break-even ngành CN chế biến
# ─────────────────────────────────────────
print("\n--- Câu 9.4.2: Ngưỡng x_H ngành CN chế biến chế tạo ---")

i2 = 1  # CN chế biến
print(f"\n  Ngành: {sectors[i2]}")
print(f"  a1={a1[i2]}, c1={c1[i2]}, risk={risk[i2]:.2f}")
print(f"  Hệ số ròng AI: a1 - c1*risk = {net_ai[i2]:.3f} > 0")
print(f"  → Mỗi tỷ VND đầu tư AI tạo thêm {net_ai[i2]:.1f} việc làm ròng (dù không có x_H)")

# Phân tích: tối đa hóa NetJob₂ = (a1-c1*risk)*x_AI₂ + b1*x_H₂
# với x_AI₂ + x_H₂ <= B₂ (ngân sách riêng ngành 2)
B2_range = np.arange(1000, 10001, 1000)
print(f"\n  Phân bổ tối ưu cho ngành 2 (toàn bộ ngân sách dồn vào ngành 2):")
print(f"  {'Ngân sách':>12} {'x_AI':>10} {'x_H':>10} {'NetJob':>12}")
print("-" * 50)
for B2 in B2_range:
    # Net AI > Net H? So sánh net_ai[1] vs b1[1]
    if net_ai[i2] > b1[i2]:
        x_ai_opt = B2; x_h_opt = 0
    else:
        x_ai_opt = 200  # sàn tối thiểu
        x_h_opt = B2 - 200
    nj = net_ai[i2]*x_ai_opt + b1[i2]*x_h_opt
    print(f"  {B2:>12,} {x_ai_opt:>10,.0f} {x_h_opt:>10,.0f} {nj:>12,.0f}")

# Tìm x_H₂ tối thiểu để đảm bảo NetJob₂ >= 0 khi x_AI₂ cố định
print(f"\n  Vì net_ai[2] = {net_ai[i2]:.2f} > 0 nên NetJob₂ >= 0 với mọi x_H₂ >= 0")
print(f"  → Ngưỡng x_H tối thiểu = 0 tỷ VND")
print(f"  → Chiến lược: ưu tiên x_AI (tạo thêm {net_ai[i2]:.1f} việc/tỷ > b1={b1[i2]} việc/tỷ? "
      f"{'AI tốt hơn H' if net_ai[i2] > b1[i2] else 'H tốt hơn AI'})")

# ─────────────────────────────────────────
# CÂU 9.4.3: Luồng lao động nhóm dễ tổn thương
# ─────────────────────────────────────────
print("\n--- Câu 9.4.3: Luồng dịch chuyển lao động dễ tổn thương ---")

vuln_idx   = [0, 2, 3]  # Nông-Lâm, Xây dựng, Bán buôn-bán lẻ
vuln_names = [sectors[i] for i in vuln_idx]

if r_base and r_base.get('x_AI') is not None:
    print(f"\n  {'Ngành':<25} {'Displaced':>12} {'Upgrade':>12} {'NewJob':>12} {'NetJob':>10}")
    print("-" * 74)
    for i in vuln_idx:
        dj = c1[i]*risk[i]*r_base['x_AI'][i]
        uj = b1[i]*r_base['x_H'][i]
        nj = a1[i]*r_base['x_AI'][i]
        net = nj + uj - dj
        pct = dj/(labor[i]*1e6)*100 if labor[i] > 0 else 0
        print(f"  {sectors[i]:<25} {dj:>12,.0f} {uj:>12,.0f} {nj:>12,.0f} {net:>10,.0f}")
        print(f"  {'':25} {'('+f'{pct:.3f}% lao động)':>12}")

# ─────────────────────────────────────────
# CÂU 9.4.4: Ràng buộc ≤ 5% lao động
# ─────────────────────────────────────────
print("\n--- Câu 9.4.4: Ràng buộc DisplacedJob ≤ 5% lao động ---")

r_c5 = solve_labor(budget=BUDGET, max_disp_pct=0.05, verbose=False)
r_c10= solve_labor(budget=BUDGET, max_disp_pct=0.10, verbose=False)

print(f"\n  {'Kịch bản':<30} {'Z* (việc)':>15} {'Trạng thái':>12}")
print("-" * 60)
z_base = r_base['Z'] if r_base and r_base.get('Z') else 0
print(f"  {'Không ràng buộc (gốc)':<30} {z_base:>15,.0f} {'✓ OK':>12}")
if r_c5 and r_c5.get('Z'):
    print(f"  {'Ràng buộc ≤5% lao động':<30} {r_c5['Z']:>15,.0f} {'✓ OK':>12}")
    print(f"  {'  → Chênh lệch:':<30} {r_c5['Z']-z_base:>+15,.0f} {(r_c5['Z']-z_base)/z_base*100:>+11.2f}%")
else:
    print(f"  {'Ràng buộc ≤5%':<30} {'—':>15} {'✗ Infeasible':>12}")

if r_c10 and r_c10.get('Z'):
    print(f"  {'Ràng buộc ≤10%':<30} {r_c10['Z']:>15,.0f} {'✓ OK':>12}")

# Giải thích
print("\n  Kiểm tra: DisplacedJob tối đa mỗi ngành (tại nghiệm gốc):")
if r_base and r_base.get('x_AI') is not None:
    for i in range(N):
        dj = c1[i]*risk[i]*r_base['x_AI'][i]
        pct = dj/(labor[i]*1e6)*100
        flag = " ← VƯỢT 5%!" if pct > 5 else ""
        print(f"    {sectors[i]:<25}: {dj:>10,.0f} việc = {pct:.3f}%{flag}")

# ─────────────────────────────────────────
# BIỂU ĐỒ TỔNG HỢP
# ─────────────────────────────────────────
fig = plt.figure(figsize=(18, 13))
fig.suptitle("Bài 9: Tác động AI tới Thị trường Lao động — Việt Nam",
             fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)

# Plot 1: Phân bổ x_AI và x_H
ax1 = fig.add_subplot(gs[0, 0])
if r_base and r_base.get('x_AI') is not None:
    x_pos = np.arange(N)
    ax1.barh(x_pos + 0.2, r_base['x_AI'], 0.35, color='#1565C0', label='x_AI', alpha=0.85)
    ax1.barh(x_pos - 0.2, r_base['x_H'],  0.35, color='#E65100', label='x_H',  alpha=0.85)
    ax1.set_yticks(x_pos)
    ax1.set_yticklabels([s[:14] for s in sectors], fontsize=8)
    ax1.set_xlabel("Tỷ VND")
    ax1.set_title("Phân bổ đầu tư tối ưu (tỷ VND)", fontweight='bold')
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3, axis='x')

# Plot 2: Phân rã NetJob (stacked)
ax2 = fig.add_subplot(gs[0, 1])
if r_base and r_base.get('x_AI') is not None:
    x_pos = np.arange(N)
    nj_ai  = a1 * r_base['x_AI']
    nj_h   = b1 * r_base['x_H']
    nj_dis = -(c1 * risk * r_base['x_AI'])
    ax2.bar(x_pos, nj_ai,  color='#2E7D32', label='NewJob(AI)',    alpha=0.85)
    ax2.bar(x_pos, nj_h,   bottom=nj_ai, color='#1565C0', label='Upgrade(H)', alpha=0.85)
    ax2.bar(x_pos, nj_dis, color='#C62828', label='−Displaced',    alpha=0.85)
    # Vẽ NetJob ròng
    net_vals = nj_ai + nj_h + nj_dis
    ax2.plot(x_pos, net_vals, 'ko-', ms=6, lw=2, label='NetJob ròng', zorder=5)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([s[:7] for s in sectors], rotation=45, ha='right', fontsize=7)
    ax2.set_title("Phân rã NetJob từng ngành", fontweight='bold')
    ax2.set_ylabel("Số việc làm"); ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Risk vs Net AI coefficient (bubble)
ax3 = fig.add_subplot(gs[0, 2])
sc = ax3.scatter(risk*100, net_ai, s=labor*30, c=range(N), cmap='Set2',
                 alpha=0.85, edgecolors='gray', linewidth=0.7)
for i in range(N):
    ax3.annotate(sectors[i][:9], (risk[i]*100, net_ai[i]),
                 textcoords='offset points', xytext=(5, 3), fontsize=7)
ax3.axhline(0, color='red', linestyle='--', alpha=0.5, label='Break-even')
ax3.set_xlabel("Rủi ro tự động hóa (%)")
ax3.set_ylabel("Hệ số NetJob ròng/tỷ AI")
ax3.set_title("Risk vs Net AI Impact\n(bong bóng ~ quy mô lao động)", fontweight='bold', fontsize=9)
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

# Plot 4: Sankey-style dịch chuyển nhóm dễ tổn thương
ax4 = fig.add_subplot(gs[1, :2])
if r_base and r_base.get('x_AI') is not None:
    bar_w = 0.22
    x_pos = np.arange(len(vuln_idx))
    dj_v = np.array([c1[i]*risk[i]*r_base['x_AI'][i] for i in vuln_idx])
    uj_v = np.array([b1[i]*r_base['x_H'][i] for i in vuln_idx])
    nj_v = np.array([a1[i]*r_base['x_AI'][i] for i in vuln_idx])
    net_v = nj_v + uj_v - dj_v

    ax4.bar(x_pos - bar_w,  dj_v, bar_w, color='#C62828', label='Lao động dịch chuyển', alpha=0.85)
    ax4.bar(x_pos,          uj_v, bar_w, color='#1565C0', label='Nâng cấp kỹ năng', alpha=0.85)
    ax4.bar(x_pos + bar_w,  nj_v, bar_w, color='#2E7D32', label='Việc làm mới từ AI', alpha=0.85)

    for i, xi in enumerate(x_pos):
        y_top = max(dj_v[i], uj_v[i], nj_v[i])
        color_net = '#1B5E20' if net_v[i] >= 0 else '#B71C1C'
        ax4.annotate(f'NetJob\n{net_v[i]:+,.0f}',
                     xy=(xi, y_top), xytext=(0, 10),
                     textcoords='offset points', ha='center',
                     fontsize=9, fontweight='bold', color=color_net,
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(vuln_names, fontsize=10)
    ax4.axhline(0, color='black', lw=0.8)
    ax4.set_title("Luồng Dịch chuyển Lao động — Nhóm Dễ tổn thương", fontweight='bold')
    ax4.set_ylabel("Số việc làm"); ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: So sánh Z* với ràng buộc mất việc
ax5 = fig.add_subplot(gs[1, 2])
scenarios = ['Không ràng buộc', '≤ 10% lao động', '≤ 5% lao động']
z_vals = [
    r_base['Z'] if r_base and r_base.get('Z') else 0,
    r_c10['Z'] if r_c10 and r_c10.get('Z') else 0,
    r_c5['Z']  if r_c5  and r_c5.get('Z')  else 0,
]
colors_bar = ['#1565C0', '#2E7D32', '#E65100']
bars = ax5.bar(scenarios, z_vals, color=colors_bar, alpha=0.85, width=0.5)
for bar, val in zip(bars, z_vals):
    if val > 0:
        ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10000,
                 f'{val:,.0f}', ha='center', fontsize=9, fontweight='bold')
ax5.set_title("Z* theo ràng buộc mất việc", fontweight='bold')
ax5.set_ylabel("Tổng NetJob (việc làm ròng)")
ax5.tick_params(axis='x', rotation=15, labelsize=9)
ax5.grid(True, alpha=0.3, axis='y')

plt.savefig('/mnt/user-data/outputs/bai09_labor_market.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ Biểu đồ đã lưu: bai09_labor_market.png")

print("\n--- Thảo luận chính sách ---")
if r_base and r_base.get('x_AI') is not None:
    top_h = sectors[np.argmax(r_base['x_H'])]
    top_ai = sectors[np.argmax(r_base['x_AI'])]
    print(f"""
a) Ngành cần đào tạo lại nhiều nhất: {top_h}
   Ngành đầu tư AI nhiều nhất: {top_ai}

b) Tài chính-Ngân hàng: risk 52% nhưng net_ai=8.1 việc/tỷ (cao)
   → Đầu tư AI vẫn tạo việc ròng → chiến lược "reskill đồng thời AI"

c) Nông-Lâm: net_ai=7.6 nhưng 13.2M lao động → cần x_H cao để
   upgrade kỹ năng trước khi tăng tốc AI

d) Ràng buộc "Displaced ≤ RetrainCap" biểu diễn nguyên tắc:
   "Tốc độ TĐH ≤ năng lực đào tạo lại" — cốt lõi của chính sách
   lao động công bằng trong kỷ nguyên AI
""")

print("=" * 60)
print("HOÀN THÀNH BÀI 9")
print("=" * 60)
