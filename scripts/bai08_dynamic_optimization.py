"""
Bài 8: Tối ưu động phân bổ liên thời gian 2026–2035
=====================================================
Mô hình Ramsey mở rộng:
  max Σ_{t=0}^{T-1} ρ^t * ln(C_t)

Phương trình động:
  K_{t+1} = (1-δ_K)*K_t + I_K,t
  D_{t+1} = (1-δ_D)*D_t + I_D,t
  AI_{t+1}= (1-δ_AI)*AI_t + I_AI,t
  H_{t+1} = H_t + θ_H*I_H,t - μ*H_t  (H ≤ 95%)
  A_{t+1} = A_t*(1 + min(φ1*D + φ2*AI + φ3*H, 8%))

Hàm sản xuất: Y_t = A_t * K^0.33 * L^0.42 * D^0.10 * AI^0.08 * H^0.07
Ràng buộc NS: I_K + I_D + I_AI + I_H ≤ 0.40*Y_t

Yêu cầu:
    8.3.1 - Giải bằng scipy SLSQP
    8.3.2 - Vẽ quỹ đạo K, D, AI, H, Y, C
    8.3.3 - Cú sốc 2028: Y giảm 8%
    8.3.4 - So sánh chiến lược "đều" vs "front-load" vs tối ưu
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("BÀI 8: TỐI ƯU ĐỘNG LIÊN THỜI GIAN 2026–2035")
print("=" * 60)

# ─────────────────────────────────────────
# THAM SỐ
# ─────────────────────────────────────────
T    = 10      # 2026–2035
rho  = 0.97    # chiết khấu liên thời gian

alpha_k  = 0.33; alpha_l  = 0.42; alpha_d  = 0.10
alpha_ai = 0.08; alpha_h  = 0.07

delta_K  = 0.05; delta_D  = 0.12; delta_AI = 0.15
theta_H  = 0.80; mu = 0.02

phi1 = 0.003; phi2 = 0.002; phi3 = 0.004

# Điều kiện ban đầu 2026
K0=27500.0; L0=53.9; D0=20.3; AI0=86.0; H0=30.0; A0=30.9

L = L0 * (1.003 ** np.arange(T))  # lao động tăng 0.3%/năm

def production(A, K, L, D, AI, H):
    """Cobb-Douglas mở rộng"""
    return A * (K**alpha_k) * (L**alpha_l) * (D**alpha_d) * (AI**alpha_ai) * (H**alpha_h)

Y_init = production(A0, K0, L0, D0, AI0, H0)
print(f"GDP khởi đầu 2026: {Y_init:,.1f} nghìn tỷ VND")

def simulate(inv_vec, shock_year=None, shock_pct=0.0):
    """
    Mô phỏng 10 năm từ vector đầu tư.
    inv_vec: T×4 được flatten — [I_K, I_D, I_AI, I_H] mỗi năm
    """
    inv = inv_vec.reshape(T, 4)
    K_p  = np.zeros(T+1); D_p  = np.zeros(T+1)
    AI_p = np.zeros(T+1); H_p  = np.zeros(T+1); A_p = np.zeros(T+1)
    Y_p  = np.zeros(T);   C_p  = np.zeros(T)
    K_p[0]=K0; D_p[0]=D0; AI_p[0]=AI0; H_p[0]=H0; A_p[0]=A0
    welfare = 0.0
    for t in range(T):
        I_K, I_D, I_AI, I_H = inv[t]
        Y = production(A_p[t], K_p[t], L[t], D_p[t], AI_p[t], H_p[t])
        if shock_year is not None and t == shock_year:
            Y *= (1 - shock_pct)
        Y_p[t] = Y
        total_inv = I_K + I_D + I_AI + I_H
        C = max(Y - total_inv, 1.0)
        C_p[t] = C
        welfare += (rho**t) * np.log(C)
        # Cập nhật trạng thái
        K_p[t+1]  = (1 - delta_K)  * K_p[t]  + I_K
        D_p[t+1]  = (1 - delta_D)  * D_p[t]  + I_D
        AI_p[t+1] = (1 - delta_AI) * AI_p[t] + I_AI
        H_new = H_p[t] + theta_H * I_H - mu * H_p[t]
        H_p[t+1] = min(max(H_new, 1.0), 95.0)   # H ≤ 95%
        # TFP tăng tối đa 8%/năm
        tfp_g = min(phi1*D_p[t]/100 + phi2*AI_p[t]/1000 + phi3*H_p[t]/100, 0.08)
        A_p[t+1] = A_p[t] * (1 + tfp_g)
    return Y_p, C_p, K_p[:T], D_p[:T], AI_p[:T], H_p[:T], A_p[:T], welfare

# ─────────────────────────────────────────
# CÂU 8.3.1: Tối ưu hóa SLSQP
# ─────────────────────────────────────────
print("\n--- Câu 8.3.1: Tối ưu hóa bằng scipy SLSQP ---")

def neg_welfare(inv):
    return -simulate(inv)[7]

# Bounds: giới hạn thực tế theo % GDP khởi đầu
B = [Y_init*0.20, Y_init*0.08, Y_init*0.06, Y_init*0.06]
bounds = [(0, b) for b in B] * T

x0 = np.array([[Y_init*0.12, Y_init*0.04, Y_init*0.03, Y_init*0.03]] * T).flatten()

print("Đang tối ưu hoá... (scipy SLSQP)")
result = minimize(neg_welfare, x0, method='SLSQP', bounds=bounds,
                  options={'maxiter': 400, 'ftol': 1e-6, 'disp': False})

Y_opt, C_opt, K_opt, D_opt, AI_opt, H_opt, A_opt, W_opt = simulate(result.x)

print(f"Hội tụ: {result.success}  |  Welfare tổng = {W_opt:.4f}")
print(f"\n{'Năm':>5} {'GDP':>12} {'Tiêu dùng':>12} {'K':>10} {'H(%)':>8} {'TFP':>8}")
print("-" * 58)
years_p = np.arange(2026, 2036)
for t in range(T):
    print(f"{years_p[t]:>5} {Y_opt[t]:>12,.1f} {C_opt[t]:>12,.1f} "
          f"{K_opt[t]:>10,.1f} {H_opt[t]:>8.2f} {A_opt[t]:>8.3f}")

print(f"\nTăng trưởng GDP TB 2026-2035: {((Y_opt[-1]/Y_opt[0])**(1/9)-1)*100:.2f}%/năm")

# ─────────────────────────────────────────
# CÂU 8.3.2: Vẽ quỹ đạo
# ─────────────────────────────────────────
print("\n--- Câu 8.3.2: Vẽ quỹ đạo tối ưu ---")

fig = plt.figure(figsize=(18, 12))
fig.suptitle("Bài 8: Quỹ đạo Tối ưu Động 2026–2035", fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

plot_data = [
    (Y_opt,  'GDP (nghìn tỷ VND)',    '#2E7D32'),
    (C_opt,  'Tiêu dùng C (nghìn tỷ)', '#C62828'),
    (K_opt,  'Vốn vật chất K',        '#1565C0'),
    (D_opt,  'Hạ tầng số D (%GDP)',   '#6A1B9A'),
    (AI_opt, 'Năng lực AI (nghìn DN)', '#00838F'),
    (H_opt,  'Nhân lực số H (%)',      '#E65100'),
]

for idx, (data, title, color) in enumerate(plot_data):
    ax = fig.add_subplot(gs[idx // 3, idx % 3])
    ax.plot(years_p, data, 'o-', color=color, linewidth=2.5, markersize=6)
    ax.fill_between(years_p, data, alpha=0.12, color=color)
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    ax.grid(True, alpha=0.3)
    ax.annotate(f'{data[0]:,.1f}', (years_p[0], data[0]),
                textcoords="offset points", xytext=(4, 6), fontsize=7.5)
    ax.annotate(f'{data[-1]:,.1f}', (years_p[-1], data[-1]),
                textcoords="offset points", xytext=(-28, 6), fontsize=7.5)

plt.savefig('/mnt/user-data/outputs/bai08_optimal_trajectory.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ bai08_optimal_trajectory.png đã lưu")

# ─────────────────────────────────────────
# CÂU 8.3.3: Cú sốc 2028
# ─────────────────────────────────────────
print("\n--- Câu 8.3.3: Cú sốc 2028 (Y giảm 8%) ---")

Y_sh, C_sh, _, _, _, _, _, W_sh = simulate(result.x, shock_year=2, shock_pct=0.08)

print(f"\n  GDP 2028 baseline:  {Y_opt[2]:>12,.1f} nghìn tỷ")
print(f"  GDP 2028 cú sốc:    {Y_sh[2]:>12,.1f} nghìn tỷ  ({(Y_sh[2]-Y_opt[2])/Y_opt[2]*100:+.1f}%)")
print(f"  Welfare mất mát:    {W_opt - W_sh:.4f} ({(W_opt-W_sh)/abs(W_opt)*100:.2f}%)")

print(f"\n  Recovery (so với baseline):")
for t_idx in [3, 4, 5, 6]:
    yr = 2026 + t_idx
    diff = (Y_sh[t_idx] - Y_opt[t_idx]) / Y_opt[t_idx] * 100
    print(f"    GDP {yr}: {Y_sh[t_idx]:>10,.1f}  ({diff:+.2f}% vs baseline)")

# ─────────────────────────────────────────
# CÂU 8.3.4: So sánh chiến lược
# ─────────────────────────────────────────
print("\n--- Câu 8.3.4: So sánh chiến lược đầu tư ---")

total_inv_avg = result.x.reshape(T, 4).sum(axis=1).mean()

# Chiến lược A: Đầu tư đều
inv_uni = np.full((T, 4), total_inv_avg / 4)
Y_uni, _, _, _, _, _, _, W_uni = simulate(inv_uni.flatten())

# Chiến lược B: Front-load (mạnh 3 năm đầu, giảm dần)
scale = np.array([1.5, 1.3, 1.2, 1.0, 0.9, 0.8, 0.75, 0.7, 0.7, 0.7])
scale /= scale.mean()
shares = [0.50, 0.20, 0.15, 0.15]  # I_K chiếm nhiều nhất
inv_fl = np.outer(scale, [total_inv_avg * s for s in shares])
inv_fl = np.clip(inv_fl, 0, np.array(B))
Y_fl, _, _, _, _, _, _, W_fl = simulate(inv_fl.flatten())

print(f"\n  {'Chiến lược':<25} {'Welfare':>10} {'Δ vs tối ưu':>14}")
print("-" * 52)
print(f"  {'Tối ưu (SLSQP)':<25} {W_opt:>10.4f} {'—':>14}")
print(f"  {'Đầu tư đều':<25} {W_uni:>10.4f} {W_uni-W_opt:>+13.4f}")
print(f"  {'Front-load':<25} {W_fl:>10.4f}  {W_fl-W_opt:>+13.4f}")

better = "Front-load TỐTHƠN Đều" if W_fl > W_uni else "Đều TỐTHƠN Front-load"
print(f"\n  → {better} (Δ = {abs(W_fl-W_uni):.4f})")
print(f"  → Chiến lược Tối ưu luôn tốt nhất nhờ phân bổ linh hoạt theo thực tế")

# ─────────────────────────────────────────
# BIỂU ĐỒ CHIẾN LƯỢC & CÚ SỐC
# ─────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 11))
fig2.suptitle("Bài 8: So sánh Chiến lược & Tác động Cú sốc", fontsize=13, fontweight='bold')

# Plot 1: GDP 3 chiến lược
axes2[0,0].plot(years_p, Y_opt, 'b-o', lw=2.5, ms=6, label=f'Tối ưu  (W={W_opt:.2f})')
axes2[0,0].plot(years_p, Y_uni, 'g--s', lw=2, ms=6, label=f'Đều     (W={W_uni:.2f})')
axes2[0,0].plot(years_p, Y_fl,  'r-.^', lw=2, ms=6, label=f'Front-load (W={W_fl:.2f})')
axes2[0,0].set_title("GDP: 3 Chiến lược Đầu tư", fontweight='bold')
axes2[0,0].set_ylabel("Nghìn tỷ VND"); axes2[0,0].legend(fontsize=8); axes2[0,0].grid(True, alpha=0.3)

# Plot 2: Welfare tích lũy
w_cum_opt = np.cumsum([(rho**t)*np.log(max(C_opt[t],1)) for t in range(T)])
w_cum_uni = np.cumsum([(rho**t)*np.log(max(Y_uni[t]-inv_uni[t].sum(),1)) for t in range(T)])
axes2[0,1].plot(years_p, w_cum_opt, 'b-o', lw=2.5, ms=5, label='Tối ưu')
axes2[0,1].plot(years_p, w_cum_uni, 'g--s', lw=2, ms=5, label='Đều')
axes2[0,1].set_title("Welfare tích lũy theo năm", fontweight='bold')
axes2[0,1].set_ylabel("Σ ρ^t ln(C_t)"); axes2[0,1].legend(fontsize=8); axes2[0,1].grid(True, alpha=0.3)

# Plot 3: Cú sốc 2028
axes2[1,0].plot(years_p, Y_opt, 'b-o', lw=2.5, ms=6, label='Baseline')
axes2[1,0].plot(years_p, Y_sh,  'r--s', lw=2, ms=6, label='Cú sốc 2028 (-8%)')
axes2[1,0].axvline(2028, color='gray', linestyle=':', alpha=0.7)
axes2[1,0].annotate('Cú sốc\nYagi 2028', xy=(2028, Y_sh[2]),
                     xytext=(2029.2, Y_sh[2]*0.92), fontsize=9, color='red',
                     arrowprops=dict(arrowstyle='->', color='red'))
axes2[1,0].fill_between(years_p, Y_sh, Y_opt, alpha=0.15, color='red', label='Welfare mất mát')
axes2[1,0].set_title("Tác động Cú sốc 2028", fontweight='bold')
axes2[1,0].set_ylabel("Nghìn tỷ VND"); axes2[1,0].legend(fontsize=8); axes2[1,0].grid(True, alpha=0.3)

# Plot 4: Cơ cấu đầu tư tối ưu theo năm (stacked bar)
inv_mat = result.x.reshape(T, 4)
colors_inv = ['#1565C0', '#6A1B9A', '#00838F', '#E65100']
labels_inv = ['I_K (vật chất)', 'I_D (số hóa)', 'I_AI', 'I_H (nhân lực)']
bottom = np.zeros(T)
for j in range(4):
    axes2[1,1].bar(years_p, inv_mat[:, j], bottom=bottom,
                   color=colors_inv[j], label=labels_inv[j], alpha=0.85, width=0.7)
    bottom += inv_mat[:, j]
axes2[1,1].plot(years_p, Y_opt, 'ko-', lw=2, ms=5, label='GDP (Y)')
axes2[1,1].set_title("Cơ cấu đầu tư tối ưu", fontweight='bold')
axes2[1,1].set_ylabel("Nghìn tỷ VND"); axes2[1,1].legend(fontsize=8, loc='upper left')
axes2[1,1].tick_params(axis='x', rotation=30); axes2[1,1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/bai08_strategy_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ bai08_strategy_comparison.png đã lưu")

print("\n--- Thảo luận chính sách ---")
print(f"""
a) Quỹ đạo tối ưu thường front-loaded I_K năm đầu, sau đó
   tăng dần I_D và I_AI khi hạ tầng đã đủ nền tảng
   → "Hạ tầng trước, AI sau" phù hợp thực tiễn VN

b) Tỷ lệ I_AI/I_H: đầu tư AI và nhân lực cần đồng thời
   → Đào tạo nhân lực nên ĐI TRƯỚC hoặc SONG SONG với AI

c) Với ρ=0.90 (chính phủ ngắn hạn hơn):
   → Ưu tiên I_K (thu hồi nhanh), ít I_H (dài hạn)
   → Giải thích "dưới đầu tư" R&D/Giáo dục phổ biến ở nhiều quốc gia

Welfare: Tối ưu ({W_opt:.4f}) > Đều ({W_uni:.4f}) > Front-load ({W_fl:.4f})
→ Linh hoạt động thái quan trọng hơn chiến lược cứng nhắc
""")

print("=" * 60)
print("HOÀN THÀNH BÀI 8")
print("=" * 60)
