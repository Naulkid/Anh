"""
Bài 11: Học tăng cường (Q-learning) cho chính sách kinh tế thích nghi
=======================================================================
MDP: Nền kinh tế Việt Nam
  State: (GDP_growth, Digital, AI_capacity, Unemployment) — mỗi yếu tố 3 mức
         → 3^4 = 81 trạng thái rời rạc
  Action: 5 chiến lược phân bổ ngân sách
  Reward: R = w1*ΔGDP - w2*ΔUnemp - w3*CyberRisk - w4*Emission

Yêu cầu:
    11.3.1 - Môi trường VietnamEconomyEnv
    11.3.2 - Q-learning 10,000 episodes với ε-greedy
    11.3.3 - Chính sách π*(s) tại các trạng thái điển hình
    11.3.4 - So sánh π* với rule-based (a1, a3, a0, random)
    11.3.5 - DQN (stable-baselines3) — nếu khả dụng
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("BAI 11: Q-LEARNING - CHINH SACH KINH TE THICH NGHI")
print("=" * 60)

# ─────────────────────────────────────────
# THAM SO MO HINH
# ─────────────────────────────────────────
# 5 chien luoc phan bo [K%, D%, AI%, H%]
ALLOCATIONS = {
    0: np.array([0.70, 0.10, 0.10, 0.10]),  # a0: Truyen thong
    1: np.array([0.40, 0.25, 0.15, 0.20]),  # a1: Can bang
    2: np.array([0.25, 0.45, 0.15, 0.15]),  # a2: So hoa nhanh
    3: np.array([0.20, 0.20, 0.45, 0.15]),  # a3: AI dan dat
    4: np.array([0.30, 0.20, 0.10, 0.40]),  # a4: Bao trum so
}
ACTION_NAMES = ['Truyen thong', 'Can bang', 'So hoa nhanh', 'AI dan dat', 'Bao trum so']

# He so reward
W_REWARD   = np.array([0.40, 0.25, 0.20, 0.15])
CYBER_RISK = np.array([0.05, 0.10, 0.15, 0.25, 0.08])
EMISSION   = np.array([0.20, 0.12, 0.10, 0.15, 0.08])
UNEMP_DELTA= np.array([0.05, 0.02, 0.01,-0.03, 0.03])

# Tham so Cobb-Douglas
ALPHA = {'K':0.33,'L':0.42,'D':0.10,'AI':0.08,'H':0.07}
BUDGET = 1000.0  # nghin ty VND / nam

# ─────────────────────────────────────────
# CAU 11.3.1: MOI TRUONG
# ─────────────────────────────────────────
print("\n--- Cau 11.3.1: Dinh nghia VietnamEconomyEnv ---")

class VietnamEconomyEnv:
    """Moi truong kinh te VN don gian hoa cho Q-learning"""

    # Nguong phan muc (3 muc: low/med/high)
    GDP_TH  = [5.0,  7.5]   # % tang truong GDP
    DIG_TH  = [15.0, 25.0]  # % kinh te so / GDP
    AI_TH   = [55.0, 75.0]  # nghin DN so
    UNP_TH  = [3.0,  5.0]   # % that nghiep

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.T   = 10
        self.n_actions = 5
        self.n_states  = 81   # 3^4

    def _disc(self, val, thresholds):
        """Chuyen gia tri lien tuc sang [0,1,2]"""
        return min(int(np.searchsorted(thresholds, val)), 2)

    def _state_idx(self, s):
        """Vector [g,d,a,u] -> index 0..80"""
        return int(s[0]*27 + s[1]*9 + s[2]*3 + s[3])

    def _to_vec(self, gdp_g, digital, ai, unemp):
        return np.array([
            self._disc(gdp_g,  self.GDP_TH),
            self._disc(digital, self.DIG_TH),
            self._disc(ai,      self.AI_TH),
            self._disc(unemp,   self.UNP_TH),
        ], dtype=int)

    def _production(self):
        return (self.A * (self.K**ALPHA['K']) * (self.L**ALPHA['L']) *
                (self.D**ALPHA['D']) * (self.AI**ALPHA['AI']) * (self.H**ALPHA['H']))

    def reset(self):
        # Trang thai thuc te VN 2026
        self.K     = 27500.0
        self.L     = 53.9
        self.D     = 20.3   # % (medium)
        self.AI    = 62.0   # nghin DN (low-medium)
        self.H     = 30.0   # %
        self.A     = 30.9   # TFP
        self.unemp = 2.3    # % that nghiep
        self.t     = 0
        self.Y_prev = self._production()
        gdp_g_init  = 7.0
        s = self._to_vec(gdp_g_init, self.D, self.AI, self.unemp)
        return s

    def step(self, action):
        alloc = ALLOCATIONS[action]
        IK, ID, IAI, IH = alloc * BUDGET

        # Cap nhat kinh te
        self.K    = 0.95*self.K + IK
        self.D    = min(0.88*self.D + ID/50.0, 50.0)
        self.AI   = 0.85*self.AI + IAI/10.0
        self.H    = min(self.H + 0.8*IH/200 - 0.02*self.H, 95.0)
        self.L   *= 1.003
        # TFP noi sinh (gioi han 5%/nam)
        tfp_g = min(0.003*self.D/100 + 0.002*self.AI/1000 + 0.004*self.H/100, 0.05)
        self.A   *= (1 + tfp_g)

        Y_new  = self._production()
        gdp_g  = (Y_new - self.Y_prev) / max(self.Y_prev, 1.0) * 100
        self.Y_prev = Y_new

        # That nghiep
        noise = self.rng.normal(0, 0.1)
        self.unemp = max(0.5, min(10.0, self.unemp + UNEMP_DELTA[action] + noise))

        # Reward
        r_gdp   = min(max(gdp_g / 10.0, -1.0), 1.0)
        r_unemp = -UNEMP_DELTA[action]
        r_cyber = -CYBER_RISK[action]
        r_emit  = -EMISSION[action]
        reward  = (W_REWARD[0]*r_gdp + W_REWARD[1]*r_unemp +
                   W_REWARD[2]*r_cyber + W_REWARD[3]*r_emit)

        next_s = self._to_vec(gdp_g, self.D, self.AI, self.unemp)
        self.t += 1
        done   = self.t >= self.T
        return next_s, reward, done

    def state_idx(self, s): return self._state_idx(s)


env = VietnamEconomyEnv(seed=42)
print(f"  Trang thai: 3^4 = {env.n_states} trang thai roi rac")
print(f"  Hanh dong: {env.n_actions} chien luoc phan bo ngan sach")
print(f"  Horizon T = {env.T} nam / episode")
print(f"  Reward: R = 0.40*dGDP - 0.25*dUnemp - 0.20*Cyber - 0.15*Emission")

# ─────────────────────────────────────────
# CAU 11.3.2: Q-LEARNING
# ─────────────────────────────────────────
print("\n--- Cau 11.3.2: Huan luyen Q-learning (10,000 episodes) ---")

Q = np.zeros((81, 5))
LR      = 0.10   # alpha
GAMMA   = 0.95   # discount
EPS_MAX = 1.0
EPS_MIN = 0.05
N_EP    = 10000

rewards_hist = []
eps_hist     = []
np.random.seed(42)

for ep in range(N_EP):
    s     = env.reset()
    s_idx = env.state_idx(s)
    ep_r  = 0.0
    # epsilon giam tuyen tinh den nua training
    eps   = max(EPS_MIN, EPS_MAX - (EPS_MAX - EPS_MIN) * ep / (N_EP * 0.6))

    done = False
    while not done:
        # e-greedy
        if np.random.rand() < eps:
            a = np.random.randint(5)
        else:
            a = int(np.argmax(Q[s_idx]))

        ns, r, done = env.step(a)
        ns_idx = env.state_idx(ns)

        # Bellman update
        Q[s_idx, a] += LR * (r + GAMMA * np.max(Q[ns_idx]) - Q[s_idx, a])

        s_idx = ns_idx
        ep_r += r

    rewards_hist.append(ep_r)
    eps_hist.append(eps)

print(f"  Huan luyen xong!")
print(f"  Reward TB 100 ep dau:   {np.mean(rewards_hist[:100]):+.4f}")
print(f"  Reward TB 100 ep cuoi:  {np.mean(rewards_hist[-100:]):+.4f}")
print(f"  Cai thien:              {np.mean(rewards_hist[-100:])-np.mean(rewards_hist[:100]):+.4f}")

# ─────────────────────────────────────────
# CAU 11.3.3: CHINH SACH PI*(S)
# ─────────────────────────────────────────
print("\n--- Cau 11.3.3: Chinh sach toi uu pi*(s) ---")

def get_action(s_vec):
    return int(np.argmax(Q[env.state_idx(np.array(s_vec, dtype=int))]))

# Cac trang thai dien hinh
test_states = [
    ("VN 2026 (GDP=med,D=med,AI=low,U=med)",  [1,1,0,1]),
    ("Tang truong cao, That nghiep thap",       [2,2,2,0]),
    ("Khung hoang (tat ca low + U=high)",       [0,0,0,2]),
    ("So hoa tot, AI cham (med,high,low,med)",  [1,2,0,1]),
    ("AI cao nhung chua bao trum",              [2,1,2,2]),
]

print(f"\n  {'Trang thai':<44} {'pi*(s)':<18} {'Q-value':>8}")
print("-" * 74)
for name, sv in test_states:
    a    = get_action(sv)
    sidx = env.state_idx(np.array(sv, dtype=int))
    qv   = Q[sidx, a]
    print(f"  {name:<44} {ACTION_NAMES[a]:<18} {qv:>8.4f}")

# ─────────────────────────────────────────
# CAU 11.3.4: SO SANH CAC CHINH SACH
# ─────────────────────────────────────────
print("\n--- Cau 11.3.4: So sanh pi* voi rule-based ---")

def evaluate(policy_fn, n=400, seed=0):
    local_rng = np.random.default_rng(seed)
    local_env = VietnamEconomyEnv(seed=seed)
    all_r = []
    for _ in range(n):
        s = local_env.reset()
        ep_r = 0.0
        done = False
        while not done:
            a = policy_fn(s)
            s, r, done = local_env.step(a)
            ep_r += r
        all_r.append(ep_r)
    return np.array(all_r)

policies = {
    'Q-learning (pi*)':   lambda s: get_action(s),
    'Can bang (a1)':       lambda s: 1,
    'AI dan dat (a3)':     lambda s: 3,
    'Truyen thong (a0)':   lambda s: 0,
    'Random':              lambda s: int(np.random.randint(5)),
}

eval_results = {}
print(f"\n  {'Chinh sach':<24} {'TB Reward':>10} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 62)
for name, fn in policies.items():
    np.random.seed(42)
    r_arr = evaluate(fn, n=300, seed=42)
    eval_results[name] = r_arr
    print(f"  {name:<24} {r_arr.mean():>10.4f} {r_arr.std():>8.4f} "
          f"{r_arr.min():>8.4f} {r_arr.max():>8.4f}")

# So sanh tuong doi
baseline = eval_results['Can bang (a1)'].mean()
print(f"\n  So voi can bang (a1) lam baseline:")
for name, r_arr in eval_results.items():
    delta = r_arr.mean() - baseline
    pct   = delta / abs(baseline) * 100
    print(f"    {name:<24}: {delta:+.4f} ({pct:+.1f}%)")

# ─────────────────────────────────────────
# CAU 11.3.5: DQN
# ─────────────────────────────────────────
print("\n--- Cau 11.3.5: Deep Q-Network (DQN) ---")

dqn_reward_mean = None
try:
    from stable_baselines3 import DQN as SB3_DQN
    import gymnasium as gym_sb3
    from gymnasium import spaces as gym_spaces

    class VNEnvGymnasium(gym_sb3.Env):
        def __init__(self):
            super().__init__()
            self._e = VietnamEconomyEnv(seed=0)
            self.action_space      = gym_spaces.Discrete(5)
            # Flatten state to Box(81,) one-hot cho DQN
            self.observation_space = gym_spaces.Box(0, 2, shape=(4,), dtype=np.int32)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            s = self._e.reset()
            return s.astype(np.int32), {}

        def step(self, action):
            s, r, done = self._e.step(int(action))
            return s.astype(np.int32), float(r), done, False, {}

    gym_env = VNEnvGymnasium()
    dqn = SB3_DQN(
        policy='MlpPolicy',
        env=gym_env,
        learning_rate=1e-3,
        buffer_size=5000,
        batch_size=32,
        gamma=0.95,
        exploration_fraction=0.4,
        exploration_final_eps=0.05,
        verbose=0,
        seed=42,
    )
    print("  Huan luyen DQN (15,000 steps)...")
    dqn.learn(total_timesteps=15000)

    # Danh gia DQN
    dqn_ep_rewards = []
    obs, _ = gym_env.reset()
    for _ in range(200):
        ep_r = 0.0
        obs, _ = gym_env.reset()
        done   = False
        while not done:
            act, _ = dqn.predict(obs, deterministic=True)
            obs, r, done, _, _ = gym_env.step(act)
            ep_r += r
        dqn_ep_rewards.append(ep_r)

    dqn_reward_mean = np.mean(dqn_ep_rewards)
    ql_mean = eval_results['Q-learning (pi*)'].mean()
    print(f"  DQN reward TB:        {dqn_reward_mean:.4f}")
    print(f"  Q-learning reward TB: {ql_mean:.4f}")
    better = "DQN tot hon" if dqn_reward_mean > ql_mean else "Q-learning tot hon"
    print(f"  -> {better} (delta={dqn_reward_mean-ql_mean:+.4f})")
    print("  -> DQN co loi khi state space lon/lien tuc")
    print("     Q-learning tabular du tot voi 81 trang thai roi rac")

except ImportError as ie:
    print(f"  stable-baselines3/gymnasium chua cai: {ie}")
    print("  -> Q-learning tabular phu hop voi 81 trang thai roi rac")
    print("  -> DQN dung khi: state space lon, du lieu lien tuc, can scale")
except Exception as ex:
    print(f"  DQN loi: {ex}")
    print("  -> Dung Q-learning tabular cho bai nay la du")

# ─────────────────────────────────────────
# BIEU DO TONG HOP
# ─────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Bai 11: Q-learning - Chinh sach Kinh te Thich nghi",
             fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

# --- Plot 1: Learning curve ---
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(rewards_hist, alpha=0.15, color='#90CAF9', lw=0.5, label='_nolegend_')
w = 300
sm = np.convolve(rewards_hist, np.ones(w)/w, mode='valid')
ax1.plot(range(w-1, N_EP), sm, color='#1565C0', lw=2.5,
         label=f'Moving avg (w={w})')
ax1.axhline(np.mean(rewards_hist[-1000:]), color='red', ls='--', alpha=0.7,
            label=f'TB 1000 ep cuoi: {np.mean(rewards_hist[-1000:]):.3f}')
ax1.set_title("Learning Curve Q-learning (10,000 episodes)", fontweight='bold')
ax1.set_xlabel("Episode"); ax1.set_ylabel("Tong Reward")
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
# Epsilon tren truc phu
ax1b = ax1.twinx()
ax1b.plot(eps_hist, color='orange', alpha=0.4, lw=1, label='Epsilon')
ax1b.set_ylabel("Epsilon", color='orange')
ax1b.tick_params(axis='y', labelcolor='orange')
ax1b.legend(loc='center right', fontsize=8)

# --- Plot 2: So sanh reward cac chinh sach ---
ax2 = fig.add_subplot(gs[0, 2])
names = list(eval_results.keys())
means = [eval_results[n].mean() for n in names]
stds  = [eval_results[n].std()  for n in names]
colors_p = ['#C62828', '#1565C0', '#E65100', '#78909C', '#B0BEC5']
ax2.bar(range(len(names)), means, yerr=stds, color=colors_p,
        alpha=0.85, capsize=5, edgecolor='white')
ax2.set_xticks(range(len(names)))
ax2.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=7.5)
ax2.set_title("So sanh Reward TB", fontweight='bold')
ax2.set_ylabel("Reward TB")
for i, (m, s) in enumerate(zip(means, stds)):
    ax2.text(i, m + 0.005, f'{m:.3f}', ha='center', fontsize=8, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# --- Plot 3: Q-value heatmap (GDP × Digital, AI=low, Unemp=med) ---
ax3 = fig.add_subplot(gs[1, 0])
q_heat = np.zeros((3, 3))
for g in range(3):
    for d in range(3):
        si = env.state_idx(np.array([g, d, 0, 1]))
        q_heat[g, d] = np.max(Q[si])
im3 = ax3.imshow(q_heat, cmap='RdYlGn', aspect='auto')
ax3.set_xticks([0,1,2]); ax3.set_yticks([0,1,2])
ax3.set_xticklabels(['Low','Med','High']); ax3.set_yticklabels(['Low','Med','High'])
ax3.set_xlabel("Digital Index"); ax3.set_ylabel("GDP Growth")
ax3.set_title("Max Q-value\n(AI=Low, Unemp=Med)", fontweight='bold')
plt.colorbar(im3, ax=ax3, shrink=0.8)
for g in range(3):
    for d in range(3):
        ax3.text(d, g, f'{q_heat[g,d]:.3f}', ha='center', va='center', fontsize=9,
                 color='black' if q_heat[g,d] > q_heat.mean() else 'white')

# --- Plot 4: Policy map (optimal action heatmap) ---
ax4 = fig.add_subplot(gs[1, 1])
pol_map = np.zeros((3, 3), dtype=int)
for g in range(3):
    for d in range(3):
        pol_map[g, d] = get_action([g, d, 1, 1])
# custom colormap 5 mau
from matplotlib.colors import ListedColormap
cmap5 = ListedColormap(['#78909C','#1565C0','#6A1B9A','#E65100','#2E7D32'])
im4 = ax4.imshow(pol_map, cmap=cmap5, vmin=0, vmax=4, aspect='auto')
ax4.set_xticks([0,1,2]); ax4.set_yticks([0,1,2])
ax4.set_xticklabels(['Low','Med','High']); ax4.set_yticklabels(['Low','Med','High'])
ax4.set_xlabel("Digital Index"); ax4.set_ylabel("GDP Growth")
ax4.set_title("Chinh sach pi*(s)\n(AI=Med, Unemp=Med)", fontweight='bold')
for g in range(3):
    for d in range(3):
        ax4.text(d, g, ACTION_NAMES[pol_map[g,d]][:8],
                 ha='center', va='center', fontsize=7.5, color='white', fontweight='bold')

# --- Plot 5: Phan phoi reward ---
ax5 = fig.add_subplot(gs[1, 2])
for name, color in zip(list(eval_results.keys())[:4], colors_p[:4]):
    ax5.hist(eval_results[name], bins=25, alpha=0.55, color=color,
             label=name.split('(')[0].strip(), density=True)
ax5.set_xlabel("Episode Reward")
ax5.set_ylabel("Mat do")
ax5.set_title("Phan phoi Reward - 4 Chinh sach", fontweight='bold')
ax5.legend(fontsize=7.5); ax5.grid(True, alpha=0.3)

plt.savefig('/mnt/user-data/outputs/bai11_qlearning_results.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ Bieu do da luu: bai11_qlearning_results.png")

# ─────────────────────────────────────────
# THAO LUAN CHINH SACH
# ─────────────────────────────────────────
print("\n--- Thao luan chinh sach ---")

a_vn2026 = get_action([1, 1, 0, 1])
a_high   = get_action([2, 2, 2, 0])
a_crisis = get_action([0, 0, 0, 2])

ql_mean  = eval_results['Q-learning (pi*)'].mean()
a1_mean  = eval_results['Can bang (a1)'].mean()
a3_mean  = eval_results['AI dan dat (a3)'].mean()

print(f"""
a) VN 2026 (GDP=med, D=med, AI=low, U=med):
   pi*(s) = '{ACTION_NAMES[a_vn2026]}'
   -> Phu hop "quick win": uu tien tang AI va nhan luc de bat kip

b) Khi GDP cao, AI cao, Unemp thap (high, high, high, low):
   pi*(s) = '{ACTION_NAMES[a_high]}'
   -> "Consolidation": cung co va mo rong thanh qua

c) Khung hoang (low, low, low, high):
   pi*(s) = '{ACTION_NAMES[a_crisis]}'
   -> "Social safety net": uu tien viec lam va nhan luc

d) So sanh hieu qua (reward TB 300 eps):
   Q-learning: {ql_mean:+.4f}
   Can bang:   {a1_mean:+.4f}  (delta={ql_mean-a1_mean:+.4f})
   AI dan dat: {a3_mean:+.4f}  (delta={ql_mean-a3_mean:+.4f})
   -> Q-learning thich nghi tot hon nho phan biet trang thai kinh te

e) Han che va khuyen nghi:
   - MDP nay don gian hoa (81 trang thai, 10 nam)
   - Thuc te: them bien vi mo (lam phat, ty gia, xuat khau)
   - pi*(s) chi la goi y ky thuat, KHONG thay the quyet dinh chinh tri
   - Tich hop: dung pi*(s) nhu "dashboard canh bao", con quyet dinh
     cuoi cung van do cac co quan Chinh phu va Quoc hoi
""")

print("=" * 60)
print("HOAN THANH BAI 11")
print("=" * 60)
