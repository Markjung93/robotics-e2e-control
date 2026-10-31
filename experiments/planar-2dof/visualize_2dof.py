#!/usr/bin/env python3
"""
planar-2dof 결과 시각화 (dynamics_compare_2dof.py 의 함수를 그대로 가져다 씁니다)

만드는 것 (기본 위치: experiments/planar-2dof/figures/)
  error_distribution.png : 손 유도 대비 오차의 분포 (무작위 상태 N개, 방법 x 양)
  freefall_compare.png   : 토크 0 자유낙하. 관절각, 방법 간 차이, 에너지 변화
  freefall.gif           : 같은 낙하를 팔 모양으로 보여 줌 (손 유도 적분 vs MuJoCo mj_step)

수식 (토크 0)
  q̈ = M(q)^-1 ( -C(q,q̇) q̇ - G(q) ),   E = 1/2 q̇ᵀ M(q) q̇ + Σ m_i g y_ci
  마찰이 없으면 E 는 시간에 따라 일정해야 합니다 (수치 적분 오차만큼만 변함).

주의
  - 이중 진자는 혼돈계라서, 방법 간 차이가 아주 작아도 시간이 지나면 지수적으로 커집니다.
    모델이 틀려서가 아니라 계의 성질입니다. 그래서 짧은 구간(기본 3초)만 비교합니다.
  - 모든 방법에 같은 적분기(RK4)와 같은 시간 간격을 씁니다. 방법 간 차이는 모델의 차이만 반영합니다.
    MuJoCo 는 mj_forward 로 만든 가속도를 같은 RK4 로 적분한 곡선과, 실제 시뮬레이터 mj_step(RK4) 곡선을 둘 다 그립니다.
  - 마찰, 백래시, 센서 노이즈, 제어 주기 지터가 없는 이상 모델입니다 (Sim-to-Real 갭은 포함하지 않음).

사용 (레포 루트에서)
  python experiments/planar-2dof/visualize_2dof.py
  python experiments/planar-2dof/visualize_2dof.py --T 5 --q0 30 60 --no-gif

필요 패키지: numpy, pyyaml, pin(Pinocchio), mujoco, matplotlib (GIF 는 pillow, matplotlib 설치 때 함께 설치됨)
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dynamics_compare_2dof as dc  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")                 # 화면 없이 파일로 저장
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
COLORS = {"NE": "#2ca02c", "pin": "#d62728", "mj": "#9467bd", "mj_step": "#111111", "hand": "#1f77b4"}
FLOOR = 1e-17                          # 로그 축에서 0 을 그릴 수 없으므로 바닥값


# ------------------------------------------------------------
# 모델 읽기 (dynamics_compare_2dof.py 와 같은 규칙)
# ------------------------------------------------------------
def load_models(models_dir):
    yaml_path = os.path.join(models_dir, dc.YAML_NAME)
    urdf = os.path.join(models_dir, dc.URDF_NAME)
    mjcf = os.path.join(models_dir, dc.MJCF_NAME)
    p = dc.load_params_yaml(yaml_path) if os.path.isfile(yaml_path) else dc.Params()
    if not (os.path.isfile(yaml_path) and os.path.isfile(urdf) and os.path.isfile(mjcf)):
        print(f"[알림] {models_dir} 에 params.yaml/URDF/MJCF 가 모두 없어 내장 생성기 모델을 사용합니다.")
        urdf = mjcf = None
    return p, urdf, mjcf


# ------------------------------------------------------------
# 그림 1: 오차 분포
# ------------------------------------------------------------
def collect_errors(p, pb, mb, n, seed):
    rng = np.random.default_rng(seed)
    errs = {}
    for _ in range(n):
        q = rng.uniform(-np.pi, np.pi, 2)
        v = rng.uniform(-3, 3, 2)
        a = rng.uniform(-5, 5, 2)
        t = rng.uniform(-10, 10, 2)
        r = dc.evaluate(q, v, a, t, p, pb, mb)
        for m in ("NE", "pin", "mj"):
            for k in r[m]:
                errs.setdefault((m, k), []).append(dc.maxerr(r[m][k], r["hand"][k]))
    return errs


def plot_error_distribution(errs, n, tol, path):
    quantities = [("M", r"$M(q)$"), ("Cv", r"$C(q,\dot q)\dot q$"), ("G", r"$G(q)$"),
                  ("tau_id", r"$\tau$ (inverse dyn.)"), ("qdd_fd", r"$\ddot q$ (forward dyn.)"),
                  ("tip", "tip pos (FK)"), ("J", "Jacobian")]
    methods = [("NE", "Newton-Euler"), ("pin", "Pinocchio"), ("mj", "MuJoCo")]
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=130)
    width = 0.24
    for j, (m, label) in enumerate(methods):
        pos, data = [], []
        for i, (k, _) in enumerate(quantities):
            if (m, k) in errs:
                pos.append(i + (j - 1) * width)
                data.append(np.maximum(errs[(m, k)], FLOOR))
        bp = ax.boxplot(data, positions=pos, widths=width * 0.9, patch_artist=True, manage_ticks=False,
                        flierprops=dict(marker=".", markersize=2, markeredgecolor=COLORS[m], alpha=0.5),
                        medianprops=dict(color="black", linewidth=1.0), whiskerprops=dict(color=COLORS[m]),
                        capprops=dict(color=COLORS[m]), boxprops=dict(color=COLORS[m]))
        for box in bp["boxes"]:
            box.set_facecolor(COLORS[m])
            box.set_alpha(0.55)
        ax.plot([], [], color=COLORS[m], linewidth=6, alpha=0.7, label=label)
    ax.axhline(tol, color="crimson", linestyle="--", linewidth=1.2)
    ax.text(len(quantities) - 0.55, tol * 1.6, f"tolerance {tol:g}", color="crimson", ha="right", va="bottom", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(1e-17, 1e-7)
    ax.set_xticks(range(len(quantities)))
    ax.set_xticklabels([lab for _, lab in quantities], fontsize=9)
    ax.set_ylabel("max abs error vs hand-derived E-L")
    ax.set_title(f"planar 2-DOF arm: error distribution over {n} random states "
                 f"(box: quartiles, line: median, dots: outliers)", fontsize=10)
    ax.grid(True, axis="y", which="major", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, ncol=3, frameon=False)
    fig.text(0.995, 0.005, "errors that are exactly 0 are drawn at the bottom edge (1e-17)", ha="right", va="bottom",
             fontsize=7, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------
# 자유낙하 (토크 0) 적분
# ------------------------------------------------------------
def rk4_step(acc, q, v, dt):
    def f(qq, vv):
        return vv, acc(qq, vv)
    k1q, k1v = f(q, v)
    k2q, k2v = f(q + 0.5 * dt * k1q, v + 0.5 * dt * k1v)
    k3q, k3v = f(q + 0.5 * dt * k2q, v + 0.5 * dt * k2v)
    k4q, k4v = f(q + dt * k3q, v + dt * k3v)
    return (q + dt / 6.0 * (k1q + 2 * k2q + 2 * k3q + k4q),
            v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v))


def integrate(acc, q0, n, dt):
    q, v = np.array(q0, float), np.zeros(2)
    Q, V = np.zeros((n + 1, 2)), np.zeros((n + 1, 2))
    Q[0], V[0] = q, v
    for i in range(n):
        q, v = rk4_step(acc, q, v, dt)
        Q[i + 1], V[i + 1] = q, v
    return Q, V


def integrate_mj_step(mb, q0, n, dt):
    """MuJoCo 의 실제 시뮬레이터 mj_step (적분기만 RK4 로 바꿈. 모델 파일은 그대로)"""
    mj = mb.mj
    m = mb.m
    m.opt.integrator = mj.mjtIntegrator.mjINT_RK4
    m.opt.timestep = dt
    d = mj.MjData(m)
    d.qpos[:] = q0
    d.qvel[:] = 0.0
    Q, V = np.zeros((n + 1, 2)), np.zeros((n + 1, 2))
    Q[0] = d.qpos
    for i in range(n):
        mj.mj_step(m, d)
        Q[i + 1], V[i + 1] = d.qpos, d.qvel
    return Q, V


def energy(Q, V, p):
    """E = 1/2 q̇ᵀ M q̇ + Σ m_i g y_ci  (y_ci 는 무게중심의 높이)"""
    E = np.zeros(len(Q))
    for i, (q, v) in enumerate(zip(Q, V)):
        M = dc.hand_M(q, p)
        y1 = p.lc1 * np.sin(q[0])
        y2 = p.l1 * np.sin(q[0]) + p.lc2 * np.sin(q[0] + q[1])
        E[i] = 0.5 * v @ M @ v + p.g * (p.m1 * y1 + p.m2 * y2)
    return E


def simulate_all(p, pb, mb, q0, T, dt):
    n = int(round(T / dt))
    z = np.zeros(2)
    acc = {
        "hand": lambda q, v: np.linalg.solve(dc.hand_M(q, p), -dc.hand_C(q, v, p) @ v - dc.hand_G(q, p)),
        "NE": lambda q, v: dc.evaluate(q, v, z, z, p, None, None)["NE"]["qdd_fd"],
        "pin": lambda q, v: pb.aba(q, v, z),
        "mj": lambda q, v: mb.forward(q, v, z),
    }
    traj = {k: integrate(a, q0, n, dt) for k, a in acc.items()}
    traj["mj_step"] = integrate_mj_step(mb, q0, n, dt)
    return np.arange(n + 1) * dt, traj


# ------------------------------------------------------------
# 그림 2: 자유낙하 비교
# ------------------------------------------------------------
def plot_freefall(t, traj, p, path):
    names = {"NE": "Newton-Euler + RK4", "pin": "Pinocchio ABA + RK4",
             "mj": "MuJoCo mj_forward + RK4", "mj_step": "MuJoCo mj_step (RK4)"}
    Qh = traj["hand"][0]
    fig, axs = plt.subplots(3, 1, figsize=(8.2, 9.6), dpi=130, sharex=True)

    ax = axs[0]
    ax.plot(t, np.degrees(Qh[:, 0]), color="#1f77b4", label=r"$q_1$ hand-derived E-L (RK4)")
    ax.plot(t, np.degrees(Qh[:, 1]), color="#ff7f0e", label=r"$q_2$ hand-derived E-L (RK4)")
    Qm = traj["mj_step"][0]
    ax.plot(t, np.degrees(Qm[:, 0]), "--", color="black", linewidth=1.0, label="MuJoCo mj_step (RK4)")
    ax.plot(t, np.degrees(Qm[:, 1]), "--", color="black", linewidth=1.0)
    ax.set_ylabel("joint angle [deg]")
    ax.set_title("Planar 2-DOF arm, zero torque (free fall from rest)", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="best")
    ax.grid(True, alpha=0.3)

    ax = axs[1]
    for k in ("NE", "pin", "mj", "mj_step"):
        diff = np.max(np.abs(traj[k][0] - Qh), axis=1)
        ax.semilogy(t, np.maximum(diff, FLOOR), color=COLORS[k], label=names[k], linewidth=1.3)
    ax.set_ylabel(r"max$_j$ $|q_j - q_j^{hand}|$ [rad]")
    ax.set_title("difference from the hand-derived model (round-off level; slowly amplified by the chaotic dynamics)", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(True, which="major", alpha=0.3)

    ax = axs[2]
    for k, lab in (("hand", "hand-derived E-L + RK4"), ("pin", "Pinocchio ABA + RK4"), ("mj_step", "MuJoCo mj_step (RK4)")):
        E = energy(traj[k][0], traj[k][1], p)
        ax.semilogy(t, np.maximum(np.abs(E - E[0]) / abs(E[0]), FLOOR), color=COLORS[k], label=lab, linewidth=1.3)
    ax.set_ylabel(r"$|E(t)-E(0)|\,/\,|E(0)|$")
    ax.set_xlabel("time [s]")
    ax.set_title("energy drift without friction: RK4 truncation error only (curves overlap: same integrator)", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(True, which="major", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------
# GIF
# ------------------------------------------------------------
def make_gif(t, traj, p, path, fps=25, trail=30):
    from matplotlib.animation import FuncAnimation, PillowWriter

    dt = t[1] - t[0]
    step = max(1, int(round(1.0 / fps / dt)))
    idx = np.arange(0, len(t), step)
    Qh, Qm = traj["hand"][0], traj["mj_step"][0]

    def joints(q):
        j1 = np.array([p.l1 * np.cos(q[0]), p.l1 * np.sin(q[0])])
        tip, _ = dc.hand_fk(q, p)
        return np.array([[0.0, 0.0], j1, tip])

    tips = np.array([dc.hand_fk(Qh[i], p)[0] for i in range(len(t))])
    R = p.l1 + p.l2 + 0.25
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=70)
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x [m]", fontsize=8)
    ax.set_ylabel("y [m]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.annotate("", xy=(-R + 0.2, -R + 0.1), xytext=(-R + 0.2, -R + 0.7), arrowprops=dict(arrowstyle="-|>", color="gray"))
    ax.text(-R + 0.3, -R + 0.35, "g", color="gray", fontsize=9)
    (l1h,) = ax.plot([], [], "-", color="#1f77b4", linewidth=5, solid_capstyle="round", label="hand-derived E-L (RK4)")
    (l2h,) = ax.plot([], [], "-", color="#ff7f0e", linewidth=5, solid_capstyle="round")
    (lm,) = ax.plot([], [], "--", color="black", linewidth=1.0, label="MuJoCo mj_step (RK4)")
    (tr,) = ax.plot([], [], "-", color="crimson", linewidth=1.2, alpha=0.7)
    ax.plot([0], [0], "ko", markersize=5)
    ttl = ax.set_title("", fontsize=9)
    ax.legend(loc="upper right", fontsize=6.5, frameon=False)

    def update(n):
        i = idx[n]
        h = joints(Qh[i])
        l1h.set_data(h[:2, 0], h[:2, 1])
        l2h.set_data(h[1:, 0], h[1:, 1])
        m = joints(Qm[i])
        lm.set_data(m[:, 0], m[:, 1])
        j0 = max(0, i - trail * step)
        tr.set_data(tips[j0:i + 1, 0], tips[j0:i + 1, 1])
        ttl.set_text(f"zero torque free fall   t = {t[i]:.2f} s")
        return l1h, l2h, lm, tr, ttl

    anim = FuncAnimation(fig, update, frames=len(idx), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="planar-2dof 결과 시각화 (오차 분포, 자유낙하 비교, GIF)")
    ap.add_argument("--models-dir", default=dc.DEFAULT_MODELS_DIR, help="params.yaml, URDF, MJCF 가 있는 폴더")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="그림을 저장할 폴더 (기본: experiments/planar-2dof/figures)")
    ap.add_argument("--n-random", type=int, default=1000, help="오차 분포에 쓸 무작위 상태 개수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--T", type=float, default=3.0, help="자유낙하 시간 [s] (혼돈계라 길게 잡으면 방법 간 차이가 커짐)")
    ap.add_argument("--dt", type=float, default=1e-3, help="적분 시간 간격 [s]")
    ap.add_argument("--q0", type=float, nargs=2, default=[30.0, 60.0], metavar=("Q1_DEG", "Q2_DEG"), help="초기 관절각 [deg]")
    ap.add_argument("--no-gif", action="store_true", help="GIF 를 만들지 않음")
    args = ap.parse_args()

    p, urdf, mjcf = load_models(args.models_dir)
    pb, mb = dc.PinBackend(p, urdf), dc.MjBackend(p, mjcf)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"모델: m=({p.m1:g},{p.m2:g}) l=({p.l1:g},{p.l2:g}) I=({p.I1:.6f},{p.I2:.6f}) g={p.g:g}")

    print(f"[1/3] 오차 분포 (무작위 상태 {args.n_random}개) ...")
    errs = collect_errors(p, pb, mb, args.n_random, args.seed)
    f1 = os.path.join(args.out_dir, "error_distribution.png")
    plot_error_distribution(errs, args.n_random, args.tol, f1)
    worst = max(max(v) for v in errs.values())
    print(f"      저장: {f1}   (전체 최대 오차 {worst:.2e}, 합격 기준 {args.tol:g})")

    q0 = np.radians(args.q0)
    print(f"[2/3] 자유낙하 q0=({args.q0[0]:g}°,{args.q0[1]:g}°), T={args.T:g}s, dt={args.dt:g}s, RK4 ...")
    t, traj = simulate_all(p, pb, mb, q0, args.T, args.dt)
    f2 = os.path.join(args.out_dir, "freefall_compare.png")
    plot_freefall(t, traj, p, f2)
    Qh = traj["hand"][0]
    for k, lab in (("NE", "Newton-Euler"), ("pin", "Pinocchio"), ("mj", "MuJoCo forward"), ("mj_step", "MuJoCo mj_step")):
        d = np.max(np.abs(traj[k][0] - Qh))
        print(f"      hand 대비 최대 |Δq| ({lab:>15s}): {d:.2e} rad")
    for k in ("hand", "pin", "mj_step"):
        E = energy(traj[k][0], traj[k][1], p)
        print(f"      최대 상대 에너지 변화 ({k:>7s}): {np.max(np.abs(E - E[0])) / abs(E[0]):.2e}")
    print(f"      저장: {f2}")

    if args.no_gif:
        print("[3/3] GIF 건너뜀 (--no-gif)")
    else:
        print("[3/3] GIF ...")
        f3 = os.path.join(args.out_dir, "freefall.gif")
        make_gif(t, traj, p, f3)
        print(f"      저장: {f3}  ({os.path.getsize(f3) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
