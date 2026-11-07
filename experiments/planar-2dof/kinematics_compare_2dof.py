#!/usr/bin/env python3
"""
2DOF 평면 팔 기구학: 사용자 유도식 vs 기존 hand_fk vs Pinocchio vs MuJoCo (파일 하나로 실행)

구현한 것 (손으로 유도한 식 기준)
  fk, jacobian, det_jacobian = l1 l2 sin(q2), jacobian_inverse (수반행렬/det),
  ik_closed_form (제2 코사인 법칙, 해 0~2개), ik_newton (감쇠 최소제곱 반복 IK), 특이점 판정

규약
  x-y 평면, q1 은 베이스 기준 절대각, q2 는 링크1 기준 상대각 [rad], 길이 [m], J 의 단위 [m/rad]
  기존 dynamics_compare_2dof.py 의 hand_fk 는 교차 검증용 식이라 그대로 두고, 이 파일의 식과 따로 비교합니다.

핵심 수식
  FK     x = l1 cos q1 + l2 cos(q1+q2),  y = l1 sin q1 + l2 sin(q1+q2)
  det J  = l1 l2 sin q2   (q2 = 0 완전히 펴짐, q2 = ±pi 완전히 접힘에서 0)
  IK     c2 = (x²+y²-l1²-l2²)/(2 l1 l2),  q2 = ±acos(c2),
         gamma = acos((x²+y²+l1²-l2²)/(2 r l1)),  phi = atan2(y,x),
         q2>0 이면 q1 = phi-gamma,  q2<0 이면 q1 = phi+gamma
  도달 조건 |l1-l2| <= r <= l1+l2  (r = sqrt(x²+y²))

검증 항목 (모두 PASS/FAIL, FAIL 이 하나라도 있으면 종료 코드 1)
  A FK, J     : 새 구현 / hand_fk / Pinocchio / MuJoCo 쌍별 최대 오차
  B det J     : det_jacobian 대 numpy det(J_pin), det(J_mj)
  C J^-1      : ||J J^-1 - I|| 와 numpy inv(J_pin) 과의 차이
  D IK 왕복   : MuJoCo tip -> 닫힌 해 -> 원래 q 복원, 두 해 모두 Pinocchio/MuJoCo FK 로 target 만족
  E IK 반복   : 임의 시작점에서 ik_newton(Pinocchio, MuJoCo) 의 수렴 해가 닫힌 해 둘 중 하나와 일치
  F 도달 범위 : 범위 밖은 빈 목록, 경계는 해 1개, 원점 근처 처리
  G 특이점    : sigma_min(J), |det J|, cond(J), J^-1 오차 표, 움직일 수 없는 방향의 수치 확인
  링크 길이가 같으면(기본값 l1=l2=1) l1 과 l2 를 바꿔 쓴 오류를 못 잡으므로,
  기본 실행은 비대칭 링크(l1=1.0, l2=0.6, 내장 생성기 모델)로 한 번 더 검증합니다.

사용법 (레포 루트에서)
  python experiments/planar-2dof/kinematics_compare_2dof.py                    # 기본 검증 (파일 모델 + 비대칭 패스)
  python experiments/planar-2dof/kinematics_compare_2dof.py --plot             # 그림 2장을 figures/ 에 저장 (안쪽 반지름이 보이도록 비대칭 링크로 그림)
  python experiments/planar-2dof/kinematics_compare_2dof.py --ik-sign-flip     # q1 의 +-gamma 부호를 일부러 뒤집어 FAIL 확인
  python experiments/planar-2dof/kinematics_compare_2dof.py --embedded         # 모델 파일 없이 내장 생성기 모델로 비교
  python experiments/planar-2dof/kinematics_compare_2dof.py --l1 1.0 --l2 0.6  # 링크 길이를 직접 지정 (비대칭 패스는 생략)

필요 패키지: numpy, pyyaml, pin(Pinocchio), mujoco  /  선택: matplotlib (--plot)
"""
import argparse
import os
import sys
from collections import namedtuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dynamics_compare_2dof as dc  # noqa: E402

np.set_printoptions(precision=6, suppress=True, linewidth=120)

DEFAULT_FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
SING_EPS = 1e-6          # |sin q2| 가 이보다 작으면 특이점 근처로 본다
CLIP_TOL = 1e-12         # |c2| 가 1 을 이만큼까지 넘으면 [-1, 1] 로 자르고, 1-|c2| 가 이 안이면 경계(해 1개)로 합친다
ORIGIN_TOL = 1e-9        # (l1+l2) 대비 이 안이면 원점(이면서 l1=l2)으로 본다
SIGMA_REGULAR = 1e-3     # D, E, C 에서 쓰는 비특이 구간: |sin q2| > 이 값

IKResult = namedtuple("IKResult", ["solutions", "status"])
# status: "two" 해 2개, "boundary" 경계(해 1개), "unreachable" 도달 불가, "infinite_origin" l1=l2 의 원점(해 무한)


class SingularJacobianError(ValueError):
    """J 가 (거의) 특이해서 역행렬을 믿을 수 없을 때 jacobian_inverse 가 던진다."""


# ============================================================
# 1. 사용자 유도식 구현
# ============================================================
def wrap_angle(a):
    """각도를 (-pi, pi] 로 정리한다.

    입력: a  스칼라 또는 임의 shape 배열 [rad]
    출력: 같은 shape, 값은 (-pi, pi]
    가정: 유한한 실수
    """
    w = np.mod(np.asarray(a, dtype=float) + np.pi, 2.0 * np.pi) - np.pi     # [-pi, pi)
    return np.where(w <= -np.pi, w + 2.0 * np.pi, w)


def fk(q, p):
    """끝단 위치 (순기구학).

    입력: q (2,) [rad] q1 은 베이스 기준 절대각, q2 는 링크1 기준 상대각 / p 링크 길이 l1, l2 [m]
    출력: tip (2,) [m] x-y 평면, 베이스 원점 기준
    가정: 평면 2링크 직렬 팔, 관절 한계 없음
    """
    q1, q12 = q[0], q[0] + q[1]
    return np.array([p.l1 * np.cos(q1) + p.l2 * np.cos(q12),
                     p.l1 * np.sin(q1) + p.l2 * np.sin(q12)])


def jacobian(q, p):
    """끝단 위치의 Jacobian J = d tip / d q.

    입력: q (2,) [rad] / p 링크 길이 [m]
    출력: J (2,2) [m/rad], 행은 (x, y), 열은 (q1, q2), x-y 평면의 월드 좌표
    가정: fk 와 같은 규약 (q2 는 상대각)
    """
    s1, c1 = np.sin(q[0]), np.cos(q[0])
    s12, c12 = np.sin(q[0] + q[1]), np.cos(q[0] + q[1])
    return np.array([[-p.l1 * s1 - p.l2 * s12, -p.l2 * s12],
                     [p.l1 * c1 + p.l2 * c12, p.l2 * c12]])


def det_jacobian(q, p):
    """det J = l1 l2 sin(q2) (닫힌 식).

    입력: q (2,) [rad] / p 링크 길이 [m]
    출력: 스칼라 [m²/rad²]. q2 = 0, ±pi 에서 0
    가정: jacobian 과 같은 규약. q1 에는 의존하지 않는다
    """
    return p.l1 * p.l2 * np.sin(q[1])


def is_near_singular(q, p, eps=SING_EPS):
    """특이점 근처인지 판정한다 (|sin q2| < eps).

    입력: q (2,) [rad] / p 링크 길이 / eps 무차원 임계값
    출력: bool
    가정: det J = l1 l2 sin q2 이므로 l1, l2 > 0 이면 |sin q2| 만 보면 충분하다
    """
    return bool(abs(np.sin(q[1])) < eps)


def jacobian_inverse(q, p, eps=SING_EPS):
    """J 의 역행렬 (수반행렬 / det).

    입력: q (2,) [rad] / p 링크 길이 [m] / eps 무차원, |sin q2| 가 이보다 작으면 예외
    출력: J^-1 (2,2) [rad/m]
    정책: 특이점 근처에서는 None 을 돌려주지 않고 SingularJacobianError 를 던진다
          (None 이 조용히 지나가 NaN 이나 큰 값으로 이어지는 것을 막기 위해).
          호출자가 미리 판단하려면 is_near_singular 를 쓴다.
    가정: 특이점이 아니어도 |sin q2| 가 작으면 오차가 1/|sin q2| 로 커진다
    """
    s2 = np.sin(q[1])
    if abs(s2) < eps:
        raise SingularJacobianError(f"|sin q2| = {abs(s2):.3e} < eps({eps:g}): J 가 특이합니다 (q2={q[1]:.6g} rad)")
    c1, s1 = np.cos(q[0]), np.sin(q[0])
    c12, s12 = np.cos(q[0] + q[1]), np.sin(q[0] + q[1])
    d = p.l1 * p.l2 * s2
    return (1.0 / d) * np.array([[p.l2 * c12, p.l2 * s12],
                                 [-p.l1 * c1 - p.l2 * c12, -p.l1 * s1 - p.l2 * s12]])


def ik_closed_form(x, y, p, gamma_sign=1.0):
    """위치 IK 의 닫힌 해 (제2 코사인 법칙).

    입력: x, y 목표 끝단 위치 [m] (베이스 원점 기준) / p 링크 길이 [m]
          gamma_sign  +1 이 정상. -1 은 q1 의 +-gamma 부호를 뒤집는 시험용 (--ik-sign-flip 이 FAIL 을 내는지 확인)
    출력: IKResult(solutions, status)
          solutions  [q (2,) [rad], ...] 각도는 (-pi, pi]. 해 2개면 [q2>0 해, q2<0 해] 순서
          status     "two" / "boundary"(q2 = 0 또는 pi, 해 1개) / "unreachable"(빈 목록) / "infinite_origin"(빈 목록)
    특수 경우:
      - |c2| 가 1 을 CLIP_TOL(1e-12) 이내로 넘으면 [-1, 1] 로 자르고, 그보다 크면 도달 불가
      - 1-|c2| <= CLIP_TOL 이면 두 해가 거의 겹치므로 해 1개(boundary)로 합친다 (q2 오차는 약 sqrt 로 커져 1e-6 rad 까지 가능)
      - l1 = l2 이고 r 이 거의 0 이면 q1 이 아무 값이나 되어 해가 무한히 많다 -> 빈 목록 + "infinite_origin"
    가정: l1, l2 > 0, 관절 한계 없음
    """
    l1, l2 = p.l1, p.l2
    r2 = x * x + y * y
    r = np.sqrt(r2)
    scale = l1 + l2
    if r <= ORIGIN_TOL * scale and abs(l1 - l2) <= ORIGIN_TOL * scale:
        return IKResult([], "infinite_origin")
    c2 = (r2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    if abs(c2) > 1.0 + CLIP_TOL:
        return IKResult([], "unreachable")
    c2 = float(np.clip(c2, -1.0, 1.0))
    phi = np.arctan2(y, x)
    cg = (r2 + l1 * l1 - l2 * l2) / (2.0 * r * l1)       # 위에서 r > 0 이 보장된다 (r = 0 이면 unreachable 또는 origin)
    gamma = np.arccos(np.clip(cg, -1.0, 1.0))
    if 1.0 - abs(c2) <= CLIP_TOL:
        q2 = 0.0 if c2 > 0 else np.pi
        return IKResult([wrap_angle(np.array([phi - gamma, q2]))], "boundary")
    q2 = np.arccos(c2)
    s = gamma_sign
    return IKResult([wrap_angle(np.array([phi - s * gamma, q2])),
                     wrap_angle(np.array([phi + s * gamma, -q2]))], "two")


def ik_newton(backend, target, q0, iters=200, tol=1e-10, damping=1e-4):
    """감쇠 최소제곱(DLS) 반복 IK:  dq = J^T (J J^T + damping² I)^-1 (target - tip).

    입력: backend  fk(q) -> (tip (2,) [m], J (2,2) [m/rad]) 를 주는 객체 (PinBackend, MjBackend)
          target (2,) [m] / q0 (2,) [rad] 시작점 / iters 최대 반복 / tol 잔차 기준 [m] / damping 무차원 아님 [m 단위 크기의 감쇠]
    출력: (q (2,) [rad] 래핑하지 않음, converged bool, n_iter int)
    가정: 해 근처가 아닌 시작점도 허용하지만 수렴을 보장하지는 않는다 (특이점 근처의 국소 정체 가능).
    """
    q = np.array(q0, dtype=float)
    lam2 = damping * damping
    eye = np.eye(2)
    for k in range(iters):
        tip, J = backend.fk(q)
        e = target - tip
        if np.linalg.norm(e) < tol:
            return q, True, k
        q = q + J.T @ np.linalg.solve(J @ J.T + lam2 * eye, e)
    tip, _ = backend.fk(q)
    return q, bool(np.linalg.norm(target - tip) < tol), iters


# ============================================================
# 2. 검증 보조
# ============================================================
class Checker:
    """OK/NG 줄을 출력하고 불합격 개수를 센다."""

    def __init__(self):
        self.fails = 0

    def le(self, name, value, limit, note=""):
        """value <= limit 이면 OK. 입력: 이름, 값, 기준(같은 단위), 덧붙일 설명. 출력: bool"""
        ok = bool(value <= limit)
        self.fails += (not ok)
        print(f"   {name:<50s} {value:10.3e}  (기준 {limit:8.1e})  {'OK' if ok else 'NG'}  {note}")
        return ok

    def flag(self, name, ok, note=""):
        """조건 ok 가 참이면 OK. 입력: 이름, 불리언, 설명. 출력: bool"""
        ok = bool(ok)
        self.fails += (not ok)
        print(f"   {name:<50s} {'':>10s}  {'':>18s}  {'OK' if ok else 'NG'}  {note}")
        return ok


def sample_q(rng, n, regular=False):
    """무작위 관절각 n 개. 입력: rng, n / regular 가 참이면 |sin q2| > SIGMA_REGULAR 만. 출력: [(2,), ...] [rad]"""
    out = []
    while len(out) < n:
        q = rng.uniform(-np.pi, np.pi, 2)
        if regular and abs(np.sin(q[1])) <= SIGMA_REGULAR:
            continue
        out.append(q)
    return out


def angle_dist(a, b):
    """두 관절각 벡터의 래핑된 최대 차이. 입력: a, b (2,) [rad]. 출력: 스칼라 [rad]"""
    return float(np.max(np.abs(wrap_angle(np.asarray(a) - np.asarray(b)))))


def best_match(sols, q):
    """해 목록 중 q 와 가장 가까운 해까지의 각도 차이. 입력: 해 목록, q (2,). 출력: 스칼라 [rad], 해가 없으면 inf"""
    return min((angle_dist(s, q) for s in sols), default=np.inf)


# ============================================================
# 3. 검증 A~G
# ============================================================
def check_A(ck, p, pb, mb, qs, tol):
    """A. FK, J: 새 구현 / hand_fk / Pinocchio / MuJoCo 의 쌍별 최대 오차. 새 구현이 낀 쌍만 판정, 나머지는 참고."""
    names = ["new", "hand", "pin", "mj"]
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    worst = {(pr, k): 0.0 for pr in pairs for k in ("tip", "J")}
    for q in qs:
        tips, Js = {}, {}
        tips["new"], Js["new"] = fk(q, p), jacobian(q, p)
        tips["hand"], Js["hand"] = dc.hand_fk(q, p)
        tips["pin"], Js["pin"] = pb.fk(q)
        tips["mj"], Js["mj"] = mb.fk(q)
        for pr in pairs:
            worst[(pr, "tip")] = max(worst[(pr, "tip")], dc.maxerr(tips[pr[0]], tips[pr[1]]))
            worst[(pr, "J")] = max(worst[(pr, "J")], dc.maxerr(Js[pr[0]], Js[pr[1]]))
    print(f"\n[A] FK, Jacobian: 방법 쌍별 최대 절대 오차 (무작위 {len(qs)}개, 기준 {tol:g}, 새 구현이 낀 쌍만 판정)")
    print(f"   {'쌍':<12s}{'tip':>14s}{'J':>14s}")
    for pr in pairs:
        judged = "new" in pr
        row = f"   {pr[0] + ' - ' + pr[1]:<12s}"
        for k in ("tip", "J"):
            e = worst[(pr, k)]
            ok = e <= tol
            if judged:
                ck.fails += (not ok)
            row += f"{e:>10.2e} {'OK' if ok else 'NG'}" if judged else f"{e:>10.2e} (참고)"
        print(row)


def check_B(ck, p, pb, mb, qs, tol):
    """B. det J: det_jacobian 대 numpy det(J_pin), det(J_mj) (절대 오차)."""
    w_pin = w_mj = w_own = 0.0
    for q in qs:
        d = det_jacobian(q, p)
        w_pin = max(w_pin, abs(d - np.linalg.det(pb.fk(q)[1])))
        w_mj = max(w_mj, abs(d - np.linalg.det(mb.fk(q)[1])))
        w_own = max(w_own, abs(d - np.linalg.det(jacobian(q, p))))
    print(f"\n[B] det J = l1 l2 sin q2 (무작위 {len(qs)}개, 절대 오차)")
    ck.le("det_jacobian - numpy det(J_new)", w_own, tol)
    ck.le("det_jacobian - numpy det(J_pin)", w_pin, tol)
    ck.le("det_jacobian - numpy det(J_mj)", w_mj, tol)


def check_C(ck, p, pb, qs, tol):
    """C. J^-1: ||J J^-1 - I||max 와 numpy inv(J_pin) 과의 차이. 기준은 tol * max(1, cond(J)) (cond 가 크면 오차도 커지는 것이 정상)."""
    w_res = w_pin = 0.0          # 기준으로 나눈 최악 비율 (<= 1 이면 합격)
    raw_res = raw_pin = 0.0
    for q in qs:
        J = jacobian(q, p)
        lim = tol * max(1.0, np.linalg.cond(J))
        Ji = jacobian_inverse(q, p)
        r1 = float(np.max(np.abs(J @ Ji - np.eye(2))))
        r2 = dc.maxerr(Ji, np.linalg.inv(pb.fk(q)[1]))
        raw_res, raw_pin = max(raw_res, r1), max(raw_pin, r2)
        w_res, w_pin = max(w_res, r1 / lim), max(w_pin, r2 / lim)
    print(f"\n[C] J^-1 ({len(qs)}개, |sin q2| > {SIGMA_REGULAR:g}, 기준 = {tol:g} * max(1, cond(J)))")
    ck.le("max ||J J^-1 - I||  (기준 대비 비율)", w_res, 1.0, f"절대 최대 {raw_res:.2e}")
    ck.le("max |J^-1 - inv(J_pin)|  (기준 대비 비율)", w_pin, 1.0, f"절대 최대 {raw_pin:.2e}")


def check_D(ck, p, pb, mb, qs, tol, gs):
    """D. IK 왕복: target = MuJoCo tip -> ik_closed_form. 원래 q 복원(1e-9)과 두 해 모두의 FK 잔차(Pinocchio, MuJoCo)."""
    w_ang = w_pin = w_mj = 0.0
    bad_status = 0
    for q in qs:
        target = mb.fk(q)[0]
        res = ik_closed_form(target[0], target[1], p, gs)
        if res.status != "two":
            bad_status += 1
        w_ang = max(w_ang, best_match(res.solutions, q))
        for s in res.solutions:
            w_pin = max(w_pin, dc.maxerr(pb.fk(s)[0], target))
            w_mj = max(w_mj, dc.maxerr(mb.fk(s)[0], target))
    print(f"\n[D] IK 왕복: MuJoCo tip -> 닫힌 해 ({len(qs)}개, |sin q2| > {SIGMA_REGULAR:g}, q2 부호 양쪽 포함)")
    ck.flag("해가 항상 2개 (status = two)", bad_status == 0, f"예외 {bad_status}개")
    ck.le("원래 q 와 가장 가까운 해의 각도 차이 [rad]", w_ang, 1e-9)
    ck.le("두 해의 FK 잔차 (Pinocchio) [m]", w_pin, tol)
    ck.le("두 해의 FK 잔차 (MuJoCo) [m]", w_mj, tol)


def check_E(ck, p, pb, mb, rng, n, gs):
    """E. 반복 IK(DLS) vs 닫힌 해: 해 근처가 아닌 임의 시작점. 수렴(잔차 < 1e-10)한 해는 닫힌 해 둘 중 하나와 1e-6 이내여야 한다. 수렴률은 정보."""
    qs = sample_q(rng, n, regular=True)
    q0s = sample_q(rng, n)
    print(f"\n[E] 반복 IK(DLS) vs 닫힌 해 ({n}개, 임의 시작점, 수렴 기준 잔차 < 1e-10, 일치 기준 1e-6 rad)")
    for label, be in (("Pinocchio", pb), ("MuJoCo", mb)):
        conv = 0
        worst = 0.0
        mismatch = 0
        iters_sum = 0
        for q, q0 in zip(qs, q0s):
            target = mb.fk(q)[0]
            qn, ok, k = ik_newton(be, target, q0)
            if not ok:
                continue
            conv += 1
            iters_sum += k
            sols = ik_closed_form(target[0], target[1], p, gs).solutions
            e = best_match(sols, qn)
            worst = max(worst, e if np.isfinite(e) else 1e9)
            mismatch += (not e <= 1e-6)
        mean_it = iters_sum / conv if conv else float("nan")
        ck.le(f"{label}: 수렴한 해와 닫힌 해의 최대 각도 차이 [rad]", worst, 1e-6,
              f"수렴 {conv}/{n} ({100.0 * conv / n:.1f}%, 평균 {mean_it:.1f}회), 불일치 {mismatch}개")


def check_F(ck, p, rng, gs):
    """F. 도달 범위: 범위 밖은 빈 목록, 경계는 해 1개(FK 로 target 만족), 원점 근처 처리, 범위 안은 해 2개. 모델 파일 없이 수식만 쓴다."""
    l1, l2 = p.l1, p.l2
    r_out, r_in = l1 + l2, abs(l1 - l2)
    th = 0.7
    u = np.array([np.cos(th), np.sin(th)])
    print(f"\n[F] 도달 범위 (l1={l1:g}, l2={l2:g}: 바깥 반지름 {r_out:g}, 안쪽 반지름 {r_in:g})")

    res = ik_closed_form(*(u * (r_out + 1e-6)), p, gs)
    ck.flag("바깥 경계 + 1e-6: 빈 목록", res.solutions == [] and res.status == "unreachable", f"status={res.status}")
    if r_in > 1e-6:
        res = ik_closed_form(*(u * (r_in - 1e-6)), p, gs)
        ck.flag("안쪽 경계 - 1e-6: 빈 목록", res.solutions == [] and res.status == "unreachable", f"status={res.status}")

    res = ik_closed_form(r_out * u[0], r_out * u[1], p, gs)
    ok = res.status == "boundary" and len(res.solutions) == 1 and abs(res.solutions[0][1]) < 1e-12
    err = dc.maxerr(fk(res.solutions[0], p), r_out * u) if res.solutions else np.inf
    ck.flag("바깥 경계 r = l1+l2: 해 1개, q2 = 0", ok and err <= 1e-9, f"status={res.status}, FK 잔차 {err:.1e}")
    res = ik_closed_form(r_out, 0.0, p, gs)
    ck.flag("바깥 경계 (x축 위): 해 1개", res.status == "boundary" and len(res.solutions) == 1, f"status={res.status}")

    if r_in > 1e-6:
        tgt = r_in * u
        res = ik_closed_form(tgt[0], tgt[1], p, gs)
        ok = res.status == "boundary" and len(res.solutions) == 1 and abs(abs(res.solutions[0][1]) - np.pi) < 1e-12
        err = dc.maxerr(fk(res.solutions[0], p), tgt) if res.solutions else np.inf
        ck.flag("안쪽 경계 r = |l1-l2|: 해 1개, q2 = pi", ok and err <= 1e-9, f"status={res.status}, FK 잔차 {err:.1e}")

    res = ik_closed_form(0.0, 0.0, p, gs)
    if abs(l1 - l2) <= ORIGIN_TOL * (l1 + l2):
        ck.flag("원점 (l1 = l2): 빈 목록 + infinite_origin", res.solutions == [] and res.status == "infinite_origin", f"status={res.status}")
    else:
        ck.flag("원점 (l1 != l2): 빈 목록 + unreachable", res.solutions == [] and res.status == "unreachable", f"status={res.status}")

    worst = 0.0
    bad = 0
    n = 200
    for _ in range(n):
        r = rng.uniform(r_in + 1e-3, r_out - 1e-3)
        a = rng.uniform(-np.pi, np.pi)
        tgt = r * np.array([np.cos(a), np.sin(a)])
        res = ik_closed_form(tgt[0], tgt[1], p, gs)
        bad += (res.status != "two")
        for s in res.solutions:
            worst = max(worst, dc.maxerr(fk(s, p), tgt))
    ck.flag(f"범위 안 {n}점: 항상 해 2개", bad == 0, f"예외 {bad}개")
    ck.le("범위 안 {}점: 두 해의 FK 잔차 [m]".format(n), worst, 1e-9)

    bad = 0
    for _ in range(n):
        r = r_out + rng.uniform(1e-3, 2.0)
        a = rng.uniform(-np.pi, np.pi)
        bad += (ik_closed_form(r * np.cos(a), r * np.sin(a), p, gs).solutions != [])
    ck.flag(f"바깥 범위 밖 {n}점: 항상 빈 목록", bad == 0, f"예외 {bad}개")


def check_G(ck, p, pb, mb, rng, n, tol):
    """G. 특이점: q2 = 1e-1..1e-4, 0, pi 근처의 sigma_min(J), |det J|, cond(J), J^-1 오차 표와 움직일 수 없는 방향 확인."""
    q1 = 0.3
    d = [1e-1, 1e-2, 1e-3, 1e-4]
    q2s = d + [0.0] + [np.pi - x for x in reversed(d)] + [np.pi]
    print(f"\n[G] 특이점 근처 (q1 = {q1} rad 고정; sigma_min, det, cond 는 q1 에 의존하지 않음)")
    print(f"   {'q2 [rad]':>12s} {'sigma_min':>10s} {'|det J|':>10s} {'cond(J)':>10s} {'|dJ-pin|':>9s} {'|dJ-mj|':>9s} {'|J Jinv-I|':>11s}")
    worst_det = 0.0
    for q2 in q2s:
        q = np.array([q1, q2])
        J = jacobian(q, p)
        sv = np.linalg.svd(J, compute_uv=False)
        det = det_jacobian(q, p)
        e_pin = abs(det - np.linalg.det(pb.fk(q)[1]))
        e_mj = abs(det - np.linalg.det(mb.fk(q)[1]))
        worst_det = max(worst_det, e_pin, e_mj)
        cond = sv[0] / sv[1] if sv[1] > 0 else float("inf")
        try:
            inv_err = f"{float(np.max(np.abs(J @ jacobian_inverse(q, p) - np.eye(2)))):11.2e}"
        except SingularJacobianError:
            inv_err = f"{'예외(특이)':>10s}"
        print(f"   {q2:12.6f} {sv[1]:10.2e} {abs(det):10.2e} {cond:10.2e} {e_pin:9.1e} {e_mj:9.1e} {inv_err}")
    ck.le("det_jacobian 대 Pinocchio, MuJoCo det (위 표 전체)", worst_det, tol)

    ck.flag("is_near_singular: q2 = 0, pi 에서 참", is_near_singular(np.array([q1, 0.0]), p) and is_near_singular(np.array([q1, np.pi]), p))
    ck.flag("is_near_singular: q2 = 1e-2 에서 거짓", not is_near_singular(np.array([q1, 1e-2]), p))
    raised = []
    for q2 in (0.0, np.pi):
        try:
            jacobian_inverse(np.array([q1, q2]), p)
            raised.append(False)
        except SingularJacobianError:
            raised.append(True)
    ck.flag("jacobian_inverse: q2 = 0, pi 에서 SingularJacobianError", all(raised))

    # 움직일 수 없는 방향: q2 = 0, pi 에서 J dq 는 팔 방향 u = (cos q1, sin q1) 에 수직이어야 한다
    print(f"\n   움직일 수 없는 방향: 임의의 dq {n}개에 대해 tip 속도 J dq 의 팔 방향 성분 (수직 성분은 참고용)")
    for label, q2 in (("q2 = 0 (펴짐)", 0.0), ("q2 = pi (접힘)", np.pi)):
        q = np.array([q1, q2])
        u = np.array([np.cos(q1), np.sin(q1)])
        up = np.array([-np.sin(q1), np.cos(q1)])
        w_dot = {"new": 0.0, "pin": 0.0, "mj": 0.0}
        w_perp = 0.0
        Js = {"new": jacobian(q, p), "pin": pb.fk(q)[1], "mj": mb.fk(q)[1]}
        for _ in range(n):
            dq = rng.uniform(-1.0, 1.0, 2)
            for k, J in Js.items():
                w_dot[k] = max(w_dot[k], abs(float(u @ (J @ dq))))
            w_perp = max(w_perp, abs(float(up @ (Js["new"] @ dq))))
        for k in ("new", "pin", "mj"):
            ck.le(f"{label}: |u . (J dq)| ({k})", w_dot[k], tol)
        print(f"      {label}: 수직 성분 |u_perp . (J dq)| 최대 {w_perp:.3f} (0 이 아니므로 수직 방향으로는 움직임)")


# ============================================================
# 4. 한 모델에 대한 전체 검증
# ============================================================
def run_pass(label, p, urdf, mjcf, args, gs):
    """한 모델(파라미터 p, 파일 또는 내장 생성기)에 대해 A~G 를 실행한다.

    입력: label 출력 제목 / p 파라미터 / urdf, mjcf 모델 파일 경로 또는 None(내장 생성기) / args 명령줄 인자 / gs gamma_sign
    출력: 불합격 항목 수. 백엔드를 만들 수 없으면 -1
    가정: Pinocchio 와 MuJoCo 가 모두 설치되어 있다 (교차 검증이 목적이라 하나라도 없으면 중단)
    """
    print("\n" + "=" * 78)
    print(f"{label}")
    print("모델 출처: Pinocchio <- {} / MuJoCo <- {}".format(urdf or "내장 생성기", mjcf or "내장 생성기"))
    print(f"링크 길이: l1={p.l1:g}, l2={p.l2:g} [m]   n={args.n_random}, tol={args.tol:g}, seed={args.seed}"
          + ("   [주의] gamma 부호를 뒤집었습니다. FAIL 이 나오는 것이 정상입니다." if gs != 1.0 else ""))
    print("=" * 78)
    try:
        pb = dc.PinBackend(p, urdf)
        mb = dc.MjBackend(p, mjcf)
    except Exception as ex:  # noqa
        print(f"[오류] Pinocchio 또는 MuJoCo 를 사용할 수 없습니다: {ex}")
        return -1, None, None
    rng = np.random.default_rng(args.seed)
    ck = Checker()
    qs = sample_q(rng, args.n_random)
    qs_reg = sample_q(rng, args.n_random, regular=True)
    check_A(ck, p, pb, mb, qs, args.tol)
    check_B(ck, p, pb, mb, qs, args.tol)
    check_C(ck, p, pb, qs_reg, args.tol)
    check_D(ck, p, pb, mb, qs_reg, args.tol, gs)
    check_E(ck, p, pb, mb, rng, args.n_random, gs)
    check_F(ck, p, rng, gs)
    check_G(ck, p, pb, mb, rng, min(args.n_random, 200), args.tol)
    print("\n" + (f"[{label}] 결과: 모두 합격" if ck.fails == 0 else f"[{label}] 결과: 불합격 항목 {ck.fails}개"))
    return ck.fails, pb, mb


# ============================================================
# 5. 그림
# ============================================================
def make_plots(p, mb, out_dir, gs):
    """그림 2장을 out_dir 에 저장한다 (글자는 어느 PC 에서도 같게 나오도록 영어).

    입력: p 파라미터 / mb MjBackend (그림 2 의 교차 점 표시용, 없으면 생략) / out_dir 저장 폴더 / gs gamma_sign
    출력: 저장한 파일 경로 목록
    가정: matplotlib 이 설치되어 있다
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    os.makedirs(out_dir, exist_ok=True)
    paths = []

    # ---- 그림 1: 한 목표점에 대한 두 해
    q_true = np.array([0.6, 1.1])
    target = fk(q_true, p)
    res = ik_closed_form(target[0], target[1], p, gs)
    r_out, r_in = p.l1 + p.l2, abs(p.l1 - p.l2)
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.add_patch(Circle((0, 0), r_out, facecolor="#e9f0f7", edgecolor="#555555", lw=1.0, zorder=0))
    if r_in > 0:
        ax.add_patch(Circle((0, 0), r_in, facecolor="white", edgecolor="#555555", lw=1.0, ls="--", zorder=1))
    ax.text(0, -r_out - 0.12, f"workspace: |l1-l2| = {r_in:g} <= r <= l1+l2 = {r_out:g}", ha="center", va="top", fontsize=9)
    for sol, color, tag in zip(res.solutions, ("#1f77b4", "#d95f02"), ("q2 > 0", "q2 < 0")):
        j1 = np.array([p.l1 * np.cos(sol[0]), p.l1 * np.sin(sol[0])])
        tip = fk(sol, p)
        ax.plot([0, j1[0], tip[0]], [0, j1[1], tip[1]], "-o", color=color, lw=3, ms=7, zorder=3,
                label=f"{tag}: q1={np.rad2deg(sol[0]):.1f} deg, q2={np.rad2deg(sol[1]):.1f} deg")
    ax.plot(*target, marker="*", color="crimson", ms=16, ls="none", zorder=4, label=f"target ({target[0]:.2f}, {target[1]:.2f}) m")
    ax.plot(0, 0, "ks", ms=8, zorder=5)
    ax.set_aspect("equal")
    lim = r_out * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Two closed-form IK solutions for one target (l1={p.l1:g}, l2={p.l2:g})")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f1 = os.path.join(out_dir, "ik_two_solutions.png")
    fig.savefig(f1, dpi=130)
    plt.close(fig)
    paths.append(f1)

    # ---- 그림 2: q2 에 따른 sigma_min(J), |det J|
    q2s = np.linspace(-np.pi, np.pi, 721)
    smin = np.array([np.linalg.svd(jacobian(np.array([0.3, a]), p), compute_uv=False)[1] for a in q2s])
    adet = np.abs([det_jacobian(np.array([0.3, a]), p) for a in q2s])
    q2m = np.linspace(-np.pi, np.pi, 25)
    fig, axs = plt.subplots(2, 1, figsize=(7.0, 6.0), sharex=True)
    series = [(axs[0], smin, "sigma_min(J) [m/rad]", lambda J: np.linalg.svd(J, compute_uv=False)[1]),
              (axs[1], adet, "|det J| = l1 l2 |sin q2| [m^2/rad^2]", lambda J: abs(np.linalg.det(J)))]
    for ax, y, ylabel, fn in series:
        ax.plot(q2s, y, color="#1f77b4", lw=2, label="this implementation")
        if mb is not None:
            ax.plot(q2m, [fn(mb.fk(np.array([0.3, a]))[1]) for a in q2m], "o", color="#9467bd", ms=5, mfc="none", label="MuJoCo J (numpy)")
        for v in (-np.pi, 0.0, np.pi):
            ax.axvline(v, color="crimson", ls=":", lw=1)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper center")
    axs[1].set_xlabel("q2 [rad]  (dotted lines: singular, q2 = 0 stretched, q2 = +-pi folded)")
    axs[1].set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    axs[1].set_xticklabels(["-pi", "-pi/2", "0", "pi/2", "pi"])
    axs[0].set_title(f"Singularity of the planar 2DOF arm (l1={p.l1:g}, l2={p.l2:g})")
    fig.tight_layout()
    f2 = os.path.join(out_dir, "singularity_sigma_det.png")
    fig.savefig(f2, dpi=130)
    plt.close(fig)
    paths.append(f2)
    return paths


# ============================================================
# 6. 메인
# ============================================================
def resolve_main_model(args):
    """명령줄 인자에서 첫 번째 패스의 (p, urdf, mjcf) 를 정한다. dynamics_compare_2dof.py 와 같은 규칙.

    입력: args (--models-dir, --embedded, --l1, --l2)
    출력: (p, urdf 경로 또는 None, mjcf 경로 또는 None)
    가정: params.yaml 이 있으면 단일 출처. 링크 길이를 덮어쓰면 파일 대신 내장 생성기를 쓴다
    """
    yaml_path = os.path.join(args.models_dir, dc.YAML_NAME)
    urdf = os.path.join(args.models_dir, dc.URDF_NAME)
    mjcf = os.path.join(args.models_dir, dc.MJCF_NAME)
    base = None
    if not args.embedded and os.path.isfile(yaml_path):
        try:
            base = dc.load_params_yaml(yaml_path)
        except ImportError:
            print("[경고] pyyaml 이 없어 params.yaml 을 읽지 못했습니다 (pip install pyyaml). 내장 값으로 진행합니다.")
    p = base if base is not None else dc.Params()
    use_files = base is not None and os.path.isfile(urdf) and os.path.isfile(mjcf)
    if base is not None and not use_files:
        print(f"[알림] {args.models_dir} 에 URDF/MJCF 파일이 없어 내장 생성기를 사용합니다.")
    if args.l1 is not None or args.l2 is not None:
        p = dc.Params(p.m1, p.m2, p.l1 if args.l1 is None else args.l1, p.l2 if args.l2 is None else args.l2, g=p.g)
        print("[알림] 링크 길이를 명령줄로 바꿨으므로 모델 파일 대신 내장 생성기로 만든 모델을 사용합니다.")
        use_files = False
    return p, (urdf if use_files else None), (mjcf if use_files else None)


def main():
    """명령줄 진입점. 출력: 종료 코드 (0 합격, 1 불합격 항목 있음, 2 설정 오류)."""
    ap = argparse.ArgumentParser(description="2DOF 기구학 비교: 사용자 유도식 / hand_fk / Pinocchio / MuJoCo")
    ap.add_argument("--n-random", type=int, default=1000, help="무작위 상태 개수 (기본 1000)")
    ap.add_argument("--tol", type=float, default=1e-9, help="허용 오차 (기본 1e-9)")
    ap.add_argument("--seed", type=int, default=0, help="난수 시드 (기본 0)")
    ap.add_argument("--models-dir", default=dc.DEFAULT_MODELS_DIR, help="params.yaml, URDF, MJCF 폴더 (기본: ../../models/planar-2dof)")
    ap.add_argument("--embedded", action="store_true", help="모델 파일을 읽지 않고 내장 생성기 모델을 사용")
    ap.add_argument("--l1", type=float, default=None, help="링크1 길이 [m] 덮어쓰기 (내장 생성기 모델, 비대칭 패스 생략)")
    ap.add_argument("--l2", type=float, default=None, help="링크2 길이 [m] 덮어쓰기 (내장 생성기 모델, 비대칭 패스 생략)")
    ap.add_argument("--plot", nargs="?", const=DEFAULT_FIG_DIR, default=None, metavar="DIR", help="그림 2장을 DIR 에 저장 (기본: figures/)")
    ap.add_argument("--ik-sign-flip", action="store_true", help="IK 의 q1 = phi -+ gamma 부호를 뒤집어 FAIL 이 나는지 확인")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

    try:
        p, urdf, mjcf = resolve_main_model(args)
    except Exception as ex:  # noqa
        print(f"[오류] params.yaml 을 읽지 못했습니다: {ex}")
        return 2
    gs = -1.0 if args.ik_sign_flip else 1.0

    fails, pb, mb = run_pass("패스 1: 기본 모델", p, urdf, mjcf, args, gs)
    if fails < 0:
        return 2
    total = fails
    p_plot, mb_plot = p, mb                      # 그림은 안쪽 반지름이 보이도록 가능하면 비대칭 링크로 그린다
    if args.l1 is None and args.l2 is None:
        p_asym = dc.Params(l1=1.0, l2=0.6)
        f2, _, mb2 = run_pass("패스 2: 비대칭 링크 (l1 != l2 라서 l1, l2 를 바꿔 쓴 오류도 잡힘)", p_asym, None, None, args, gs)
        if f2 < 0:
            return 2
        total += f2
        p_plot, mb_plot = p_asym, mb2

    if args.plot is not None:
        try:
            paths = make_plots(p_plot, mb_plot, args.plot, gs)
            print("\n그림 저장: " + ", ".join(paths))
        except ImportError as ex:
            print(f"\n[경고] 그림을 만들지 못했습니다 (pip install matplotlib): {ex}")

    print("\n" + "=" * 78)
    print("전체 결과: 모두 합격" if total == 0 else f"전체 결과: 불합격 항목 {total}개")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
