#!/usr/bin/env python3
"""
2DOF 평면 팔 동역학: 네 가지 방법 비교 (파일 하나로 실행)

  hand  : 손으로 유도한 오일러-라그랑주 최종식
  NE    : 뉴턴-오일러 재귀 (같은 시스템을 힘과 모멘트 평형으로 다시 계산)
  pin   : Pinocchio (crba, nonLinearEffects, rnea, aba, computeCoriolisMatrix)
  mj    : MuJoCo의 계산 함수 (mj_fullM, qfrc_bias, mj_inverse, mj_forward)
          (시각화는 --view 일 때만 사용. 계산에는 필요 없음)

시스템 (손 유도와 동일)
  - 수직 평면(x-y), 중력 -y 방향, 관절 각도 q1(절대), q2(상대)
  - 링크 i: 길이 l_i, 질량 m_i, 무게중심은 링크 중간 (l_i/2), 관성 I_i (무게중심 기준, z축)
  - 기본값: m = l = 1, I = m l^2 / 12 (균일한 막대)

모델 파일 (단일 출처: ../../models/planar-2dof/)
  params.yaml 의 값을 읽어 hand/NE 를 계산하고, planar_2dof.urdf (Pinocchio) 와
  planar_2dof.xml (MuJoCo) 파일을 실제로 읽어 비교합니다. 파일이 params.yaml 과 어긋나면 FAIL 이 납니다.
  --m1 등으로 값을 직접 바꾸면 파일 대신 내장 생성기로 만든 모델을 씁니다.

사용법
  python dynamics_compare_2dof.py                  # 기본 비교 (hand, NE, Pinocchio, MuJoCo)
  python dynamics_compare_2dof.py --sympy          # 손으로 쓴 L 에서 sympy로 재유도해 비교
  python dynamics_compare_2dof.py --lyapunov       # PD + 중력 보상의 V(t) 수치 확인
  python dynamics_compare_2dof.py --i1-coeff 0.5   # 예전 오류(I1 계수 1/2)를 일부러 넣어 FAIL 확인
  python dynamics_compare_2dof.py --write-models ../../models/planar-2dof   # params.yaml 로 URDF, MJCF 를 다시 생성
  python dynamics_compare_2dof.py --embedded       # 파일을 쓰지 않고 내장 생성기 모델로 비교
  python dynamics_compare_2dof.py --view           # MuJoCo 뷰어로 보기 (데스크탑 전용, 미시험)

필요 패키지: numpy, pin(Pinocchio), mujoco, pyyaml  /  선택: sympy (--sympy)
이 파일은 Python 검증용입니다. Eigen은 C++ 구현 단계(P1-2)에서 같은 식을 옮길 때 사용합니다.
"""
import argparse
import os
import sys

import numpy as np

np.set_printoptions(precision=6, suppress=True, linewidth=120)


# ============================================================
# 0. 파라미터
# ============================================================
class Params:
    def __init__(self, m1=1.0, m2=1.0, l1=1.0, l2=1.0, I1=None, I2=None, g=9.81, i1c=1.0):
        self.m1, self.m2, self.l1, self.l2, self.g = m1, m2, l1, l2, g
        self.I1 = m1 * l1 ** 2 / 12.0 if I1 is None else I1
        self.I2 = m2 * l2 ** 2 / 12.0 if I2 is None else I2
        self.lc1, self.lc2 = l1 / 2.0, l2 / 2.0      # 무게중심은 링크 중간 (손 유도의 가정)
        self.i1c = i1c                               # 손 유도 M11 의 I1 계수 (정답 1.0)


# ============================================================
# 0-1. params.yaml 읽기 / 모델 폴더 찾기
# ============================================================
DEFAULT_MODELS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "planar-2dof"))
URDF_NAME, MJCF_NAME, YAML_NAME = "planar_2dof.urdf", "planar_2dof.xml", "params.yaml"


def load_params_yaml(path):
    """params.yaml -> Params. 손 유도는 무게중심이 링크 중간(com = length/2)이라고 가정하므로 어긋나면 중단."""
    import yaml
    with open(path, encoding="utf-8") as fh:
        y = yaml.safe_load(fh)
    l1, l2 = y["link1"], y["link2"]
    for name, link in (("link1", l1), ("link2", l2)):
        if abs(link["com"] - link["length"] / 2.0) > 1e-12:
            raise ValueError(f"{name}: com({link['com']}) 이 length/2 가 아닙니다. 손 유도 식은 무게중심이 링크 중간인 경우만 다룹니다.")
    return Params(l1["mass"], l2["mass"], l1["length"], l2["length"], l1["inertia"], l2["inertia"], y["gravity"])


# ============================================================
# 1. 손으로 유도한 오일러-라그랑주 최종식 (유도 결과 그대로)
# ============================================================
def hand_M(q, p):
    c2 = np.cos(q[1])
    M11 = (p.m1 / 4 + p.m2) * p.l1 ** 2 + p.m2 * p.l2 ** 2 / 4 + p.m2 * p.l1 * p.l2 * c2 + p.i1c * p.I1 + p.I2
    M12 = p.m2 * p.l2 ** 2 / 4 + p.m2 * p.l1 * p.l2 * c2 / 2 + p.I2
    M21 = M12
    M22 = p.m2 * p.l2 ** 2 / 4 + p.I2
    return np.array([[M11, M12], [M21, M22]])


def hand_C1C2(q, v, p):
    """손 유도의 C1, C2 식 (C q̇ 의 성분)"""
    s2 = np.sin(q[1])
    k = p.m2 * p.l1 * p.l2 * s2
    C1 = -k * v[0] * v[1] - 0.5 * k * v[1] ** 2
    C2 = 0.5 * k * v[0] ** 2
    return np.array([C1, C2])


def hand_C(q, v, p):
    """크리스토펠 기호로 만든 C 행렬"""
    k = 0.5 * p.m2 * p.l1 * p.l2 * np.sin(q[1])
    return np.array([[-k * v[1], -k * (v[0] + v[1])],
                     [k * v[0], 0.0]])


def hand_G(q, p):
    G1 = (p.m1 / 2 + p.m2) * p.l1 * p.g * np.cos(q[0]) + 0.5 * p.m2 * p.l2 * p.g * np.cos(q[0] + q[1])
    G2 = 0.5 * p.m2 * p.l2 * p.g * np.cos(q[0] + q[1])
    return np.array([G1, G2])


def hand_fk(q, p):
    s1, c1 = np.sin(q[0]), np.cos(q[0])
    s12, c12 = np.sin(q[0] + q[1]), np.cos(q[0] + q[1])
    tip = np.array([p.l1 * c1 + p.l2 * c12, p.l1 * s1 + p.l2 * s12])
    J = np.array([[-p.l1 * s1 - p.l2 * s12, -p.l2 * s12],
                  [p.l1 * c1 + p.l2 * c12, p.l2 * c12]])
    return tip, J


# ============================================================
# 2. 뉴턴-오일러 재귀 (평면, 스칼라 외적)
# ============================================================
def _e(th):
    return np.array([np.cos(th), np.sin(th)])


def _ep(th):
    return np.array([-np.sin(th), np.cos(th)])


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def newton_euler(q, qd, qdd, p, gravity=None):
    """역동역학 tau = NE(q, qd, qdd). 반환: (tau, f1, f2) (f는 관절 반력)"""
    gr = p.g if gravity is None else gravity
    th1, th2 = q[0], q[0] + q[1]
    w1, w2 = qd[0], qd[0] + qd[1]
    a1, a2 = qdd[0], qdd[0] + qdd[1]
    gv = np.array([0.0, -gr])
    # 순방향: 무게중심 가속도
    ac1 = p.lc1 * (a1 * _ep(th1) - w1 ** 2 * _e(th1))
    aj2 = p.l1 * (a1 * _ep(th1) - w1 ** 2 * _e(th1))
    ac2 = aj2 + p.lc2 * (a2 * _ep(th2) - w2 ** 2 * _e(th2))
    # 역방향: 링크 2 (끝단 자유)
    f2 = p.m2 * (ac2 - gv)
    tau2 = p.I2 * a2 + p.lc2 * _cross(_e(th2), f2)
    # 링크 1
    f1 = p.m1 * (ac1 - gv) + f2
    tau1 = tau2 + p.I1 * a1 + p.lc1 * _cross(_e(th1), f1) + (p.l1 - p.lc1) * _cross(_e(th1), f2)
    return np.array([tau1, tau2]), f1, f2


# ============================================================
# 3. 모델 파일 문자열 (URDF, MJCF) : 파일 없이 이 스크립트 하나로 동작
# ============================================================
def _f(x):
    return f"{x:.17g}"


def urdf_string(p):
    def link(name, m, lc, I):
        return (f'  <link name="{name}"><inertial><origin xyz="{_f(lc)} 0 0" rpy="0 0 0"/>'
                f'<mass value="{_f(m)}"/><inertia ixx="{_f(I)}" ixy="0" ixz="0" iyy="{_f(I)}" iyz="0" izz="{_f(I)}"/>'
                f'</inertial></link>')
    return f'''<?xml version="1.0"?>
<!-- 자동 생성 파일: models/planar-2dof/params.yaml 에서 만들어집니다. 직접 고치지 말고 params.yaml 을 고친 뒤 다시 생성하세요. -->
<robot name="planar_2dof">
  <link name="base"/>
{link("link1", p.m1, p.lc1, p.I1)}
{link("link2", p.m2, p.lc2, p.I2)}
  <link name="tip"/>
  <joint name="joint1" type="revolute"><parent link="base"/><child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-6.2832" upper="6.2832" effort="1000" velocity="100"/></joint>
  <joint name="joint2" type="revolute"><parent link="link1"/><child link="link2"/>
    <origin xyz="{_f(p.l1)} 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-6.2832" upper="6.2832" effort="1000" velocity="100"/></joint>
  <joint name="tip_joint" type="fixed"><parent link="link2"/><child link="tip"/>
    <origin xyz="{_f(p.l2)} 0 0" rpy="0 0 0"/></joint>
</robot>
'''


def mjcf_string(p):
    # MuJoCo 는 관성이 양수여야 하므로 I=0 이면 아주 작은 값으로 대체
    I1, I2 = max(p.I1, 1e-9), max(p.I2, 1e-9)
    return f'''<!-- 자동 생성 파일: models/planar-2dof/params.yaml 에서 만들어집니다. 직접 고치지 말고 params.yaml 을 고친 뒤 다시 생성하세요. -->
<mujoco model="planar_2dof">
  <compiler angle="radian" inertiafromgeom="false"/>
  <option gravity="0 {_f(-p.g)} 0" timestep="0.001"/>
  <worldbody>
    <body name="link1" pos="0 0 0">
      <joint name="q1" type="hinge" axis="0 0 1"/>
      <inertial pos="{_f(p.lc1)} 0 0" mass="{_f(p.m1)}" diaginertia="{_f(I1)} {_f(I1)} {_f(I1)}"/>
      <geom type="capsule" fromto="0 0 0 {_f(p.l1)} 0 0" size="0.03" contype="0" conaffinity="0" rgba="0.2 0.5 0.9 1"/>
      <body name="link2" pos="{_f(p.l1)} 0 0">
        <joint name="q2" type="hinge" axis="0 0 1"/>
        <inertial pos="{_f(p.lc2)} 0 0" mass="{_f(p.m2)}" diaginertia="{_f(I2)} {_f(I2)} {_f(I2)}"/>
        <geom type="capsule" fromto="0 0 0 {_f(p.l2)} 0 0" size="0.03" contype="0" conaffinity="0" rgba="0.9 0.5 0.2 1"/>
        <site name="tip" pos="{_f(p.l2)} 0 0" size="0.03"/>
      </body>
    </body>
  </worldbody>
</mujoco>
'''


# ============================================================
# 4. 라이브러리 래퍼
# ============================================================
class PinBackend:
    def __init__(self, p, urdf_path=None):
        import pinocchio as pin
        self.pin = pin
        if urdf_path:
            self.model = pin.buildModelFromUrdf(urdf_path)       # 실제 URDF 파일
        else:
            self.model = pin.buildModelFromXML(urdf_string(p))   # 내장 생성기
        self.model.gravity.linear = np.array([0.0, -p.g, 0.0])   # 기본값은 -z 이므로 -y 로 변경
        self.data = self.model.createData()
        self.fid = self.model.getFrameId("tip")

    def M(self, q):
        self.pin.crba(self.model, self.data, q)
        M = self.data.M.copy()
        return np.triu(M) + np.triu(M, 1).T          # crba 는 위쪽 삼각만 채움

    def nle(self, q, v):
        return self.pin.nonLinearEffects(self.model, self.data, q, v).copy()

    def G(self, q):
        return self.pin.computeGeneralizedGravity(self.model, self.data, q).copy()

    def C(self, q, v):
        return self.pin.computeCoriolisMatrix(self.model, self.data, q, v).copy()

    def rnea(self, q, v, a):
        return self.pin.rnea(self.model, self.data, q, v, a).copy()

    def aba(self, q, v, tau):
        return self.pin.aba(self.model, self.data, q, v, tau).copy()

    def fk(self, q):
        pin = self.pin
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        tip = self.data.oMf[self.fid].translation[:2].copy()
        pin.computeJointJacobians(self.model, self.data, q)
        J = pin.getFrameJacobian(self.model, self.data, self.fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:2, :].copy()
        return tip, J


class MjBackend:
    def __init__(self, p, mjcf_path=None):
        import mujoco
        self.mj = mujoco
        if mjcf_path:
            self.m = mujoco.MjModel.from_xml_path(mjcf_path)     # 실제 MJCF 파일
        else:
            self.m = mujoco.MjModel.from_xml_string(mjcf_string(p))
        self.d = mujoco.MjData(self.m)
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "tip")

    def _set(self, q, v):
        self.d.qpos[:] = q
        self.d.qvel[:] = v
        self.d.qacc[:] = 0
        self.d.qfrc_applied[:] = 0
        self.mj.mj_forward(self.m, self.d)

    def M(self, q):
        self._set(q, np.zeros(2))
        M = np.zeros((self.m.nv, self.m.nv))
        try:
            self.mj.mj_fullM(self.m, M, self.d.qM)       # 공식 시그니처
        except (TypeError, AttributeError):
            self.mj.mj_fullM(self.m, self.d, M)          # 버전에 따라 다른 시그니처
        return M

    def bias(self, q, v):
        """qfrc_bias = C q̇ + G (MuJoCo 에는 C 행렬 자체를 주는 함수가 없음)"""
        self._set(q, v)
        return self.d.qfrc_bias.copy()

    def inverse(self, q, v, a):
        self._set(q, v)
        self.d.qacc[:] = a
        self.mj.mj_inverse(self.m, self.d)
        return self.d.qfrc_inverse.copy()

    def forward(self, q, v, tau):
        self.d.qpos[:] = q
        self.d.qvel[:] = v
        self.d.qfrc_applied[:] = tau
        self.mj.mj_forward(self.m, self.d)
        a = self.d.qacc.copy()
        self.d.qfrc_applied[:] = 0
        return a

    def fk(self, q):
        self._set(q, np.zeros(2))
        tip = self.d.site_xpos[self.sid][:2].copy()
        jacp = np.zeros((3, self.m.nv))
        self.mj.mj_jacSite(self.m, self.d, jacp, None, self.sid)
        return tip, jacp[:2, :].copy()


# ============================================================
# 5. 한 상태에서 모든 방법의 결과를 계산
# ============================================================
def evaluate(q, v, a, tau, p, pb, mb):
    """반환: {방법: {양: 값}}. 양: M, Cv, G, tau_id(역동역학), qdd_fd(순동역학), tip, J"""
    out = {}
    # hand
    M = hand_M(q, p)
    C = hand_C(q, v, p)
    G = hand_G(q, p)
    Cv = C @ v
    tip, J = hand_fk(q, p)
    out["hand"] = dict(M=M, Cv=Cv, G=G, tau_id=M @ a + Cv + G,
                       qdd_fd=np.linalg.solve(M, tau - Cv - G), tip=tip, J=J)
    # NE
    z = np.zeros(2)
    G_ne = newton_euler(q, z, z, p)[0]
    Cv_ne = newton_euler(q, v, z, p)[0] - G_ne
    M_ne = np.zeros((2, 2))
    for i in range(2):
        ei = np.zeros(2)
        ei[i] = 1.0
        M_ne[:, i] = newton_euler(q, z, ei, p, gravity=0.0)[0]
    out["NE"] = dict(M=M_ne, Cv=Cv_ne, G=G_ne, tau_id=newton_euler(q, v, a, p)[0],
                     qdd_fd=np.linalg.solve(M_ne, tau - Cv_ne - G_ne))
    # Pinocchio
    if pb is not None:
        nle = pb.nle(q, v)
        G_p = pb.G(q)
        tip_p, J_p = pb.fk(q)
        out["pin"] = dict(M=pb.M(q), Cv=nle - G_p, G=G_p, tau_id=pb.rnea(q, v, a),
                          qdd_fd=pb.aba(q, v, tau), tip=tip_p, J=J_p)
    # MuJoCo
    if mb is not None:
        G_m = mb.bias(q, z)
        tip_m, J_m = mb.fk(q)
        out["mj"] = dict(M=mb.M(q), Cv=mb.bias(q, v) - G_m, G=G_m, tau_id=mb.inverse(q, v, a),
                         qdd_fd=mb.forward(q, v, tau), tip=tip_m, J=J_m)
    return out


QUANT_LABEL = {
    "M": "M(q) 관성 행렬",
    "Cv": "C(q,q̇)q̇ 코리올리-원심력",
    "G": "G(q) 중력",
    "tau_id": "역동역학 τ=f(q,q̇,q̈)",
    "qdd_fd": "순동역학 q̈=f(q,q̇,τ)",
    "tip": "끝단 위치 (FK)",
    "J": "Jacobian",
}


def maxerr(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


# ============================================================
# 6. 비교 실행
# ============================================================
def run_compare(p, n_random, tol, seed, urdf_path=None, mjcf_path=None):
    try:
        pb = PinBackend(p, urdf_path)
    except Exception as ex:  # noqa
        print(f"[경고] Pinocchio 를 사용할 수 없어 건너뜁니다: {ex}")
        pb = None
    try:
        mb = MjBackend(p, mjcf_path)
    except Exception as ex:  # noqa
        print(f"[경고] MuJoCo 를 사용할 수 없어 건너뜁니다: {ex}")
        mb = None

    print("=" * 78)
    print("모델 출처: Pinocchio <- {} / MuJoCo <- {}".format(urdf_path or "내장 생성기", mjcf_path or "내장 생성기"))
    print("시스템: m=({:.3g},{:.3g}) l=({:.3g},{:.3g}) I=({:.5g},{:.5g}) g={:.5g}  (무게중심 = 링크 중간)".format(
        p.m1, p.m2, p.l1, p.l2, p.I1, p.I2, p.g))
    if p.i1c != 1.0:
        print(f"[주의] 손 유도 M11 의 I1 계수를 {p.i1c} 로 두었습니다 (정답은 1.0). FAIL 이 나오는 것이 정상입니다.")
    print("=" * 78)

    # ---- 예시 상태 (q = 30°, 60°)
    q = np.array([np.deg2rad(30.0), np.deg2rad(60.0)])
    v = np.array([1.0, 0.5])
    a = np.array([-9.006136, 15.961706])
    tau = np.array([2.0, 0.5])
    res = evaluate(q, v, a, tau, p, pb, mb)
    print("\n[예시 상태] q=(30°,60°), q̇=(1,0.5), q̈=(-9.006136,15.961706), τ=(2,0.5)")
    for key in ["M", "Cv", "G", "tau_id", "qdd_fd", "tip", "J"]:
        print(f"\n -- {QUANT_LABEL[key]}")
        for name in ["hand", "NE", "pin", "mj"]:
            if name in res and key in res[name]:
                val = np.asarray(res[name][key])
                print(f"   {name:>4s}: " + np.array2string(val.reshape(-1), separator=", ") +
                      ("" if val.ndim < 2 else "   (행 우선)"))

    # ---- 무작위 상태 N개
    rng = np.random.default_rng(seed)
    methods = [m for m in ["NE", "pin", "mj"] if m in res]
    worst = {(m, k): 0.0 for m in methods for k in res[m]}
    for _ in range(n_random):
        q_ = rng.uniform(-np.pi, np.pi, 2)
        v_ = rng.uniform(-3, 3, 2)
        a_ = rng.uniform(-5, 5, 2)
        t_ = rng.uniform(-10, 10, 2)
        r = evaluate(q_, v_, a_, t_, p, pb, mb)
        for m in methods:
            for k in r[m]:
                worst[(m, k)] = max(worst[(m, k)], maxerr(r[m][k], r["hand"][k]))

    print("\n" + "=" * 78)
    print(f"[무작위 상태 {n_random}개] 손 유도(hand) 대비 최대 절대 오차   (합격 기준 {tol:g})")
    print("=" * 78)
    header = f"{'양':<30s}" + "".join(f"{m:>14s}" for m in methods)
    print(header)
    fails = 0
    method_ng = {m: False for m in methods}      # 방법별로 NG 가 하나라도 있었는지
    for k in ["M", "Cv", "G", "tau_id", "qdd_fd", "tip", "J"]:
        row = f"{QUANT_LABEL[k]:<28s}"
        for m in methods:
            if (m, k) in worst:
                e = worst[(m, k)]
                ok = e <= tol
                fails += (not ok)
                method_ng[m] = method_ng[m] or (not ok)
                row += f"{e:>10.2e} {'OK' if ok else 'NG':<3s}"
            else:
                row += f"{'-':>14s}"
        print(row)

    # ---- C 행렬 자체 (비유일성 때문에 Pinocchio 와만 비교) + 성질
    print("\n[C 행렬 비교] C 는 유일하지 않으므로 C q̇ (위 표)로 판정하고, 크리스토펠 C 끼리는 참고로 비교")
    if pb is not None:
        wc = 0.0
        for _ in range(n_random):
            q_ = rng.uniform(-np.pi, np.pi, 2)
            v_ = rng.uniform(-3, 3, 2)
            wc = max(wc, maxerr(hand_C(q_, v_, p), pb.C(q_, v_)))
        print(f"   hand C 대 Pinocchio computeCoriolisMatrix 최대 오차: {wc:.2e}")

    print("\n[성질 검사] (손 유도식 기준)")
    sym = pd_min = c_consist = skew = md_cct = 0.0
    pd_min = np.inf
    eps = 1e-6
    for _ in range(n_random):
        q_ = rng.uniform(-np.pi, np.pi, 2)
        v_ = rng.uniform(-3, 3, 2)
        M = hand_M(q_, p)
        sym = max(sym, maxerr(M, M.T))
        pd_min = min(pd_min, np.linalg.eigvalsh(M).min())
        c_consist = max(c_consist, maxerr(hand_C(q_, v_, p) @ v_, hand_C1C2(q_, v_, p)))
        Md = (hand_M(q_ + eps * v_, p) - hand_M(q_ - eps * v_, p)) / (2 * eps)
        C = hand_C(q_, v_, p)
        N = Md - 2 * C
        skew = max(skew, maxerr(N + N.T, 0))
        md_cct = max(md_cct, maxerr(Md - C - C.T, 0))
    checks = [
        ("M = Mᵀ (대칭)", sym, tol),
        ("C 행렬·q̇ = 손 유도의 C1, C2 식", c_consist, tol),
        ("Ṁ − 2C 반대칭 (수치 미분)", skew, 1e-6),
        ("Ṁ − C − Cᵀ = 0 (수치 미분)", md_cct, 1e-6),
    ]
    for name, e, t in checks:
        ok = e <= t
        fails += (not ok)
        print(f"   {name:<34s} 최대 오차 {e:9.2e}  {'OK' if ok else 'NG'}")
    ok = pd_min > 0
    fails += (not ok)
    print(f"   {'M 양정치 (최소 고유값 > 0)':<34s} 최소 고유값 {pd_min:9.3e}  {'OK' if ok else 'NG'}")

    print("\n" + ("전체 결과: 모두 합격" if fails == 0 else f"전체 결과: 불합격 항목 {fails}개"))

    # 힌트: 모델 파일을 읽었고, params.yaml 만으로 계산하는 NE 는 hand 와 일치하는데
    # 파일을 읽은 라이브러리만 어긋나면, 가장 흔한 원인은 params.yaml 을 고치고 파일을 다시 만들지 않은 경우
    stale = []
    if urdf_path and method_ng.get("pin") and not method_ng.get("NE"):
        stale.append(urdf_path)
    if mjcf_path and method_ng.get("mj") and not method_ng.get("NE"):
        stale.append(mjcf_path)
    if stale:
        models_dir = os.path.dirname(stale[0])
        print("\n[힌트] hand 와 NE 는 서로 일치하는데, 모델 파일을 읽은 라이브러리의 결과만 어긋났습니다.")
        print("  가장 흔한 원인: params.yaml 을 고친 뒤 URDF/MJCF 파일을 다시 만들지 않은 경우입니다.")
        for path in stale:
            print(f"    어긋난 파일: {path}")
        print("  파일을 params.yaml 에서 다시 생성하고 같은 실행에서 비교하려면 (레포 루트에서):")
        print(f'    python experiments/planar-2dof/dynamics_compare_2dof.py --write-models "{models_dir}"')
        print("  (관성 I=0 인 점질량은 MuJoCo 가 1e-9 로 대체해 생기는 오차일 수 있으며, 이 경우 --tol 1e-6 으로 확인하세요.)")
    return fails


# ============================================================
# 7. 선택: 손으로 쓴 L 에서 sympy 재유도
# ============================================================
def run_sympy(p, n_random, tol, seed):
    import sympy as sp
    m1, m2, l1, l2, g, I1, I2, i1c = sp.symbols("m1 m2 l1 l2 g I1 I2 i1c", positive=True)
    q1, q2, d1, d2, a1, a2 = sp.symbols("q1 q2 d1 d2 a1 a2")
    # 손 유도의 K, P (K 괄호 안의 I1 계수 = i1c)
    K = (sp.Rational(1, 2) * ((m1 / 4 + m2) * l1 ** 2 + i1c * I1) * d1 ** 2
         + m2 * l1 * l2 * d1 * (d1 + d2) * sp.cos(q2) / 2
         + (m2 * l2 ** 2 / 8 + I2 / 2) * (d1 + d2) ** 2)
    P = (m1 / 2 + m2) * l1 * g * sp.sin(q1) + m2 * l2 * g * sp.sin(q1 + q2) / 2
    L = K - P
    qs, ds, dd = [q1, q2], [d1, d2], [a1, a2]

    def ddt(e):
        return sum(sp.diff(e, qs[i]) * ds[i] + sp.diff(e, ds[i]) * dd[i] for i in range(2))
    tau_sym = sp.Matrix([ddt(sp.diff(L, ds[i])) - sp.diff(L, qs[i]) for i in range(2)])
    f = sp.lambdify((q1, q2, d1, d2, a1, a2, m1, m2, l1, l2, g, I1, I2, i1c), tau_sym, "numpy")
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_random):
        q = rng.uniform(-np.pi, np.pi, 2)
        v = rng.uniform(-3, 3, 2)
        a = rng.uniform(-5, 5, 2)
        ts = np.array(f(*q, *v, *a, p.m1, p.m2, p.l1, p.l2, p.g, p.I1, p.I2, p.i1c), float).ravel()
        th = hand_M(q, p) @ a + hand_C(q, v, p) @ v + hand_G(q, p)
        worst = max(worst, maxerr(ts, th))
    ok = worst <= tol
    print(f"\n[sympy 재유도] 손 유도의 L=K−P 를 오일러-라그랑주에 넣은 τ 와 손 유도 M,C,G 의 최대 오차: {worst:.2e}  {'OK' if ok else 'NG'}")
    return 0 if ok else 1


# ============================================================
# 8. 선택: PD + 중력 보상 Lyapunov 수치 확인 (손 유도 동역학 사용)
# ============================================================
def run_lyapunov(p):
    Kp = np.diag([60.0, 40.0])
    Kd = np.diag([12.0, 8.0])
    qdes = np.array([1.0, 0.5])

    def simulate(label, Ts=None, delay=0.0, sat=None, gmodel=1.0, fc=0.0, T=3.0, dt=2e-4):
        q = np.array([-0.5, 1.2])
        v = np.zeros(2)
        n = int(T / dt)
        Vs = np.zeros(n)
        k_ts = 1 if Ts is None else max(1, int(round(Ts / dt)))
        buf = [np.zeros(2)] * int(round(delay / dt) + 1)
        tau_cmd = np.zeros(2)
        for k in range(n):
            if k % k_ts == 0:
                tau_cmd = gmodel * hand_G(q, p) - Kp @ (q - qdes) - Kd @ v
                if sat is not None:
                    tau_cmd = np.clip(tau_cmd, -sat, sat)
            buf.append(tau_cmd)
            tau = buf.pop(0)

            def f(qq, vv):
                M = hand_M(qq, p)
                rhs = tau - hand_C(qq, vv, p) @ vv - hand_G(qq, p) - fc * np.tanh(vv / 1e-3)
                return vv, np.linalg.solve(M, rhs)
            k1 = f(q, v)
            k2 = f(q + dt / 2 * k1[0], v + dt / 2 * k1[1])
            k3 = f(q + dt / 2 * k2[0], v + dt / 2 * k2[1])
            k4 = f(q + dt * k3[0], v + dt * k3[1])
            q = q + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            v = v + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            e = q - qdes
            Vs[k] = 0.5 * v @ hand_M(q, p) @ v + 0.5 * e @ Kp @ e
        dV = np.diff(Vs)
        thr = 1e-6 * Vs[0]
        print(f"  {label:<26s} V0={Vs[0]:7.2f}  V_end={Vs[-1]:9.2e}  "
              f"허용치 초과 증가 스텝={int((dV > thr).sum()):6d}  |e_end|={np.linalg.norm(q - qdes):.2e}")

    print("\n[Lyapunov] PD + 중력 보상, V=½q̇ᵀMq̇+½eᵀKp e, e=q−q_d, 판정 허용치 = 1e-6·V0")
    print("  (Kp=diag(60,40), Kd=diag(12,8), q0=(-0.5,1.2), q_d=(1,0.5), 3초)")
    simulate("이상적")
    simulate("이산화 Ts=1 ms", Ts=1e-3)
    simulate("지연 5 ms", delay=5e-3)
    simulate("지연 20 ms", delay=20e-3)
    simulate("포화 ±25 Nm", sat=25.0)
    simulate("포화 ±13 Nm", sat=13.0)
    simulate("모델 오차 (Ĝ=0.8 G)", gmodel=0.8)
    simulate("쿨롱 마찰 0.5 Nm", fc=0.5)


# ============================================================
# 9. 메인
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="2DOF 동역학 비교: 손 유도(E-L) / 뉴턴-오일러 / Pinocchio / MuJoCo")
    ap.add_argument("--m1", type=float, default=None)
    ap.add_argument("--m2", type=float, default=None)
    ap.add_argument("--l1", type=float, default=None)
    ap.add_argument("--l2", type=float, default=None)
    ap.add_argument("--I1", type=float, default=None, help="미지정이면 params.yaml 값 (파일이 없으면 m1*l1^2/12)")
    ap.add_argument("--I2", type=float, default=None, help="미지정이면 params.yaml 값 (파일이 없으면 m2*l2^2/12)")
    ap.add_argument("--g", type=float, default=None)
    ap.add_argument("--models-dir", default=DEFAULT_MODELS_DIR, help="params.yaml, URDF, MJCF 가 있는 폴더 (기본: ../../models/planar-2dof)")
    ap.add_argument("--embedded", action="store_true", help="모델 파일을 읽지 않고 내장 생성기로 만든 모델을 사용")
    ap.add_argument("--n-random", type=int, default=1000)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--i1-coeff", type=float, default=1.0, help="손 유도 M11 의 I1 계수 (정답 1.0, 0.5 로 두면 FAIL 확인)")
    ap.add_argument("--sympy", action="store_true", help="손 유도의 L 에서 sympy 로 재유도해 비교")
    ap.add_argument("--lyapunov", action="store_true", help="PD+중력보상 V(t) 수치 확인")
    ap.add_argument("--write-models", metavar="DIR", help="현재 파라미터로 planar_2dof.urdf / planar_2dof.xml 을 DIR 에 생성")
    ap.add_argument("--view", action="store_true", help="MuJoCo 뷰어 실행 (데스크탑 전용, 미시험)")
    args = ap.parse_args()

    overrides = [args.m1, args.m2, args.l1, args.l2, args.I1, args.I2, args.g]
    yaml_path = os.path.join(args.models_dir, YAML_NAME)
    urdf_path = os.path.join(args.models_dir, URDF_NAME)
    mjcf_path = os.path.join(args.models_dir, MJCF_NAME)

    # 파라미터: params.yaml (있으면) -> 명령줄 값으로 덮어쓰기 -> 없으면 기본값
    base = None
    if not args.embedded and os.path.isfile(yaml_path):
        try:
            base = load_params_yaml(yaml_path)
        except ImportError:
            print("[경고] pyyaml 이 없어 params.yaml 을 읽지 못했습니다 (pip install pyyaml). 내장 값으로 진행합니다.")
        except Exception as ex:  # noqa
            print(f"[오류] params.yaml 을 읽지 못했습니다: {ex}")
            return 2
    b = base if base is not None else Params()
    pick = lambda v, d: d if v is None else v  # noqa: E731
    p = Params(pick(args.m1, b.m1), pick(args.m2, b.m2), pick(args.l1, b.l1), pick(args.l2, b.l2),
               pick(args.I1, b.I1), pick(args.I2, b.I2), pick(args.g, b.g), args.i1_coeff)

    # 모델 파일을 쓸 수 있는 조건: yaml 을 읽었고, 값을 덮어쓰지 않았고, 두 파일이 모두 있음
    use_files = (base is not None and not any(v is not None for v in overrides)
                 and os.path.isfile(urdf_path) and os.path.isfile(mjcf_path) and not args.embedded)
    if base is not None and any(v is not None for v in overrides):
        print("[알림] 명령줄로 파라미터를 바꿨으므로 모델 파일 대신 내장 생성기로 만든 모델을 사용합니다.")
    elif base is not None and not use_files and not args.embedded:
        print(f"[알림] {args.models_dir} 에 URDF/MJCF 파일이 없어 내장 생성기를 사용합니다 (--write-models 로 생성 가능).")
    if not use_files:
        urdf_path = mjcf_path = None

    if args.write_models:
        os.makedirs(args.write_models, exist_ok=True)
        with open(os.path.join(args.write_models, URDF_NAME), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(urdf_string(p))
        with open(os.path.join(args.write_models, MJCF_NAME), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(mjcf_string(p))
        print(f"모델 파일을 {args.write_models} 에 저장했습니다.")

    fails = run_compare(p, args.n_random, args.tol, args.seed, urdf_path, mjcf_path)
    if args.sympy:
        fails += run_sympy(p, min(args.n_random, 200), args.tol, args.seed)
    if args.lyapunov:
        run_lyapunov(p)
    if args.view:
        import mujoco
        import mujoco.viewer
        m = mujoco.MjModel.from_xml_path(mjcf_path) if mjcf_path else mujoco.MjModel.from_xml_string(mjcf_string(p))
        d = mujoco.MjData(m)
        d.qpos[:] = [np.deg2rad(30.0), np.deg2rad(60.0)]
        mujoco.viewer.launch(m, d)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
