# planar-2dof: 동역학 교차 검증

평면 2DOF 팔(수직 평면)의 동역학 `M(q) q̈ + C(q,q̇) q̇ + G(q) = τ` 를 네 가지 방법으로 계산해 서로 비교합니다.

| 방법 | 내용 |
|---|---|
| hand | 손으로 유도한 오일러-라그랑주 최종식 |
| NE | 뉴턴-오일러 재귀 (힘과 모멘트 평형) |
| Pinocchio | `models/planar-2dof/planar_2dof.urdf` 를 읽어 CRBA, RNEA, ABA 로 계산 |
| MuJoCo | `models/planar-2dof/planar_2dof.xml` 을 읽어 `mj_fullM`, `mj_inverse`, `mj_forward` 로 계산 |

파라미터는 `models/planar-2dof/params.yaml` 이 단일 출처입니다. URDF 와 MJCF 가 이 값과 어긋나면 비교가 FAIL 로 끝납니다.

## 설치

Python 3.10 이상. 패키지: `numpy`, `pyyaml`, `pin`(Pinocchio), `mujoco`, 선택 `sympy`.

```bash
# Linux, WSL2 (pip 로 모두 설치 가능)
pip install numpy pyyaml pin mujoco sympy

# Windows: pin 의 pip 설치는 지원되지 않으므로 conda-forge 를 사용
conda create -n robotics python=3.12
conda activate robotics
conda install -c conda-forge pinocchio numpy pyyaml
pip install mujoco sympy
```

## 실행

```bash
python experiments/planar-2dof/dynamics_compare_2dof.py            # 네 가지 방법 비교 + 성질 검사
python experiments/planar-2dof/dynamics_compare_2dof.py --sympy    # 라그랑지안에서 sympy 로 재유도해 비교
python experiments/planar-2dof/dynamics_compare_2dof.py --lyapunov # PD + 중력 보상의 V(t) 수치 확인
```

모든 항목이 `OK` 이면 종료 코드 0, 하나라도 `NG` 이면 1 입니다.

| 옵션 | 용도 |
|---|---|
| `--i1-coeff 0.5` | 손 유도 M11 의 I1 계수를 예전 오류(1/2)로 두어 FAIL 이 나는지 확인 |
| `--m1 --m2 --l1 --l2 --I1 --I2 --g` | 파라미터를 바꿔 비교 (이때는 파일 대신 내장 생성기의 모델을 사용) |
| `--n-random N`, `--tol T`, `--seed S` | 무작위 상태 개수(기본 1000), 허용 오차(기본 1e-9), 시드 |
| `--write-models DIR` | 현재 파라미터로 URDF, MJCF 를 DIR 에 생성 (`models/planar-2dof` 갱신용) |
| `--embedded` | 모델 파일을 읽지 않고 내장 생성기 모델로 비교 |
| `--view` | MuJoCo 뷰어 (미시험) |

## 결과 (기본 파라미터, 무작위 상태 1000개, hand 대비 최대 오차)

| 양 | NE | Pinocchio | MuJoCo |
|---|---|---|---|
| M(q) | 8.9e-16 | 4.4e-16 | 1.8e-15 |
| C q̇ | 6.7e-15 | 5.4e-15 | 7.2e-15 |
| G | 3.6e-15 | 7.1e-15 | 7.1e-15 |
| 역동역학 τ | 7.1e-15 | 7.1e-15 | 1.8e-14 |
| 순동역학 q̈ | 1.6e-13 | 1.0e-13 | 2.3e-13 |

## 한계와 현실 적용 시 고려사항

- 무게중심이 링크 중간이고 마찰, 백래시, 탄성이 없는 이상적인 강체 모델입니다.
- 관성이 0인 점질량은 MuJoCo 가 양수 관성을 요구해 1e-9 로 대체하므로 오차가 커집니다 (`--tol` 을 완화).
- 실제 로봇에서는 파라미터 식별 오차, 마찰, 제어 주기 지터, 센서 노이즈가 이 모델과의 차이를 만듭니다.
- 이 스크립트는 Python 검증용이며, 실시간 영역의 C++ 구현은 이 결과를 정답(회귀 시험 기준)으로 삼습니다.
