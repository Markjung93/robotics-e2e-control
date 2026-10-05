# Robotics E2E Control: Roadmap

로봇 제어의 수학 기초부터 시뮬레이션, 강화학습, 통합 애플리케이션까지를 **하나의 C++ 프레임워크 위에서** 직접 구현하고 검증하는 프로젝트입니다.

## 목표

- 수학적 개념을 직접 구현하고(Eigen), 검증된 라이브러리(Pinocchio, MuJoCo)와 교차 검증합니다.
- 모든 결과는 **재현 가능한 README + 시각화 + 비교 실험 + 한계 분석**을 포함합니다.
- 모든 구현에 "실제 로봇에 적용하려면 무엇이 필요한가"(실시간 제약, 안전, 센서 노이즈, Sim-to-Real 갭)를 함께 기록합니다.

## 아키텍처 (계획)

```
apps (Web UI / Unity 디지털 트윈)
        │ gRPC
comm ─ planning ─ control ─ model ─ core ─ hal
                            │ IRobotModel
                            ├─ 자체 구현 (Eigen)
                            └─ Pinocchio 백엔드
```

- 실시간(RT) 영역과 비실시간 영역을 분리합니다.
- 모델 계산은 `IRobotModel` 인터페이스 뒤에서 자체 구현과 Pinocchio를 교체해서 비교합니다.

## 개발 방식: 학습 사이클

매 학습 단위(동역학, PID, FK, LQR 등)를 아래 순서로 끝내고, 결과를 이전 버전과 데이터로 비교합니다.

```mermaid
flowchart LR
  A[개념 노트] --> B[직접 구현]
  B --> C[프레임워크 모듈]
  C --> D[MuJoCo 반영]
  D --> E[UI에서 조작]
  E --> F[이전 버전과 비교]
  F --> G[리포트]
```

- 새 알고리즘은 제어기 레지스트리에 등록하면 UI가 파라미터 패널을 자동으로 구성합니다.
- 표준 시나리오(스텝 응답, 원 궤적, 외란 주입)와 표준 지표(RMS 오차, 최대 오차, 정착 시간, 최대 토크, 계산 시간)로 비교합니다.
- UI는 비실시간 게이트웨이를 통해서만 제어 루프와 통신합니다.

## 로드맵

모든 Phase는 UI에서 조작 가능한 시연과 이전 버전 대비 비교 데이터를 포함합니다. 일정은 진행하면서 확정합니다.

| Phase | 주제 | 핵심 내용 | 상태 |
|---|---|---|---|
| P0 | 기반 정리 | 레포 구조, 개발 환경, 문서 체계 | [ ] |
| P1 | 2DOF 동역학 | 오일러-라그랑주 유도, C++/Eigen 구현, MuJoCo 검증, PID와 계산토크 제어 | [ ] |
| P2 | 6축 기구학 | FK, Jacobian, 역기구학, 특이점 회피, Pinocchio 교차 검증 | [ ] |
| P3 | 고급 제어 | LQR, 선형/비선형 MPC, 외란 강건성 | [ ] |
| P4 | ROS2 | ros2_control 연동, EtherCAT 드라이버 구조 분석 | [ ] |
| P5 | 7축 여유 자유도 | 영공간 제어, 텔레오퍼레이션 | [ ] |
| P6 | 강화학습 | MuJoCo에서 PPO, Isaac Lab 확장, Sim-to-Real 정리 | [ ] |
| P7 | 모방학습 / VLA | ACT 기반 모방학습, VLM 기반 task planning | [ ] |
| P8 | 통합 | gRPC 전환, Unity 디지털 트윈, 모터 하드웨어 실습, 데모 | [ ] |

## 기술 스택

- C++17 이상, CMake, Eigen, Pinocchio, Lie group 라이브러리
- MuJoCo, Isaac Lab, ROS2
- Python (분석, 시각화, 학습), JavaScript(React), Unity(C#)

## 폴더 구조 (계획)

```
robotics-e2e-control/
├── README.md
├── framework/        # core, model, control, hal, comm
├── experiments/      # 01_2dof_dynamics, 02_6dof_kinematics, ...
├── apps/             # web-ui, unity-twin
├── sim/              # urdf, mjcf
└── docs/             # 리포트, 다이어그램
```

## 리포트 형식

알고리즘 구현과 실험은 논문 형식(Abstract, Related Work, Method, Experiments, Results, Limitations, Reproducibility, 현실 적용 고찰)으로 정리합니다.

## 참고 프로젝트

- [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics)
- [SOEM](https://github.com/OpenEtherCATsociety/SOEM)
- [ethercat_driver_ros2](https://github.com/ICube-Robotics/ethercat_driver_ros2)
- [control-libraries](https://github.com/epfl-lasa/control-libraries)
