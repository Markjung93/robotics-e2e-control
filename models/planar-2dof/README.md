# planar-2dof (평면 2DOF 팔, 수직 평면)

| 파일 | 역할 | 읽는 쪽 |
|---|---|---|
| `params.yaml` | 질량, 길이, 무게중심, 관성, 중력 (**단일 출처**) | 손 유도식, 뉴턴-오일러, 시험 스크립트 |
| `planar_2dof.urdf` | 링크와 관절 구조, 질량 특성 (로봇 모델 형식) | Pinocchio |
| `planar_2dof.xml` | 같은 시스템의 MuJoCo 모델 (MJCF) | MuJoCo |

- 출처: 직접 설계한 값입니다 (외부 로봇 모델을 사용하지 않음).
- 좌표계: x-y 평면, 중력 -y, 관절 축 z. q1은 절대 각도, q2는 링크1 기준 상대 각도.
- URDF와 MJCF는 `params.yaml`에서 생성합니다. 파일을 직접 고치지 말고 `params.yaml`을 고친 뒤 다시 생성하세요.

```bash
python experiments/planar-2dof/dynamics_compare_2dof.py --write-models models/planar-2dof
python experiments/planar-2dof/dynamics_compare_2dof.py
```

두 번째 명령은 파일이 `params.yaml`과 일치하는지, 손 유도식, 뉴턴-오일러, Pinocchio, MuJoCo가 서로 일치하는지 한 번에 확인합니다.
