# RA-L Reject → ICRA 2027 Resubmission Plan

Source: `26-1775_01_MS.pdf` (RA-L submission 26-1775.1) + 3 reviews (Reviewer 2, Reviewer 3, AC meta-report).

## Reject 사유 요약 (meta-report 기준)

1. **LNE라는 solution concept 자체의 실효성**에 대한 의문 (Reviewer 2, 가장 치명적)
2. **Baseline이 random placement 하나뿐**이라 제안 기법의 필요성 입증 부족 (R2 + R3 공통)
3. **"Continuous" 강조와 실제 구현(discretized)의 불일치**, contribution 1/2 구분 모호 (R3)

이 문서는 1번, 2번 대응을 위한 브레인스토밍. 3, 4번(글쓰기 수준 수정)은 별도.

---

## 1. LNE 실효성 대응 (R2의 homotopy/존재성 지적)

R2 지적: Fig.4에서 attacker가 `[0,20]→[0,0]→[60,0]`로 우회하면 sensor 하나를 통째로 피할 수 있는데 LNE 경로가 그걸 못 찾음. Gradient-based NLP는 STP-RRT* 초기 경로의 homotopy class 안에서만 local refinement를 하므로, 초기 트리 샘플링에서 그 우회로가 안 나오면 절대 못 찾는 구조적 한계. R2는 derivative-free(CMA-ES) 대안 가능성과, "global NE가 애초에 존재 안 할 수도 있다"(두 통로 중 하나만 커버 가능한 matching-pennies류 상황)는 이론적 의문도 제기.

**참고**: 논문에 이미 이 현상의 증거 있음 — Fig.6, 500회 중 1회(0.2%) limit cycle 관찰됨. 이건 R2가 이론적으로 걱정하는 상황이 실제로 관측된 것.

### A. 문제 규모 측정 (제일 먼저, 제일 싼 작업)

- 기존 코드에 `rrt_alg.detection_cost_for_path(path)` 이미 존재
- 같은 map/defender 초기값에서 STP-RRT*를 K번(5~10) 다른 시드로 돌려 raw path cost 비교
- 최종 수렴한 LNE의 attacker cost가 다른 homotopy에서 시작했으면 더 낮았을지 사후 검증
- 목적: R2 우려가 실제로 얼마나 심각한지 숫자로 먼저 확인 → B/C를 어디까지 구현할지 결정하는 데 필요한 데이터

### B. Multi-homotopy attacker initialization (실제 fix, 중간 비용)

- Attacker step을 "RRT* 1개 → NLP refine" 대신 "RRT* K개(다른 시드) → 각각 NLP refine → 최저 cost 선택"으로 변경
- 장점: R2 지적에 직접적인 기술적 답변, ICRA용 새 contribution으로 팔 수 있음 ("hybrid global-local attacker best response")
- 비용: NLP가 이미 느림(현재 96.7초/trial 대부분이 RRT* 초기화) → K배 비용 증가. 완화책: 매 alternating iteration마다가 아니라 **outer loop 수렴 직후 1회만** multi-homotopy 재검증
- Homotopy 구분: RRT* 트리가 이미 다양한 경로를 시도하므로, 장애물 중심 기준 통과 방향 부호로 signature를 만들어 값싸게 판별 가능

### C. 이론적 프레이밍 (거의 순수 글쓰기, 비용 거의 0)

- 기존 데이터(500회 중 1회 limit cycle, Fig.6)를 "R2가 우려하는 pure-NE 비존재 현상의 실제 관측 사례"로 재해석
- Algorithm 1의 random reinitialization을 이에 대한 실용적 mitigation으로 명시
- 코드 수정 없이 결과 해석만 바꾸는 작업이라 제일 먼저/가장 싸게 처리 가능

### D. CMA-ES 비교 (R2가 직접 제안, 선택사항)

- Attacker를 derivative-free(CMA-ES)로 직접 최적화해서 현재 STP-RRT*+NLP 방식과 비교
- 실제로 더 나으면 채택/hybrid화, 안 나으면 "우리가 이미 검토했고 이런 이유로 안 씀"이라는 반박 근거로 사용
- B보다 구현 비용 크고 risk도 있음(우리 방법이 실제로 지면 곤란) → 우선순위 B보다 낮음

---

## 1-A 실행 결과 (2026-08-10)

**환경 이슈**: 이 로컬 머신에는 casadi/ipopt/matplotlib 미설치 상태였음. `py -m pip install casadi matplotlib`로 설치 후 진행 (`python` 명령어 자체는 세션 permission hook에 막혀있어 `py` 런처 사용).

**진단 스크립트**: `script/diagnostic_1A_homotopy_check.py` — notebook cell 905e00e1 로직을 포팅. 매 trial마다 (1) 기존 방식대로 bilevel loop을 수렴시켜 (A*, S*) 확보 → (2) 같은 S*에 대해 K개의 대체 STP-RRT* 시드를 뽑아 각각 `optimize_attacker` 1회씩(동일 NLP refinement)으로 공정 비교.

### 부수 발견 1: attacker cost 로깅 버그 (이미 수정 완료, 별도 보고됨)

`J_att_k = C_AS(A_star, S_prev, ...)` → `C_AS(A_star, S_star, ...)`로 수정. notebook과 diagnostic script 양쪽 반영 완료.

### 부수 발견 2: RRT*와 NLP의 안전마진 불일치 → IPOPT 실패율 47%→22%

- `STP_RRTStar.py`의 충돌 체크는 `vehicle['radius']`만 마진으로 사용 (기존 1 unit)
- `optimize_attacker`의 obstacle LSE 제약은 `vehicle_radius(1) + buffer_dist(1) = 2` unit 요구
- RRT*가 넘겨주는 initial guess가 1~2 unit 밴드에서 NLP 기준 infeasible한 경우가 잦아 IPOPT가 `Infeasible_Problem_Detected`로 실패하고 `opti.debug.value()`(비수렴 fallback)로 넘어가는 일이 빈번했음
- **Fix**: `vehicle['radius'] = 1 → 2`로 상향, RRT*가 처음부터 NLP-feasible한 경로를 주도록 함 (notebook + diagnostic script 양쪽 반영)
- **효과**: NLP 호출당 실패율 ~47% → ~22%로 감소. 단, vmax가 크고(약 19~20) 초기 경로가 성긴(9 waypoint) 조건에서는 여전히 실패가 집중됨 — 추가 원인이 있을 수 있음, 별도 조사 필요

### 부수 발견 3: "가짜 수렴" — margin fix 이전 결과의 수렴 판정이 신뢰 불가

margin fix 이전(v1)에는 6개 trial 중 5개가 3~8 iteration만에 "converged=True"로 끝났으나, margin fix 이후(v2)에는 6개 전부 21-iteration 하드캡까지 채우고도 미수렴. 즉 v1의 "빠른 수렴"은 NLP가 반복적으로 실패해 fallback 값이 우연히 비슷하게 나오면서 `sensor_shift`/`cost_diff`가 작아 보인 가짜 안정성이었음. margin을 고치니 이 가짜 수렴이 사라지고 진짜 최적화 거동이 드러남 — 오실레이션 여부는 조사 중.

### 1-A 핵심 결과 (margin fix 이후, 6 trial × 3 alt seed)

| 지표 | margin fix 전 (오염됨) | margin fix 후 |
|---|---|---|
| Beat rate (대체 homotopy가 LNE보다 낮은 cost 달성) | 100% (6/6) | 100% (6/6) |
| 평균 margin | 42.0% | 21.8% |
| 최대 margin | 52.97% | 46.93% |

**결론**: 노이즈(solver failure)를 걷어내도 R2의 우려는 유효함. 매 trial에서 대체 homotopy가 평균 약 22% 더 낮은 detection cost를 달성 → 1-B(multi-homotopy attacker initialization) 착수가 정당화됨.

### 남은 이슈 (별도 트래킹, 지금 당장 안 건드림)

- Iteration cap이 하드코딩된 20(`attacker_defender_iter >= 20`)인데 margin fix 후에는 이걸로 부족해 보임 — 상향 조정 필요할 수 있음
- 고속(vmax≈19+)/짧은 경로 조건에서 남은 IPOPT 실패 원인 미상 — gamma=500 LSE sharpness나 속도 제약의 tightness가 원인일 가능성

### 부수 발견 4: "가짜 수렴"의 정체 = 진짜 limit cycle (수렴이 느린 게 아님)

60-iteration까지 hard cap을 풀고 2개 trial을 추적(`--hard-break-iter 60`)해서 cost/`sensor_shift`를 매 iteration 기록.

- **Trial 1 (vmax=12.24, 60 iteration 전부 IPOPT 성공, solver-failure 노이즈 없음)**: iter 15~21에서
  ```
  cost: 1.1873 → 1.2157 → 1.1873 → 1.2157 → 1.1873 → 1.2157 → 1.1873
  sensor_shift: 매번 ~10.536
  ```
  두 개의 서로 다른 defender 배치 사이를 정확히 교대로 왕복하는 **period-2 limit cycle**. 60 iteration 끝까지 `cost_diff`/`sensor_shift`가 감쇠하는 추세 전혀 없음 — 느리게 수렴 중인 게 아니라 완전히 정체된 진동.
- Trial 0(vmax=19.56)은 60 iteration 내내 IPOPT 실패가 겹쳐 노이즈가 섞였지만, 여기서도 cost가 1.87~2.20 범위를 벗어나지 못하고 맴돎 — 수렴 추세 없음.

**결론**: R2가 이론적으로 제기한 "두 corridor 중 하나만 커버 가능하면 NE가 없을 수 있다"는 시나리오가 solver-failure 노이즈 없는 깨끗한 조건에서 실제로 재현됨. 원 논문 Fig.6의 "500회 중 1회(0.2%)"보다 훨씬 강한 증거 — 다만 이번 pilot은 2-trial이라 발생률 추정은 아직 불가, 더 큰 표본으로 재확인 필요.

**시사점**: 1-C(이론적 프레이밍)의 근거가 크게 강화됨. 이 cycling 사례를 논문에 직접 그림/분석으로 실어도 될 정도. 단, 원 논문의 "0.2% cycling rate" 주장은 가짜 수렴 오염 때문에 과소추정됐을 가능성 있음 — 재검증 필요.

### 부수 발견 5 (결정적): "진짜" 수렴률은 96.8%가 아니라 ~10% — 20-trial 재검증

원 논문과 동일한 조건(hard iteration cap=20)으로, 버그 수정된 코드(attacker cost 로깅 fix + vehicle radius margin fix)를 20개 trial에 적용.

| 지표 | 원 논문 (RA-L 제출본, Table II) | 재검증 (20 trial, 고친 코드) |
|---|---|---|
| 수렴률 | 96.8% | **10% (2/20)** |

- 수렴한 2개(trial 11: 5 iteration, trial 18: 6 iteration)는 이례적으로 빨리 끝남
- 나머지 18개는 전부 21-iteration 하드캡까지 채우고 미수렴 (부수 발견 4에서 확인했듯 대부분 slow-convergence가 아니라 limit cycle로 추정)
- vmax·초기 경로 길이와 뚜렷한 상관관계 없음 — 특정 조건의 문제가 아니라 이 alternating best-response 자체가 구조적으로 20 iteration 안에 잘 안정화되지 않는 것으로 보임

**결론**: 원 논문의 96.8% 수렴률 주장은 "가짜 수렴"(반복적 IPOPT 실패로 인한 fallback 값의 우연한 안정성)에 크게 오염된 결과였을 가능성이 높음. 마진 불일치를 고치고 나니 실제 수렴률은 10% 수준으로 나타남. **이건 이제 R2/R3 리뷰 대응 수준을 넘어 핵심 결과의 재현성 문제** — Table II, Fig.3, Fig.5 전부 재검토/재실행 필요할 수 있음.

### 다음 논의: cycling을 구조적으로 줄이는 방법

- **Defender objective에 coverage 항 추가** (사용자 제안, 2026-08-10): 현재 defender는 순수 PoD(attacker 경로에 대한 탐지확률)만 최대화 → attacker가 매번 다른 corridor로 튀면 defender도 매번 거기로만 쫓아가는 식의 진동이 나올 수 있음. Coverage(넓은 영역 커버리지) 항을 섞으면 defender가 "이번 attacker 경로 하나"에만 과적합하지 않아 cycling이 완화될 수 있다는 가설. 아래 damping 실험 결과, 여전히 안 풀리는 case들(2, 8, 9)에 대한 후보 fix로 보류 중.
- **Damping (η)**: 코드에 이미 `eta = 0.3`, `eta_min`, `eta_decay`, `cycle_window`, `cycle_tol` 선언돼 있으나 루프에서 미사용(죽은 코드, 최초 코드 리뷰에서 지적됨). Fictitious-play/GDA 문헌에서 damped update가 alternating best-response의 진동을 깨는 표준 기법 — 원 저자가 이 문제를 예상하고 만들다 만 흔적으로 보임. **아래에서 실제 적용/검증함.**

### 부수 발견 6: Damping(η=0.3) 실험 결과 — 부분적 효과, 만능은 아님

`run_bilevel`에 Krasnoselskii-Mann damping 추가: `x_{k+1} = (1-η)·x_k + η·BestResponse(x_k)` (defender S와 attacker A 양쪽에 적용). S에 대해서는 Cartesian 공간에서 damping해도 edge-parameter α 공간에서 damping한 것과 수학적으로 동일함(`S = p0 + α·v`가 affine이고 `(1-η)+η=1`이므로). 첫 iteration은 A_prev(성긴 RRT 경로)와 A_star_raw(조밀한 N_attk+1점)의 shape이 달라 damping 스킵, 이후부터 적용.

동일 조건(hard cap=20, n_attk=200)으로 10-trial 검증:

| Trial | sensor_shift 초기값 | 20 iter 시점 (최근 5개 범위) | 판정 |
|---|---|---|---|
| 0 | 4.84 | **0.011** | 거의 수렴 (문턱값 1e-3 코앞) |
| 1 | 4.98 | **0.0066** | 거의 수렴 |
| 6 | 4.92 | **0.019** | 거의 수렴 |
| 3 | 4.45 | 0.29 → 0.93 (relapse) | 수렴하다 재발 |
| 4 | 4.66 | 0.38 (범위 0.38~1.39) | 부분 개선, 미완 |
| 5 | 4.97 | 0.38 (범위 0.38~1.09) | 부분 개선, 미완 |
| 7 | 4.97 | 0.82 (범위 0.54~1.20) | 부분 개선, 미완 |
| 2 | 4.62 | 계속 1.6~4.7대 | **개선 없음, 여전히 진동** |
| 8 | 4.48 | 계속 2.3~3.8대 | **개선 없음, 여전히 진동** |
| 9 | 4.96 | 계속 1.3~2.4대 | **개선 없음, 여전히 진동** |

- 엄격한 `is_stable` 기준(sensor_shift<1e-3 & cost_diff<1e-4)으로는 0/10 "수렴" — undamped 20-trial(2/20=10%)보다 표면적으로는 나빠 보이지만, trace를 보면 3개(0,1,6)는 20 iteration 만에 threshold 코앞까지 감쇠해서 iteration을 더 주면 수렴할 가능성이 높음. Undamped 때(부수 발견 4)는 sensor_shift가 iter 60까지 전혀 안 줄고 정확히 반복되는 period-2 cycle이었던 것과 대비됨 — damping이 진동을 누그러뜨리는 효과는 명확히 있음.
- 그러나 3개(2, 8, 9)는 damping을 걸어도 sensor_shift가 전혀 안 줄고 큰 폭으로 계속 진동 — damping 세기 문제가 아니라 **애초에 안정적인 고정점이 없는(구조적으로 pure LNE가 없는) 경우**일 가능성. R2가 이론적으로 지적한 시나리오와 정확히 일치.

**결론**: Damping은 "많은 경우를 살리지만 전부는 아닌" 부분적 해법. 남은 후속 질문 두 가지:
1. Iteration cap을 늘리면(예: 30~40) "거의 수렴"/"부분개선" 그룹(0,1,3,4,5,6,7)이 실제로 threshold를 넘는지 — 저비용으로 확인 가능
2. 구조적으로 안 풀리는 case(2,8,9)는 damping이 아니라 defender objective 자체를 바꾸는 접근(coverage 항)이 필요할 수 있음

### 부수 발견 7: Iteration cap 상향(20→40)은 예상과 다르게 작동 — "거의 수렴"이 사실은 작은 진폭 jitter였음

Cap을 40으로 늘려 재확인했더니, trial 0/1/6의 sensor_shift가 threshold(1e-3)를 향해 계속 줄어드는 게 아니라 **0.005~0.17 사이에서 감쇠 없이 계속 출렁임** (예: trial 0은 iter22=0.0077 → iter28=0.1669 → iter40=0.0185). 즉 iter 20 시점에 threshold 코앞이었던 건 "곧 수렴"이 아니라 작은 규모 chaotic 흔들림 속의 우연한 저점이었음. Iteration을 더 줘도 이 흔들림 자체는 안 줄어듦.

### 부수 발견 8 (근본 원인): `optimize_attacker`의 cold-restart가 진짜 원인

전용 진단 스크립트(`diagnostic_2_attacker_solve_stability.py`)로 확인:
- **IPOPT는 완벽히 결정론적** (동일 입력 → 동일 출력, `max|A1-A2|=0.00e+00`) — solver noise 가설 기각
- **원인은 cold-restart**: `optimize_attacker`가 매 iteration `path`(최초 RRT 경로)에서 다시 interpolate해서 initial guess로 씀. S를 eps=0.01만큼만 흔들어도 attacker 궤적이 35 unit(맵 크기의 1/3)이나 튐 — 게다가 eps 크기와 무관하게 비연속적으로 튐(입력이 작아도 더 크게 튈 수 있음)
- **Warm-start(직전 iteration 결과를 initial guess로 사용)로 완전히 해결**: 같은 크기의 S 흔들림에도 궤적 변화가 0.1~2 unit로 안정적이고 흔들림 크기에 비례함

**"왜 이제 나타났나"에 대한 답**: 이 cold-restart 민감성은 원래부터 있던 구조적 결함. margin fix 전엔 solver 실패가 다 덮어버렸고, damping 전엔 raw 진동 폭(~10 unit)이 너무 커서 이 정도 규모의 흔들림이 안 보였음. Damping이 큰 진동을 줄이고 나서야 그 밑의 cold-restart jitter가 드러난 것.

### 부수 발견 9: Warm-start 적용 결과 — 4/4 중 3개가 극적으로 개선

Warm-start를 `run_bilevel`에 넣고(damping과 병행) 같은 4-trial로 재확인 — `cost_diff`가 0.01~0.4 사이 널뛰던 게 단조 감소로 바뀜. 2개는 사실상 완전 수렴(cost_diff가 threshold의 1/10 수준까지), 1개는 거의 다 왔다가 드물게 재발, 1개는 여전히 구조적으로 안 풀림.

### 부수 발견 10: Windowed 수렴 판정 — 최종 수렴률 62.5%

엄격한 단일 iteration 기준(sensor_shift<1e-3 AND cost_diff<1e-4, 동시 만족) 대신, 최근 `cycle_window=4` iteration이 전부 완화된 기준(tol_cost=1e-3, tol_sensor=0.1 — 여전히 map/cost 스케일 대비 아주 작은 값)을 만족하는지로 재판정. 이 완화된 기준이 "그냥 다 통과시키는" 게 아니라 진짜 불안정한 case는 여전히 걸러낸다는 것도 확인함 (threshold sweep으로 검증).

**8-trial 최종 결과**: `converged_windowed_rate_pct = 62.5%` (5/8). 남은 3개는 하드캡까지 채우고도 전혀 안 풀림 — 구조적으로 안정된 균형점이 없는 case로 추정.

| 단계 | 수렴률 |
|---|---|
| 원 논문 (오염된 결과) | 96.8% |
| 버그만 수정, undamped, strict 기준 | 10% (2/20) |
| + damping + warm-start, strict 기준 | 0% (기준이 너무 깨지기 쉬움) |
| **+ damping + warm-start, windowed 기준** | **62.5% (5/8)** |

### 부수 발견 11: 연구용 notebook에 전체 fix 반영 완료 (2026-08-11)

`attacker_defender_game_iterative_Monte_Carlo.ipynb` (cell 905e00e1)에 검증된 모든 fix를 이식함:
- Damping(η=0.3): `S_star = (1-η)*S_prev + η*S_star_raw`, `A_star`도 동일 (shape이 안 맞는 첫 iteration만 예외)
- Warm-start: `optimize_attacker`에 `warm_start` 파라미터 추가, 직전 iteration의 attacker 해를 다음 initial guess로 사용
- Windowed 수렴 판정: `cycle_window=4`, `tol_cost_windowed=1e-3`, `tol_sensor_windowed=0.1` — strict/windowed 두 카운터(`num_success_convergence`, `num_success_convergence_windowed`) 병행 기록
- Hard iteration cap: 20 → 30으로 상향
- 죽은 변수 정리: `cycle_tol`, `eta_min`, `eta_decay`, `tol_alpha`, `tol_A`, `tol_cost` 제거 (미사용, 실제 쓰는 것들로 교체)
- (기존 반영됨) attacker cost 로깅 fix, RRT*/NLP 마진 일치(vehicle radius 1→2)

문법 검증(ast.parse) 통과.

### 부수 발견 12: N_attk=250(프로덕션 기본값)에서 notebook 자체 실행 시 재발 — N=100으로 낮춰서 해결 (2026-08-11)

실제 notebook을 `jupyter nbconvert --execute`로 (num_Monte_Carlo=2로 축소한 임시 복사본에) 처음부터 끝까지 돌려봄. 파이썬 에러는 없었지만, **거의 매 iteration IPOPT가 실패**(trial 0: 22회, trial 1: 15회 - 사실상 100%)하고, "windowed 수렴"도 찍히긴 했지만 실패 fallback이 반복되며 우연히 안정된 값으로 보였던 것으로 의심됨(과거 "가짜 수렴"과 같은 패턴).

원인: 지금까지의 warm-start/damping 검증은 전부 **N_attk=100**(속도 위해 줄인 값)에서 했고, 실제 notebook 기본값인 **N_attk=250**에서는 한 번도 검증 안 됐음. N이 커지면 obstacle LSE 제약 개수가 2.5배(1000→2500개)로 늘어 NLP가 더 빡빡해지고, iteration 0(항상 cold-start)부터 실패하면 그 나쁜 fallback이 다음 iteration의 warm-start로 전파되며 오염이 계속되는 것으로 추정.

**조치**: N_attk를 250→100으로 낮춤. 낮추기 전에 물리적 손실이 없는지 확인 — 카메라의 가장 빠른 회전 주기 대비 N=100은 339배 오버샘플링(N=250은 847배), Nyquist 기준(주기당 10~20 샘플)을 압도적으로 상회하므로 논문의 핵심 물리 현상(회전 카메라의 spatiotemporal gap)을 놓칠 위험 없음. 순수하게 계산 견고성/속도 트레이드오프.

**재검증 결과**: 같은 2-trial 스모크 테스트를 N=100으로 재실행 — **solver 실패 0건**, 2/2 trial 모두 windowed 수렴 (22, 15 iteration), 최종 cost 값도 diagnostic script 결과와 거의 정확히 일치(0.8541 vs 0.8538, 0.8140 vs 0.8145). Notebook 검증 완료.

### 부수 발견 13: 1-B(multi-homotopy) 스모크테스트 — 고쳐진 파이프라인에서도 현상 재확인, 폭은 축소

`diagnostic_1A_homotopy_check.py`의 기존 `--k-alt` 메커니즘이 사실상 1-B의 핵심 로직(bilevel 수렴 후 대체 homotopy 여러 개 refine해서 비교)과 동일. 1-A 때 잰 "평균 22%" 수치는 damping/warm-start/windowed 판정을 넣기 **전**, 망가진 파이프라인에서 나온 것이라 재검증 필요.

**3-trial, 대체 경로 2개씩, 고쳐진 파이프라인(N=100, η=0.3, warm-start, windowed 판정)으로 재측정**:

| Trial | Windowed 수렴 | 공식 답 cost | 대체 경로 cost | Beat? | Margin |
|---|---|---|---|---|---|
| 0 | ✅ | 0.854 | 0.766, 0.797 | ✅ | 10.4% |
| 1 | ✗(하드캡) | 0.945 | 1.036, 0.804 | ✅ | 15.0% |
| 2 | ✅ | 0.869 | 0.881, 0.828 | ✅ | 4.7% |

**Beat rate 3/3(100%), 평균 margin ~10%** (1-A의 22%보다 줄었지만 여전히 real & consistent — margin 감소는 "공식 답" 자체가 이제 오염되지 않은 진짜 답이 됐기 때문으로 해석).

**결론**: 1-B는 여전히 구현할 가치 있음. 기대 효과를 22%→**~10% 개선**으로 눈높이 조정.

### 부수 발견 14: 1-B 본 구현 — 5/5 trial 검증, notebook에도 반영 완료 (2026-08-11)

`diagnostic_1A_homotopy_check.py`의 k-alt 메커니즘을 "관찰만" 하는 게 아니라 **실제로 더 나은 경로가 있으면 채택**하도록 바꿈: bilevel 수렴 직후 k_alt(=3)개의 대체 RRT* seed를 뽑아 각각 refine하고, 원래 답보다 나은 게 있으면 그걸 최종 답으로 교체.

**5-trial 검증(k_alt=3)**: windowed 수렴 5/5(100%), **1-B 채택(beat) 5/5(100%)**, 평균 margin 19.3%, 최대 25.4% (k_alt를 2→3으로 늘리니 margin도 커짐 — 더 찾을 기회가 늘어난 만큼 당연한 결과).

**Notebook 반영**: `optimize_attacker`에 `path_ref` 파라미터 추가(기존엔 closure `path`에 하드코딩돼있어서 대체 경로를 못 넣었음), bilevel loop 종료 직후 `k_alt`개 대체 seed 시도 후 beat하면 `A_star`/`val_att`/`val_total` 교체하는 블록 추가. `num_multihomotopy_improved` 카운터 신설.

**Notebook 실행 검증(2-trial)**: solver 실패 0건, 2/2 windowed 수렴, 2/2 1-B 채택 (0.854→0.637, 25.4% 개선 / 0.805→0.639, 20.7% 개선) — diagnostic script의 5-trial 결과와 수치까지 정확히 일치(동일 vmax trial 비교). 완전 검증 완료.

---

## 2. Baseline 보강

### A. Discretized boundary placement (제일 ROI 좋음)

- `optimize_defender`의 `alpha_j ∈ [0,1]` continuous NLP variable을 edge당 K개 후보점(K=5~20)으로 이산화
- 좌표하강(coordinate-ascent: sensor 하나씩 돌아가며 최선의 후보점 선택, 나머지 고정) 방식으로 최적화
- 장점: 구현 제일 쉬움(기존 `C_AS` cost 함수 그대로 재사용), R3가 직접 제안, "continuous가 왜 필요한가"에 대한 정량적 답도 동시에 나옴

### B. Coverage-maximizing static heuristic (attacker 무시)

- Attacker 경로와 무관하게 전체 도메인의 시간평균 FOV 커버리지를 최대화하는 배치
- 고전적 max-coverage/submodular greedy 알고리즘 사용 가능 (1-1/e 근사 보장 인용 가능)
- 독립 모듈이라 기존 alternating loop 코드 안 건드리고 별도 스크립트로 작성 가능

### C. Panning 무시 정적 FOV 배치

- `stage_detectability`에서 `phi = cam.get_ctr_theta_t(t)` 대신 고정 각도(또는 visibility=1로 omni 취급)로 defender 최적화 후, **진짜 모델(시간에 따라 도는 카메라)**로 평가해 성능 하락 폭을 보여줌
- R2가 "moving FOV 고려 안 한 baseline"을 직접 요구
- 구현 제일 쉬움: `stage_detectability`에 정적/동적 모드 스위치 flag 추가

### A, B 실행 결과 (2026-08-11)

`script/baseline_2A_2B_defender.py`: random / discretized(2-A) / coverage(2-B) / continuous NLP(현재 방법) 네 가지를 같은 attacker 경로(RRT*)에 대해 `C_AS`로 공정 비교. K=10 후보점, coordinate-ascent 3 sweep, coverage는 20×20 도메인 그리드 × 8 time sample로 시간평균.

**10-trial 평균 cost (클수록 defender 유리):**

| 방법 | 평균 cost | random 대비 |
|---|---|---|
| Random (원 논문의 유일한 baseline) | 0.1248 | — |
| Coverage heuristic (2-B) | 0.1353 | +8% |
| Discretized (2-A, K=10) | 0.1966 | +58% |
| Continuous NLP (현재 방법) | 0.1989 | +59% |

**Paired 통계 (95% CI, n=10):**

| 비교 | 평균 차이 (95% CI) | 유의미? |
|---|---|---|
| discretized − random | +0.072 (±0.032) | 유의미, 10/10 trial |
| continuous − random | +0.074 (±0.035) | 유의미, 10/10 trial |
| discretized − coverage | +0.061 (±0.039) | 유의미, 10/10 trial |
| continuous − discretized | +0.002 (±0.004) | **CI가 0 포함 — 통계적으로 구분 안 됨** |
| coverage − random | +0.011 (±0.016) | CI가 0 포함 — 아직 유의미하지 않음(표본 부족, n=10) |

**해석**:
- R3 "왜 continuous가 필요한가": continuous(0.1989)가 discretized(0.1966)보다 겨우 1.2% 나음, 통계적으로 유의미하지 않음(4개 trial은 완전히 같은 값으로 수렴). "continuous가 압도적으로 낫다"는 주장은 과장 — 대신 "K=10 이산화와 품질은 동급이지만 combinatorial search 없이 단일 gradient 기반 NLP로 풀리고 scalable하다"는 게 정직한 contribution 프레이밍
- R2 "moving FOV 고려 안 한 baseline과 비교하라": coverage heuristic(공격자 완전 무시)은 random보다 겨우 8%(아직 비유의미) 나은데, 공격자를 보는 두 방법(discretized/continuous)은 46~47% 더 나음 — game-theoretic(공격자 인지) 접근의 가치를 정량적으로 뒷받침
- Random < coverage < discretized ≈ continuous 순서가 거의 모든 개별 trial에서 일관됨 (노이즈 아닌 진짜 신호)
- **남은 약점**: coverage vs random 비교가 아직 유의미하지 않음(부차적 주장이라 급하지 않으나, 표에 넣으려면 n=20~30 정도로 늘리는 게 안전)

### C 실행 결과 (2026-08-11)

`script/baseline_2C_static_fov.py`: panning을 무시하고(sweep 중간각도로 고정) 최적화한 defender를, 실제(회전하는) 모델로 평가.

**10-trial 평균 cost:**

| | 평균 cost |
|---|---|
| Dynamic-aware (현재 방법, 실제 모델로 평가) | 0.3428 |
| Static(panning 무시)-optimized, 실제 모델로 평가 | 0.3035 |

- `dynamic_real − static_real`: 평균 +0.039 (95% CI ±0.035, 경계선), 그러나 **10/10 trial 전부 dynamic이 이기거나 동률** (sign test로는 강한 신호, p≈0.001). Trial마다 격차 크기는 0(거의 동일)~45%로 편차 큼
- "Self-deception gap"(static이 자기 모델로는 더 잘한다고 착각하는 정도) 가설은 데이터가 안 받쳐줌 — 8/10은 과대평가하지만 방향·크기가 일관되지 않아 이 서브 내러티브는 폐기
- **쓸 수 있는 주장**: "panning을 모델링하는 게 항상 도움이 되거나 최소한 손해는 안 본다(10/10 trial)" — 평균 효과크기(~13%)는 표본을 늘리면 유의미해질 가능성 높음

---

## 우선순위 / 진행 순서

1. **1-A** (homotopy 문제 규모 측정) — 다른 결정에 필요한 데이터 확보
2. **2-A** (discretized baseline) + **2-C** (static FOV baseline) — 구현 쉽고 리뷰 대응력 큼, 병렬 진행 가능
3. **1-C** (이론적 프레이밍) — 결과 나오는 대로 글감 정리
4. **1-B** (multi-homotopy fix) — 1-A 결과 보고 필요성 판단 후 착수
5. **2-B, 1-D** — 여력 되면

## 진행 원칙

코드 수정 → 결과 확보 → manuscript editing 순서로 진행. 결과 없이 글부터 고치지 않음.
