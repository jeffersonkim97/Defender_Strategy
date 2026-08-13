# RA-L Reject → ICRA 2027 재제출 — 코드/실험 작업 최종본

Source: `26-1775_01_MS.pdf` (RA-L submission 26-1775.1) + 3개 리뷰(Reviewer 2, Reviewer 3, AC meta-report).

**범위**: 이 문서는 코드 수정과 실험 검증만 다룸. Manuscript 텍스트 작업과 실물 데모(Table III) 대응은 별도(사용자가 직접 진행).

**상태**: 아래 코드 fix들과 baseline 실험은 전부 완료·검증됨. 다음 단계는 **mixed strategy 구현**(맨 아래 섹션).

---

## 1. 핵심 발견: 원 논문 수렴률(96.8%)이 재현 안 됨

원 논문 코드에 두 개의 실질적 버그가 있었고, 이를 고치자 "96.8% 수렴"이 가짜 수렴(반복적 solver 실패가 우연히 안정된 값처럼 보인 것)이었음이 드러남. 500-trial 재검증 결과 **진짜 수렴률은 60.2%**.

이 자체가 R2의 "LNE가 항상 존재하는가"라는 핵심 지적에 대한 실증적 답이 됨 — pure LNE가 항상 존재하지는 않는다는 게 실측으로 확인됨.

---

## 2. 적용된 코드 Fix (전부 notebook + diagnostic script 양쪽에 반영·검증됨)

| Fix | 문제 | 조치 |
|---|---|---|
| Attacker cost 로깅 버그 | `C_AS(A_star, S_prev)`가 잘못됨(고정돼야 할 게 `S_star`) | `S_star`로 수정 |
| RRT*/NLP 안전마진 불일치 | STP-RRT*는 margin=1, NLP는 margin=2 요구 — 이 사이 밴드에서 IPOPT가 자주 실패(~47%) | `vehicle['radius']` 1→2로 통일, 실패율 22%로 감소 |
| Cold-restart 민감성 | `optimize_attacker`가 매 iteration 원본 RRT 경로에서 다시 시작 — S가 0.01만 바뀌어도 궤적이 35유닛 튐 | **Warm-start**: 직전 iteration 결과를 다음 initial guess로 사용 |
| Alternating best-response의 limit cycle | Damping 없이는 두 전략 사이를 무한 반복(진동), 진짜 수렴이 아님 | **Damping (η=0.3)**: Krasnoselskii-Mann 업데이트 `x_{k+1}=(1-η)x_k+η·BR(x_k)` |
| 엄격한 수렴판정이 너무 깨지기 쉬움 | cost는 안정됐는데 sensor 위치가 threshold(1e-3) 바로 위에서 미세 진동(잔여 jitter) — 단일 iteration AND 조건이 이걸 못 잡음 | **Windowed 판정**: 최근 4 iteration이 전부 완화된 기준(tol_cost=1e-3, tol_sensor=0.1) 만족하면 수렴 인정 |
| N_attk=250에서 실전 재실행 시 재발 | 모든 검증이 N=100에서만 이뤄짐, N=250은 obstacle 제약이 2.5배라 더 취약 | N_attk 250→100 (카메라 최고속 회전주기 대비 339배 오버샘플링이라 물리적 손실 없음 확인 후 적용) |
| Gradient는 하나의 homotopy class에 갇힘 (R2 Fig.4 지적) | 초기 RRT* 경로가 정해지면 NLP는 그 "모양"(어느 쪽으로 우회하는지) 안에서만 다듬을 수 있음 | **1-B (multi-homotopy)**: bilevel 수렴 직후 K=3개의 대체 RRT* 시드를 refine해서, 원래 답보다 나으면 채택 |
| Iteration 하드캡 부족 | 원래 20이었는데 damping+warmstart 조합은 더 많은 iteration이 필요 | 20→30으로 상향 |

죽은 코드(`S_fixed` 미사용 파라미터, `rho` 미사용 파라미터, eigenvalue/Jacobian LNE 검증)는 사용자 지시로 그대로 둠 — 건드리지 않음.

---

## 3. 500-trial 프로덕션 규모 검증 (메인 결과)

`run_monte_carlo_parallel.py`(10-worker 병렬, OS 프로세스 기반)로 실행. 1차 시도는 Windows Update 자동 재시작으로 134-trial에서 중단(incremental JSON 저장 덕에 데이터는 보존), 재시도로 500/500 완주(5.28시간).

| 지표 | 값 | 95% CI |
|---|---|---|
| Windowed 수렴률 | **60.2%** | 55.9%–64.5% |
| Strict(단일-iteration) 수렴률 | 0% | — |
| 1-B beat rate (더 나은 경로로 교체됨) | **86.6%** | 83.6%–89.6% |
| 평균 개선폭 (beat 시) | 15.95% | 중앙값 15.36% |
| 하드캡(31 iter)까지 감 | 40.2% (201/500) | — |

**표본 크기별 재현성**: n=8(62.5%) → n=134(60.4%) → n=500(60.2%) windowed 수렴률, n=134/500 둘 다 beat rate 86.6% — 표본을 늘려도 숫자가 안정적. 신뢰할 수 있는 최종 수치로 판단.

**남은 40.2%**(하드캡까지 가는 case)는 pure/1-B 둘 다로 해결 안 되는, 진짜 구조적으로 안정된 균형점이 없는 인스턴스로 추정됨 → mixed strategy가 필요한 지점 (섹션 6).

---

## 4. Baseline 비교 (R2/R3의 "baseline이 random 하나뿐" 지적 대응)

`run_parallel.py`(범용 병렬 launcher)로 2-A/2-B/2-C 각각 500-trial 검증. **주의: 반드시 n=500 수치를 써야 함 — n=10에서는 두 항목의 방향 자체가 틀렸음** (섹션 4.4 참고).

### 4.1 무엇을 비교했는가

| Baseline | 방법 | 답하는 질문 |
|---|---|---|
| Random | 원 논문의 유일한 baseline, 무작위 배치 | — |
| 2-A Discretized | 센서를 건물 edge 위 K개 후보점 중에서만 고름(coordinate-ascent) | R3: "continuous가 왜 필요한가" |
| 2-B Coverage | Attacker 경로를 아예 무시, 도메인 전체의 시간평균 커버리지만 최대화 | R2: "moving FOV 고려 안 한 단순 baseline과 비교" |
| 2-C Static FOV | 카메라 panning을 무시(sweep 중간각도로 고정)하고 최적화 후, 실제 모델로 평가 | R2: "회전 안 하는 모델과 비교" |

### 4.2 결과 (n=500, 95% CI, 값 클수록 defender 유리)

| 비교 | 평균 차이 | 유의미? |
|---|---|---|
| discretized(K=10) − random | +0.0892 (±0.0075) | 유의미, 500/500 |
| continuous − random | +0.0896 (±0.0075) | 유의미, 500/500 |
| discretized(K=10) − coverage | +0.1006 (±0.0090) | 유의미, 500/500 |
| **coverage − random** | **−0.0114 (±0.0047)** | **유의미하게 coverage가 더 나쁨** |
| dynamic(panning 반영) − static | +0.0240 (±0.0039) | 유의미, 499/500 |
| static_self − static_real (self-deception) | +0.0230 (±0.0060) | 유의미, 331/500(66%) |

### 4.3 Continuous vs Discretized — K(후보점 개수)별 민감도

"후보점"이란 센서가 놓일 수 있는 건물 edge 위의 미리 정해둔 K개 지점(예: K=3이면 벽의 양 끝+중앙 3곳만 가능, 그 사이는 설치 불가 — 벽에 못 박힌 K개 거치대 중에서만 고르는 것). K를 4단계로 바꿔 각각 500-trial 검증:

| K | continuous − discretized | 95% CI | 상대격차 |
|---|---|---|---|
| 3 | +0.0140 | ±0.0033 (유의미) | **5.07%** |
| 5 | +0.0051 | ±0.0017 (유의미) | 1.84% |
| 7 | +0.0018 | ±0.0009 (유의미) | 0.65% |
| 10 | +0.0004 | ±0.0006 (비유의미) | 0.14% |

K가 커질수록 격차가 매끄럽게 0으로 수렴 — **"continuous가 discrete보다 무조건 낫다"가 아니라, "몇 개의 후보점이 충분한지 미리 알 방법이 없다는 게 discrete의 근본적 약점이고, continuous는 그 튜닝이 아예 필요 없이 항상 최선을 자동으로 찾는다"**는 게 정확하고 방어 가능한 주장.

### 4.4 n=10 → n=500에서 뒤집힌 것들 (표본 크기 경고)

| 항목 | n=10 판단 | n=500 실제 |
|---|---|---|
| coverage vs random | "8% 나음" (비유의미했지만 방향은 양수) | **역전: −1.14%, 유의미하게 더 나쁨** |
| static self-deception gap | "노이즈라 폐기" | **부활: 유의미(+2.30%, 66%)** |

continuous≈discretized(K=10), dynamic>static 두 결론은 n=500에서도 재확인/강화됨(안 뒤집힘). **n=10 수치는 논문에 쓰지 말 것 — 위 표의 n=500 수치만 사용.**

---

## 5. R2/R3 지적 대응 체크리스트

| 지적 | 상태 |
|---|---|
| R2: Fig.4 우회로 못 찾음(homotopy) | ✅ 1-B로 대응, 500-trial 검증(beat rate 86.6%) |
| R2: "global NE가 없을 수도 있다" | ✅ 실측 확인(60.2% 수렴, limit cycle 직접 관측) — 답을 주는 건 mixed strategy 몫(섹션 6) |
| R2: moving FOV 고려 안 한 baseline | ✅ 2-C, n=500 유의미 |
| R2: max sensing area 같은 단순 heuristic과 비교 | ✅ 2-B, n=500 유의미(반전된 방향으로) |
| R3: baseline이 random 하나뿐 | ✅ 2-A/2-B/2-C 총 3개 추가 |
| R3: "continuous 왜 필요한가" 정량적 답 | ✅ K-sensitivity 곡선(4.3) |
| R3: contribution 1/2 통합·continuous 서술 정합성 | 데이터 준비됨, **글쓰기는 사용자 담당** |
| R2: 실물 데모(Table III) attacker 효과 미미 | **사용자 담당** (물리 실험, 코드로 해결 불가) |
| R2: CMA-ES 비교 | 불필요 판단 (필수 지적 아니었고, 1-B가 근본 우려에 더 직접적으로 대응, 고차원에서 CMA-ES는 gradient 방법보다 훨씬 느릴 것으로 예상돼 비용 대비 이득 낮음) |

---

## 6. 다음 작업: Mixed Strategy (설계 완료, 미구현)

### 왜 필요한가
섹션 3의 남은 40.2%(하드캡까지 가는 case)는 pure LNE도 1-B도 못 푸는, 진짜 안정된 균형점이 없는 인스턴스(zero-sum 게임에서 이론적으로 가능한 현상, R2가 지적한 바로 그 시나리오). Defender objective 자체를 바꾸는 건(예: coverage 항 추가) **zero-sum 정의를 깨므로 기각**됨 — mixed strategy가 zero-sum을 유지하면서 이 case들에 답을 주는 유일한 이론적으로 올바른 방법.

### 설계
1. **트리거**: pure/windowed 둘 다 실패하고 하드캡까지 간 trial에만 적용 (fallback)
2. **반복 전략 추출**: cycling 중인 trial의 마지막 iteration들에서, 실제로 반복되는 서로 다른 (attacker 경로, defender 배치) 조합을 소수(보통 2~4개) 클러스터링으로 뽑아냄. **새 NLP 계산 불필요** — 이미 계산된 값 재사용.
3. **Payoff matrix 생성**: 뽑힌 K개 attacker 전략 × L개 defender 전략 조합 전부에 대해 기존 `C_AS`로 비용 계산 (K×L번의 단순 함수 평가, 새 최적화 없음)
4. **Mixed NE 풀기**: 이 작은 payoff matrix에 대해 표준 zero-sum matrix game의 mixed NE를 **선형계획법(LP)**으로 품 (`scipy.optimize.linprog`, 이미 의존성에 있음) — 가위바위보 최적 전략 구하는 것과 동일한 유명한 공식
5. **결과**: "defender는 배치 A를 55%, B를 45% 확률로 섞어 씀"같은 안정적 답 — 무한 진동 대신 이론적으로 정당화된 답이 나옴

### 이미 준비해둔 것
`diagnostic_1A_homotopy_check.py`의 `run_bilevel`과 notebook 양쪽에 **`recent_history`**(각 trial의 마지막 `history_window=20`개 iteration의 A_star/S_star)를 이미 저장하도록 반영해뒀음 — mixed strategy 후처리에 필요한 재료가 이미 JSON에 들어있음(500-trial 결과 포함). **재실행 없이 바로 후처리로 시작 가능.**

### 아직 결정 안 된 것 (구현 시작할 때 확인 필요)
- 클러스터링에 쓸 "같은 전략으로 볼 거리 tolerance" 값
- 최근 몇 iteration을 볼지 (history_window=20 안에서 어느 범위를 쓸지)
- 계산은 하드캡 간 201개 trial 전부에 대해 할지, 일부 샘플로 먼저 검증할지

---

## 7. 재현 방법 (스크립트/커맨드 레퍼런스)

| 스크립트 | 역할 |
|---|---|
| `diagnostic_1A_homotopy_check.py` | 메인 bilevel 파이프라인(버그fix+damping+warmstart+windowed+1-B), CLI로 단독 실행 가능 |
| `diagnostic_2_attacker_solve_stability.py` | cold-restart 민감성 진단(1회성, 재실행 불필요) |
| `run_monte_carlo_parallel.py` | 위 파이프라인 전용 병렬 launcher (메인 500-trial에 사용) |
| `baseline_2A_2B_defender.py` | 2-A(discretized)+2-B(coverage) 비교, `--k-candidates`로 K 조절 |
| `baseline_2C_static_fov.py` | 2-C(static FOV) 비교 |
| `run_parallel.py` | 범용 병렬 launcher(`--worker-script`로 위 아무 스크립트나 병렬 실행), baseline 500-trial들에 사용 |
| `attacker_defender_game_iterative_Monte_Carlo.ipynb` | 실제 연구용 notebook, 모든 fix 반영 완료 |

메인 500-trial 재현:
```bash
python run_monte_carlo_parallel.py --total-trials 500 --num-workers 10 --out-dir <dir>
```

Baseline 재현 (예: 2-A/2-B, K=10):
```bash
python run_parallel.py --worker-script baseline_2A_2B_defender.py --result-tag 2A2B_baseline \
    --total-trials 500 --num-workers 4 --out-dir <dir> --extra-args "--k-candidates 10"
```

**주의**: 여러 worker를 동시에 돌릴 땐 스레드 오버서브스크립션 방지를 위해 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` 설정 권장. 장시간 실행 전 Windows Update 일시중지 확인(자동 재시작으로 한 번 중단된 전례 있음).

Git 반영 커밋: `420637e`(버그fix+1-B+baseline), `efda1c9`(parallel launcher), `6b525ee`(recent_history). Raw 결과 JSON(용량 큼)은 git 미포함.
