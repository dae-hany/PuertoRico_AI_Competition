# Action Value Heuristic Agent - 수식 정리 문서 (v3.0)

> Puerto Rico 보드게임을 위한 휴리스틱 기반 에이전트의 **상태 평가 함수** 및 **액션 보너스 함수** 수식을 정리한 문서입니다.
> 구현 코드: [`action_value_agent.py`](./action_value_agent.py)

---

## 📋 목차

0. [에이전트 개요 및 핵심 개념](#0-에이전트-개요-및-핵심-개념)
1. [핵심 상수](#1-핵심-상수)
2. [메인 휴리스틱 함수: H(s)](#2-메인-휴리스틱-함수-hs)
3. [V_realized (확정 가치)](#3-v_realized-확정-가치)
4. [V_potential (잠재 가치)](#4-v_potential-잠재-가치)
5. [액션 보너스 함수](#5-액션-보너스-함수)
6. [헬퍼 함수](#6-헬퍼-함수)
7. [부록](#7-부록)

---

## 0. 에이전트 개요 및 핵심 개념

### 0.1 이 에이전트는 무엇인가?

**ActionValueAgent**는 보드게임 **Puerto Rico**(3인 플레이)를 자동으로 플레이하는 **휴리스틱(규칙 기반) AI**입니다.
이 에이전트는 PPO(강화학습) 에이전트의 성능을 객관적으로 평가하기 위한 **강력한 Baseline 상대**로 설계되었습니다.

#### 의사결정 과정 (매 턴마다 반복)

```
┌─────────────────────────────────────────────────────────────┐
│  1. 현재 게임 상태(s)에서 합법적인 행동 목록을 받음               │
│  2. 각 합법 행동(a)에 대해 "행동 가치"를 계산:                   │
│     action_value(a) = H(s) + bonus(a)                       │
│       ├── H(s): 현재 상태의 휴리스틱 평가 점수                  │
│       └── bonus(a): 해당 행동을 했을 때 추가되는 보너스          │
│  3. 가장 높은 action_value를 가진 행동을 선택                   │
└─────────────────────────────────────────────────────────────┘
```

#### 평가 함수 구조 (H(s))

```
H(s) = V_realized + V_potential
         │                │
         │                ├── V_goods         (보유 상품의 기대 VP)
         │                ├── V_doubloons     (더블론의 기대 VP)
         │                ├── V_production    (생산력의 미래 기대 VP)
         │                ├── V_commercial    (상업 건물의 미래 기대 VP)
         │                └── V_infrastructure(빈 슬롯의 잠재 VP)
         │
         ├── VP 칩 (이미 획득한 승점)
         ├── 건물 기본 VP (건물에 인쇄된 승점)
         └── 대형 건물 동적 보너스 (게임 종료 시 확정 추가 VP)
```

### 0.2 핵심 용어 사전

이 문서를 읽기 전에 알아야 할 Puerto Rico 게임 용어와 수식 용어입니다.

#### 게임 용어

| 용어 | 설명 |
|------|------|
| **VP (Victory Point)** | 승리 포인트. 게임 종료 시 가장 많은 VP를 가진 플레이어가 승리 |
| **더블론 (Doubloon)** | 게임 내 화폐. 건물 구매에 사용 |
| **역할 (Role)** | 매 라운드 플레이어가 선택하는 행동 카드 (Settler, Builder, Mayor, Craftsman, Trader, Captain) |
| **농장 (Plantation)** | 섬 보드에 배치하는 생산 타일. Colonist를 배치해야 활성화됨 |
| **Colonist** | 농장/건물에 배치하여 활성화하는 일꾼 토큰 |
| **상품 (Goods)** | 농장+건물로 생산하는 물품 (Corn, Indigo, Sugar, Tobacco, Coffee) |
| **선적 (Shipping)** | 상품을 화물선에 실어 VP를 획득하는 행동 |
| **판매 (Trading)** | 상품을 상관에 팔아 더블론을 획득하는 행동 |
| **대형 건물** | 10더블론 건물 (City Hall, Customs House, Fortress, Residence, Guildhall). 게임 종료 시 조건부 보너스 VP 제공 |
| **채석장 (Quarry)** | 건물 구매 시 할인을 제공하는 특수 타일 |

#### 수식 용어

| 용어 | 의미 | 직관적 설명 |
|------|------|------------|
| **H(s)** | 상태 s의 휴리스틱 점수 | "지금 이 상황이 얼마나 좋은가?" |
| **V_realized** | 확정 가치 | "게임이 지금 끝나면 확실히 받는 VP" |
| **V_potential** | 잠재 가치 | "앞으로 행동하면 VP로 바꿀 수 있는 자원의 기대값" |
| **decay** | 감쇠 계수 (0~1) | "게임이 얼마나 남았는가?" (1=초반, 0=종료 직전) |
| **progress** | 진행도 (0~1) | "게임이 얼마나 진행되었는가?" (0=시작, 1=종료) |
| **E[role]** | 역할 기대 사용 횟수 | "남은 게임 동안 특정 역할이 몇 번 더 선택될까?" |
| **P_ship** | 선적 성공 확률 | "내 상품을 성공적으로 배에 실을 확률" |
| **bonus(a)** | 액션 보너스 | "이 행동을 하면 추가로 얻는 가치" |

### 0.3 decay(감쇠)는 왜 필요한가?

잠재 가치(V_potential)는 **"앞으로 VP로 바꿀 수 있는 자원"**을 평가합니다.
하지만 게임이 거의 끝나간다면, 그 자원을 VP로 바꿀 **시간이 부족**합니다.

```
예시:
- 게임 초반 (progress=0.1, decay=0.9): 더블론 4개 → 0.9 VP 기대
  → 건물을 살 시간이 충분하므로 높은 가치
- 게임 후반 (progress=0.8, decay=0.2): 더블론 4개 → 0.2 VP 기대
  → 건물을 살 기회가 거의 없으므로 낮은 가치
- 게임 종료 직전 (progress=1.0, decay=0.0): 더블론 4개 → 0 VP
  → 더 이상 사용할 수 없으므로 가치 없음
```

반면 V_realized(확정 가치)에는 decay를 적용하지 않습니다. VP 칩이나 건물 VP는 게임이 언제 끝나든 확정된 점수이기 때문입니다.

---

## 1. 핵심 상수

> 이 섹션의 상수들은 게임 메커니즘을 VP 단위로 환산하기 위한 변환 계수입니다.
> 모든 가치를 **VP 단위**로 통일함으로써, 서로 다른 자원(더블론, 상품, 건물 등)을 직접 비교할 수 있게 합니다.

### 1.1 환산 상수

| 상수명 | 값 | 설명 | 설계 근거 |
|--------|-----|------|----------|
| `_DOUBLOON_TO_VP` | **0.25** | 더블론 → VP 환산 비율 | 게임에서 평균적으로 4더블론을 투자하면 1VP 상당의 건물을 구매할 수 있음. 따라서 1더블론 ≈ 0.25VP |
| `_SHIPPING_SUCCESS_PROB` | **0.7** | 선적 성공 확률 | 보유 상품이 항상 선적되지는 않음 (배 공간 부족, 다른 플레이어와 경쟁, 강제 폐기 리스크). 보수적으로 70%로 추정 |
| `_TOTAL_ROLE_SELECTIONS` | **51.0** | 총 역할 선택 횟수 | 3인 플레이 기준 약 17라운드 × 3명 = 51회. 게임 전체에서 역할이 총 51번 선택됨 |
| `_NUM_ROLES` | **6.0** | 역할 종류 수 | Settler, Mayor, Builder, Craftsman, Trader, Captain (3인 기준 Prospector 포함 7개 중 실질 6개) |
| `_EXPECTED_ROLE_USES_BASE` | **8.5** | 역할당 기대 선택 횟수 | 51 ÷ 6 ≈ 8.5. 균등 분포 가정 시 각 역할이 게임 전체에서 약 8.5번 선택됨 |
| `COLONIST_DISCOUNT` | **0.5** | colonist 배치 불확실성 할인 | 빈 슬롯에 colonist가 **반드시** 배치되는 것은 아님. Mayor 역할 선택 시기, colonist 부족 등의 불확실성을 반영하여 50% 할인 |

### 1.2 상품 가격표

Puerto Rico에는 5종의 상품이 있으며, 각각 **선적(VP 획득)**과 **판매(더블론 획득)** 두 가지 방법으로 사용할 수 있습니다.

| 상품 | `_GOOD_TRADE_PRICES` | `_GOOD_UNIT_VALUES` | `_GOOD_TRADE_BONUS` | 설명 |
|------|:-------------------:|:-------------------:|:-------------------:|------|
| Coffee | 4 | 1.0 | 0.5 | 가장 비싼 상품. 판매 시 4더블론 획득. 생산이 어려움 (농장+건물+colonist 필요) |
| Tobacco | 3 | 1.0 | 0.4 | 판매 시 3더블론. Coffee 다음으로 생산 난이도 높음 |
| Sugar | 2 | 1.0 | 0.2 | 판매 시 2더블론. 중간 난이도 |
| Indigo | 1 | 1.0 | 0.1 | 판매 시 1더블론. 생산이 비교적 쉬움 |
| Corn | 0 | 1.0 | 0.0 | 판매 불가 (0더블론). 대신 건물 없이 농장만으로 생산 가능 |

- **`_GOOD_UNIT_VALUES`**: 선적 시 상품 1개 = 1VP이므로, 모든 상품의 기본 단위 가치는 동일하게 1.0
- **`_GOOD_TRADE_BONUS`**: 비싼 상품은 판매를 통해 더블론을 벌어 건물을 구매할 수 있는 **유연성**이 있음. 이 추가 가치는 게임 초반에 의미 있고, 후반(decay 적용)에는 감소

### 1.3 상업 건물 능력 가치 (`_COMMERCIAL_ABILITY_VALUES`)

> 상업 건물은 colonist를 1개 이상 배치해야 활성화됩니다.
> 아래 `ability_vp`는 해당 건물이 **1회 사용**될 때 얻는 VP 환산 가치입니다.
> 이 값에 `E[role]`(남은 기대 사용 횟수)을 곱하면 **총 기대 VP**가 됩니다.

| 건물 | ability_vp | 게임 내 실제 효과 | VP 환산 근거 |
|------|:----------:|------------------|-------------|
| Small Market | 0.25 | 판매 시 +1더블론 추가 | 1 × 0.25 = 0.25VP |
| Large Market | 0.50 | 판매 시 +2더블론 추가 | 2 × 0.25 = 0.50VP |
| Office | 0.20 | Trading House에 같은 상품이 있어도 판매 가능 | 중복 판매 허용의 유연성 |
| Harbor | 1.0 | 선적 시 **추가 1VP** 획득 | 직접 1VP 추가이므로 가장 직관적 |
| Wharf | 1.5 | 개인 화물선으로 **어떤 상품이든** 자유롭게 선적 | 배 경쟁 회피 + 모든 상품 선적 가능 → 높은 유연성 |
| Small Warehouse | 0.3 | Captain Phase 후 상품 **1종류** 보존 | 폐기 방지로 미래 VP/더블론 기회 확보 |
| Large Warehouse | 0.5 | Captain Phase 후 상품 **모든 종류** 보존 | 대량 보존으로 더 큰 가치 |
| Factory | **동적** | Craftsman Phase에서 생산 종류 수에 따라 0/1/2/3/5더블론 | `_factory_bonus_value()`로 동적 계산 (§6.1 참조) |
| Hacienda | 0.15 | Settler Phase에서 **추가 농장 1개** 무료 획득 | 간접적 생산력 증가, 효과는 점진적 |
| Construction Hut | 0.15 | Settler Phase에서 채석장 선택 가능 | 채석장 = 건물 할인, 간접적 가치 |
| Hospice | 0.20 | 새 농장/채석장 배치 시 **무료 colonist** 1개 추가 | colonist 절약 가치 |
| University | 0.20 | 새 건물 구매 시 **무료 colonist** 1개 배치 | colonist 절약 가치 |

---

## 2. 메인 휴리스틱 함수: H(s)

## 2. 메인 휴리스틱 함수: H(s)

> **"지금 이 게임 상태가 나에게 얼마나 유리한가?"**를 VP 단위의 숫자로 평가하는 함수입니다.

```
H(s) = V_realized + V_potential
```

- **V_realized** (확정 가치): 게임이 **지금 당장 끝나면** 확실히 얻는 VP. decay 미적용
- **V_potential** (잠재 가치): 미래 행동을 통해 VP로 **전환할 수 있는** 자원의 기대값. decay 적용

### Game Progress (진행도)와 Decay (감쇠)

게임의 진행도는 **3가지 종료 조건** 중 가장 빠르게 진행되는 것을 기준으로 측정합니다.

$$\text{progress} = \max(\text{vp\_progress}, \text{city\_progress}, \text{colonist\_progress})$$

| 진행도 지표 | 수식 | 의미 |
|-------------|------|------|
| vp_progress | $1 - \frac{\text{remaining\_vp}}{\text{initial\_vp}}$ | VP 칩이 얼마나 소진되었는가 |
| city_progress | $\frac{\max(\text{player\_city\_fill})}{12}$ | 가장 많이 건설한 플레이어의 도시 채움 비율 (12칸 만점) |
| colonist_progress | $1 - \frac{\text{remaining\_colonists}}{\text{initial\_colonists}}$ | colonist 풀이 얼마나 소진되었는가 |

$$\text{decay} = \max(0, 1 - \text{progress})$$

```
예시: VP 칩 50% 소진, 도시 6/12 채움, colonist 40% 소진인 경우
  vp_progress = 0.5, city_progress = 0.5, colonist_progress = 0.4
  progress = max(0.5, 0.5, 0.4) = 0.5
  decay = 1 - 0.5 = 0.5  (잠재 가치를 50%로 할인)
```

---

## 3. V_realized (확정 가치)

> **게임이 지금 끝나면 확실히 받는 VP**입니다. 이미 획득하여 되돌릴 수 없는 점수이므로 decay를 적용하지 않습니다.

$$V_{realized} = \text{VP}_{chips} + \sum_{b \in Buildings} \text{VP}_b + V_{large\_active}$$

### 3.1 VP 칩

선적(Shipping) 등으로 획득한 VP 토큰의 합계입니다.

$$\text{VP}_{chips} = p.\text{vp\_chips}$$

### 3.2 건물 기본 VP

도시 보드에 건설된 모든 건물에 인쇄된 VP의 합계입니다. 건물이 활성화되지 않아도(colonist 미배치) 기본 VP는 게임 종료 시 획득합니다.

$$\sum_{b \in Buildings} \text{VP}_b = \sum_{b \in \text{city\_board}} \text{BUILDING\_DATA}[b].\text{vp}$$

### 3.3 활성화된 대형 건물 동적 보너스

> 대형 건물(10더블론)은 게임 종료 시 **조건부 보너스 VP**를 제공합니다.
> 단, colonist가 배치되어 **활성화된 경우에만** 보너스를 받을 수 있습니다.
> 활성화되어 있다면, 이 보너스는 **확정 가치**에 포함됩니다.

조건: `colonists > 0` AND `is_large == True`

| 건물 | 보너스 수식 | 예시 |
|------|------------|------|
| **City Hall** | $N_{violet}$ (보라색 건물 수) | 보라색 건물 5개 → +5VP |
| **Customs House** | $\lfloor \text{VP}_{chips} / 4 \rfloor$ | VP 칩 16개 → +4VP |
| **Fortress** | $\lfloor N_{colonists} / 3 \rfloor$ | colonist 총 12개 → +4VP |
| **Residence** | `_residence_bonus(island_tiles)` | 섬 타일 10개 → +5VP (아래 표 참조) |
| **Guildhall** | $2 \times N_{large\_prod} + 1 \times N_{small\_prod}$ | 대형 생산건물 2개 + 소형 1개 → +5VP |

#### Residence 보너스 테이블

| 섬 타일 수 | ≤9 | 10 | 11 | 12 |
|:---------:|:--:|:--:|:--:|:--:|
| 보너스 VP | 4 | 5 | 6 | 7 |

---

## 4. V_potential (잠재 가치)

> **아직 VP로 확정되지 않았지만, 앞으로의 행동을 통해 VP로 전환할 수 있는 자원**의 기대값입니다.
> 게임 후반으로 갈수록 전환할 시간이 줄어드므로 decay가 적용됩니다.

$$V_{potential} = V_{goods} + V_{doubloons} + V_{production} + V_{commercial} + V_{infrastructure}$$

---

### 4.1 V_goods (보유 상품 가치)

> **"지금 내가 들고 있는 상품을 선적하거나 판매하면 얼마나 얻을까?"**

보유 상품은 두 가지 방법으로 VP에 기여합니다:
1. **선적** → 직접 VP 획득 (확률 P_ship 적용)
2. **판매** → 더블론 획득 → 건물 구매 → 간접적 VP (게임 초반에만 유의미)

$$V_{goods} = \left[ \sum_{g \in Goods} \left( \text{qty}(g) \times 1.0 \times P_{ship} + \text{qty}(g) \times \text{trade\_bonus}(g) \times \text{decay} \right) \right] \times \text{weak\_decay}$$

- $P_{ship} = 0.7$ (선적 성공 확률)
- $\text{weak\_decay} = 0.5 + 0.5 \times \text{decay}$ (상품은 가까운 미래에 사용되므로 **약한 감쇠**만 적용)
- $\text{trade\_bonus}(g)$: 상품별 판매 옵션 가치 (§1.2 참조)

```
계산 예시: Coffee 2개 보유, 게임 중반 (decay=0.5)

  선적 기대값 = 2 × 1.0 × 0.7 = 1.4 VP
  판매 보너스 = 2 × 0.5 × 0.5 = 0.5 VP
  weak_decay  = 0.5 + 0.5 × 0.5 = 0.75
  
  V_goods = (1.4 + 0.5) × 0.75 = 1.425 VP
```

---

### 4.2 V_doubloons (더블론 가치)

> **"보유 더블론으로 건물을 사면 얼마나 VP를 얻을 수 있을까?"**

더블론은 건물 구매에만 사용되며, 게임 후반에는 사용 기회가 줄어듭니다.

$$V_{doubloons} = \text{doubloons} \times 0.25 \times \text{decay}$$

```
예시: 더블론 6개, 게임 초반 (decay=0.9)
  V_doubloons = 6 × 0.25 × 0.9 = 1.35 VP

예시: 더블론 6개, 게임 후반 (decay=0.2)
  V_doubloons = 6 × 0.25 × 0.2 = 0.3 VP
```

---

### 4.3 V_production (생산력 가치)

> **"내 생산 시설이 남은 게임 동안 얼마나 많은 VP를 만들어낼 수 있을까?"**

현재 **활성화된 생산 체인**(colonist가 배치된 농장 + 건물)의 미래 VP 기대값입니다.

$$V_{production} = \sum_{g \in Goods} \text{capacity}(g) \times 1.0 \times E[role] \times P_{ship}$$

- $E[role] = 8.5 \times \text{decay}$ (남은 게임 동안 Craftsman이 선택될 기대 횟수)
- $P_{ship} = 0.7$ (생산된 상품이 선적될 확률)

#### capacity(g) — 상품별 현재 생산 능력

생산 능력은 **활성화된 농장 수**와 **활성화된 건물 슬롯 수** 중 작은 값으로 결정됩니다.
(Corn은 건물이 필요 없으므로 농장 수만 계산)

```python
def _production_capacity(good):
    occupied_plantations = count_occupied_plantations(good)
    
    if good == CORN:
        return occupied_plantations  # Corn은 건물 불필요
    
    building_slots = count_building_colonists(good)
    return min(occupied_plantations, building_slots)
```

```
예시: Sugar 농장 3개(활성 2개) + Sugar Mill(colonist 1개), decay=0.6
  capacity(Sugar) = min(2, 1) = 1
  E[role] = 8.5 × 0.6 = 5.1
  V_production(Sugar) = 1 × 1.0 × 5.1 × 0.7 = 3.57 VP
```

---

### 4.4 V_commercial (활성 상업 건물 가치)

> **"내 상업 건물이 남은 게임 동안 얼마나 VP를 벌어다 줄까?"**

colonist가 배치되어 **활성화된** 상업 건물의 미래 VP 기대값입니다.
1회 사용 시 얻는 `ability_vp`에 남은 기대 사용 횟수를 곱합니다.

$$V_{commercial} = \sum_{b \in \text{Active\_Commercial}} \text{ability\_vp}(b) \times E[role]$$

조건: `colonists > 0` (건물이 활성화된 경우만)

```
예시: Harbor(활성) + Small Market(활성), decay=0.6
  E[role] = 8.5 × 0.6 = 5.1
  V_commercial = (1.0 × 5.1) + (0.25 × 5.1) = 5.1 + 1.275 = 6.375 VP
```

#### Factory 동적 가치

Factory는 Craftsman Phase에서 **생산하는 상품 종류 수**에 따라 더블론을 지급하므로,
현재 생산 능력에 따라 가치가 동적으로 변합니다.

```python
def _factory_bonus_value():
    num_types = count_producible_good_types()
    
    # Factory: 생산 종류에 따라 0/1/2/3/5 더블론
    doubloon_bonus = {
        0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 5
    }.get(num_types, 5)
    
    return doubloon_bonus * 0.25  # VP 환산
```

| 생산 가능 상품 종류 수 | 0-1 | 2 | 3 | 4 | 5 |
|:--------------------:|:---:|:-:|:-:|:-:|:-:|
| 더블론 보너스 | 0 | 1 | 2 | 3 | 5 |
| VP 환산 (×0.25) | 0 | 0.25 | 0.5 | 0.75 | 1.25 |

---

### 4.5 V_infrastructure (인프라 잠재 가치)

> **"colonist만 배치하면 활성화될 수 있는 빈 슬롯은 얼마나 잠재 가치가 있을까?"**

아직 활성화되지 않은(colonist 미배치) 농장/건물이 활성화되었을 때 얻을 수 있는 **추가적 VP 기대값**입니다.
V_production과의 중복을 방지하기 위해, **현재 capacity에 이미 반영된 부분은 제외**하고 **추가로 늘어나는 부분만** 계산합니다.

$$V_{infrastructure} = V_{empty\_plantations} + V_{inactive\_buildings} + V_{inactive\_large}$$

#### (a) 빈 농장의 잠재 생산 가치

빈 농장(colonist 미배치)에 colonist를 배치하면 생산력이 증가할 수 있습니다.
단, **매칭되는 건물 슬롯이 없으면** colonist를 배치해도 생산력이 늘지 않습니다.

$$V_{empty\_plantations} = \sum_{g \in Goods} \text{additional\_capacity}(g) \times 1.0 \times E[role] \times P_{ship} \times \text{COLONIST\_DISCOUNT}$$

```python
def additional_capacity(good):
    occupied = count_occupied_plantations(good)
    unoccupied = count_unoccupied_plantations(good)
    
    if good == CORN:
        return unoccupied  # Corn은 건물 제한 없음
    
    building_slots = count_building_slots(good)
    current_capacity = min(occupied, building_slots)
    headroom = max(0, building_slots - current_capacity)
    
    return min(unoccupied, headroom)
```

```
예시: Sugar 빈 농장 2개, Sugar Mill colonist 슬롯 3개, 현재 occupied 농장 1개
  current_capacity = min(1, 3) = 1
  headroom = max(0, 3 - 1) = 2    ← 건물에 여유 슬롯 2개
  additional_capacity = min(2, 2) = 2  ← 빈 농장 2개 모두 활성화 가능
```

#### (b) 빈 건물 슬롯의 잠재 가치

**생산 건물** (colonist 빈 슬롯이 있는 경우):

매칭 농장이 충분할 때만 추가 생산력이 생깁니다.

$$V = \text{effective\_empty} \times 1.0 \times E[role] \times P_{ship} \times \text{COLONIST\_DISCOUNT}$$

```python
# 유효 빈 슬롯: 빈 건물 슬롯과 매칭 가능한 잉여 농장 중 작은 값
effective_empty = min(
    empty_slots,
    max(0, occupied_plantations - current_building_colonists)
)
```

**상업 건물** (colonists=0, 아직 비활성화된 경우):

첫 colonist 배치로 건물이 활성화되면 앞으로 ability_vp만큼의 가치를 얻습니다.

$$V = \text{ability\_vp}(b) \times E[role] \times \text{COLONIST\_DISCOUNT}$$

#### (c) 비활성 대형 건물

대형 건물은 아직 비활성화 상태이더라도, 활성화되면 높은 보너스 VP를 기대할 수 있습니다.
현재 상태를 기반으로 **보너스를 추정**하되, 미래 성장 가능성(+1~2)을 더해줍니다.

$$V_{inactive\_large} = \text{estimated\_bonus} \times \text{COLONIST\_DISCOUNT}$$

| 건물 | estimated_bonus | 추정 근거 |
|------|:---------------:|----------|
| City Hall | $N_{violet} + 2$ | 현재 보라색 건물 수 + 앞으로 2개 더 지을 것으로 기대 |
| Customs House | $\lfloor \text{VP}_{chips} / 4 \rfloor + 2$ | 현재 VP 칩 기준 + 추가 선적 기대 |
| Fortress | $\lfloor N_{colonists} / 3 \rfloor + 1$ | 현재 colonist 기준 + 추가 colonist 기대 |
| Residence | `_residence_bonus(island_tiles)` | 현재 섬 타일 기준 (이미 거의 확정적) |
| Guildhall | $2 \times N_{large} + 1 \times N_{small} + 2$ | 현재 생산 건물 + 추가 건설 기대 |
| 기타 | 5.0 | 기본 추정치 |

---

## 5. 액션 보너스 함수

> 에이전트는 각 합법 행동(a)에 대해 `action_value(a) = H(s) + bonus(a)`를 계산합니다.
> H(s)는 **현재 상태의 기본 평가**이고, bonus(a)는 **그 행동을 했을 때 추가되는 이득**입니다.
> 모든 합법 행동의 action_value를 비교하여 가장 높은 것을 선택합니다.

$$\text{action\_value} = H(s) + \text{bonus}(a)$$

### 5.1 역할 선택 보너스 (action 0-7)

> 매 라운드 시작 시 플레이어는 6개 역할 중 하나를 선택합니다.
> 선택된 역할은 **모든 플레이어**에게 적용되지만, 선택자에게 **특별 이점(privilege)**이 있습니다.
> 따라서 "이 역할을 내가 선택하면 나에게 얼마나 이득인가?"를 평가합니다.

| 역할 | 보너스 수식 | 게임 맥락 |
|------|------------|----------|
| **Settler** | $0.3 \times \text{decay}$ (if island_space > 0) | 농장 배치. 섬에 빈 공간이 있을 때만 의미 있음 |
| **Mayor** | $\min(\text{empty\_slots}, \text{colonist\_supply}) \times 0.15 \times \text{decay}$ | colonist 분배. 빈 슬롯이 많을수록 이득 |
| **Builder** | $0.5 \times \text{decay}$ (if doubloons ≥ 1) | 건물 구매. 더블론이 있어야 의미 있음 |
| **Craftsman** | $\sum_g \text{capacity}(g) \times 0.3 \times \text{decay}$ | 상품 생산. 생산력이 높을수록 보너스 큼 |
| **Trader** | $\sum_g P_{trade}(g) \times \text{price}(g) \times 0.25$ | 상품 판매. 판매 성공 확률 × 가격으로 기대값 계산 |
| **Captain** | $\sum_g \text{goods}(g) \times 0.4 \times \text{decay}$ | 상품 선적. 보유 상품이 많을수록 유리 |
| **Prospector** | $0.25 \times \text{decay}$ | 1더블론 획득. 고정 보너스 |

**공통 보너스**: 역할 카드에 쌓인 더블론도 추가됩니다 (아무도 선택하지 않은 역할에 매 라운드 1더블론 누적).

$$+\ \text{role\_doubloons} \times 0.25 \times \text{decay}$$

```
예시: Craftsman 선택, Corn 생산력 2 + Indigo 생산력 1, decay=0.7, 카드 위 2더블론
  역할 보너스 = (2+1) × 0.3 × 0.7 = 0.63
  카드 더블론 = 2 × 0.25 × 0.7 = 0.35
  총 보너스 = 0.98
```

---

### 5.2 농장 선택 보너스 (action 8-14)

> Settler Phase에서 어떤 농장/채석장을 가져갈지 결정합니다.

| 타입 | 보너스 수식 | 설명 |
|------|------------|------|
| **Quarry** | $0.8 \times \text{decay}$ | 채석장은 건물 할인을 제공하므로 높은 가치 |
| **Corn** | $(0.3 + 0 \times 0.1) \times \text{decay}$ | Corn은 건물 없이 바로 생산 가능 |
| **기타 (건물 있음)** | $(0.4 + \text{price} \times 0.15) \times \text{decay}$ | 매칭 건물이 있으면 즉시 생산 체인 완성 |
| **기타 (건물 없음)** | $(0.2 + \text{price} \times 0.05) \times \text{decay}$ | 건물 없으면 당장 생산 불가 → 낮은 가치 |

---

### 5.3 건물 구매 보너스 (action 16-38)

> Builder Phase에서 어떤 건물을 구매할지 결정합니다. 건물의 가치는 여러 요소의 합입니다.

$$\text{bonus} = \text{VP}_b + \text{large\_bonus} + \text{production\_bonus} + \text{commercial\_bonus}$$

| 컴포넌트 | 수식 | 설명 |
|----------|------|------|
| VP_b | `BUILDING_DATA[b].vp` | 건물에 인쇄된 기본 VP (확정 가치) |
| large_bonus | $+2.0$ (if is_large) | 대형 건물은 게임 종료 보너스가 크므로 추가 가치 부여 |
| production_bonus | $\text{price}(g) \times 0.25 \times \text{decay}$ | 생산 건물은 비싼 상품일수록 가치 높음 |
| commercial_bonus | $\text{ability\_vp}(b) \times E[role]$ | 상업 건물은 남은 게임 동안 반복 사용 가능 |

> **Factory 특수 처리:** Factory의 commercial_bonus는 현재 생산 가능 상품 종류 수에 따라 동적으로 `_factory_bonus_value()`를 호출합니다.

---

### 5.4 선적 보너스 (action 44-63)

> Captain Phase에서 어떤 상품을 어떤 배에 실을지 결정합니다.
> 선적은 **직접 VP를 획득**하는 행동이므로, 보너스가 직관적입니다.

$$\text{bonus} = \text{qty} \times 1.0 + \text{harbor\_bonus}$$

```python
qty = min(goods[good], ship_capacity - ship_load)  # 실제 실을 수 있는 양
harbor_bonus = qty * 1.0  # Harbor 건물 활성화 시 추가 VP
```

```
예시: Indigo 3개 보유, 배 여유공간 2칸, Harbor 활성
  qty = min(3, 2) = 2
  bonus = 2 × 1.0 + 2 × 1.0 = 4.0  (선적 2VP + Harbor 보너스 2VP)
```

---

### 5.5 저장 보너스 (action 64-68)

> Captain Phase 종료 후 남은 상품 중 **1개를 보존**할 수 있습니다 (Warehouse 미보유 시).
> 비싼 상품을 보존하는 것이 더 가치 있습니다.

$$\text{bonus} = \text{qty} \times \max(0.3, \text{price} \times 0.25 \times 0.6) \times \text{decay}$$

---

### 5.6 판매 보너스 (action 69-73)

> Trader Phase에서 상품을 Trading House에 팔아 더블론을 획득합니다.
> Small/Large Market 건물이 활성화되어 있으면 추가 더블론을 받습니다.

$$\text{bonus} = \text{total\_price} \times 0.25 \times \text{decay}$$

```python
total_price = base_price                         # 상품 기본 가격
if small_market_active: total_price += 1         # Small Market: +1더블론
if large_market_active: total_price += 2         # Large Market: +2더블론
```

```
예시: Coffee 판매, Small Market + Large Market 활성, decay=0.8
  total_price = 4 + 1 + 2 = 7 더블론
  bonus = 7 × 0.25 × 0.8 = 1.4 VP
```

---

### 5.7 Wharf 선적 보너스 (action 74-78)

> Wharf 건물 보유 시, 일반 화물선 대신 **개인 화물선**으로 자유롭게 선적할 수 있습니다.
> 배 경쟁이 없으므로 보유한 상품 전부를 선적할 수 있습니다.

$$\text{bonus} = \text{qty}(g) \times 1.0$$

---

## 6. 헬퍼 함수

> 위 수식들에서 반복적으로 사용되는 보조 함수들입니다.

### 6.1 `_trade_probability(good)` — 판매 성공 확률 추정

> Trader Phase에서 상품을 판매할 때, **나보다 먼저 행동하는 경쟁자**가 같은 상품을 판매하면
> Trading House에 중복 상품이 들어가서 내가 판매할 수 없게 됩니다.
> 이 확률을 추정하여 Trader 역할 선택 보너스에 반영합니다.

$$P_{trade}(g) = \frac{1}{1 + 0.3 \times N_{ahead}}$$

```python
def _trade_probability(good):
    # Office 건물이 있으면 Trading House 중복 제한을 무시할 수 있음
    if good in trading_house and not has_office:
        return 0.0  # 이미 같은 상품이 있으면 판매 불가
    
    # 나보다 턴 순서가 앞인 플레이어 중 해당 상품을 *보유한* 수
    ahead_competitors = count_players_with_good_ahead_in_turn(good)
    
    return 1.0 / (1.0 + ahead_competitors * 0.3)
```

```
예시: Coffee 판매, 나보다 앞 턴에 Coffee 보유 플레이어 2명
  P_trade(Coffee) = 1 / (1 + 0.3 × 2) = 1 / 1.6 = 0.625 (62.5%)
```

---

### 6.2 `_count_empty_slots()` — 빈 colonist 슬롯 수

> Mayor 역할 선택 보너스에서 사용됩니다.
> 빈 슬롯이 많을수록 Mayor를 선택했을 때 더 많은 colonist를 배치할 수 있어 유리합니다.

$$N_{empty} = N_{empty\_plantation} + N_{empty\_building}$$

---

### 6.3 `_expected_role_uses(decay)` — 남은 역할 기대 사용 횟수

> V_production, V_commercial, V_infrastructure 등에서 **"앞으로 이 역할이 몇 번 더 선택될까?"**를 추정합니다.
> 게임이 진행될수록 남은 횟수가 줄어들므로 decay가 적용됩니다.

$$E[role] = \frac{51}{6} \times \text{decay} \approx 8.5 \times \text{decay}$$

```
예시:
  게임 초반 (decay=0.9): E[role] = 8.5 × 0.9 = 7.65회
  게임 중반 (decay=0.5): E[role] = 8.5 × 0.5 = 4.25회
  게임 후반 (decay=0.1): E[role] = 8.5 × 0.1 = 0.85회
```

---

## 📊 요약 테이블

### 잠재 가치 구성요소 (V_potential)

| 구성요소 | 핵심 수식 | decay 적용 | 직관적 의미 |
|----------|----------|-----------|------------|
| V_goods | $\sum qty \times (P_{ship} + \text{trade\_bonus} \times \text{decay}) \times \text{weak\_decay}$ | weak_decay | 들고 있는 상품의 기대 VP |
| V_doubloons | $\text{doubloons} \times 0.25 \times \text{decay}$ | ✅ | 보유 더블론의 기대 VP |
| V_production | $\sum \text{capacity} \times E[role] \times P_{ship}$ | E[role]에 포함 | 활성 생산 체인의 미래 VP |
| V_commercial | $\sum \text{ability\_vp} \times E[role]$ | E[role]에 포함 | 활성 상업 건물의 미래 VP |
| V_infrastructure | 빈 슬롯 × E[role] × P_{ship} × 0.5 | E[role]에 포함 | 비활성 슬롯의 잠재 VP |

### 핵심 파라미터 영향

| 파라미터 | 값 | 영향 범위 | 민감도 | 설명 |
|----------|:--:|----------|:------:|------|
| P_ship | 0.7 | V_goods, V_production, V_infrastructure | **High** | 선적 성공 확률. 낮추면 생산/보유 자원 가치 하락 |
| E[role] | 8.5 | V_production, V_commercial, V_infrastructure | **High** | 역할 기대 횟수. 게임 길이 인식에 직접 영향 |
| COLONIST_DISCOUNT | 0.5 | V_infrastructure | Medium | 빈 슬롯 활성화 불확실성. 낮추면 인프라 과소평가 |
| DOUBLOON_TO_VP | 0.25 | 전체 더블론 관련 | Medium | 더블론-VP 환산 비율. 건물 구매 전략에 영향 |

---

## 7. 부록

### 7.1 Action Space 매핑

> 에이전트의 행동은 **정수 인덱스**로 표현됩니다. 아래 표는 action_idx가 어떤 게임 행동에 대응하는지 보여줍니다.

| Action 범위 | 게임 행동 | Phase |
|:-----------:|----------|:-----:|
| 0 - 7 | 역할 선택 (Settler, Mayor, Builder, Craftsman, Trader, Captain, Prospector×2) | Role Selection |
| 8 - 14 | 농장/채석장 선택 (Coffee, Tobacco, Corn, Sugar, Indigo, Quarry) | Settler |
| 15 | Pass (패스) | 모든 Phase |
| 16 - 38 | 건물 구매 (23종) | Builder |
| 39 - 43 | 상품 판매 선택 (5종) | Trader |
| 44 - 63 | 화물선 선적 (4척 × 5종 상품) | Captain |
| 64 - 68 | 상품 저장 선택 (5종) | Captain Store |
| 74 - 78 | Wharf 선적 (5종 상품) | Captain (Wharf) |
| 120 - 125 | 농장에 colonist 배치 | Mayor |
| 140 - 162 | 건물에 colonist 배치 | Mayor |

### 7.2 설계 철학 요약

1. **모든 가치를 VP 단위로 통일**: 더블론, 상품, 건물 능력 등 이질적인 자원을 VP 환산하여 직접 비교 가능
2. **확정 vs 잠재 분리**: V_realized는 decay 없이 정확히, V_potential은 decay로 시간 가치 반영
3. **보수적 추정**: P_ship=0.7, COLONIST_DISCOUNT=0.5 등 불확실성에 대해 보수적으로 할인
4. **중복 방지**: V_production(현재 활성 capacity)과 V_infrastructure(추가 가능한 capacity)를 분리하여 이중 계산 방지
5. **동적 적응**: Factory, 대형 건물 등은 현재 게임 상태에 따라 가치가 동적으로 변화

### 7.3 참고

- 구현 코드: [`action_value_agent.py`](./action_value_agent.py)
- 게임 상수 정의: [`configs/constants.py`](../configs/constants.py)
- 게임 규칙 원문: [`rule_documents/puerto-rico-rules-en.pdf`](../rule_documents/puerto-rico-rules-en.pdf)

---

*문서 버전: v3.0 (2026-04-27)*
