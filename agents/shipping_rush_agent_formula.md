# Shipping Rush Agent - 전략 및 로직 정리 문서 (v1.0)

> Puerto Rico 보드게임을 위한 **선적(Shipping) 극대화 전략** 기반 휴리스틱 에이전트의 의사결정 로직을 정리한 문서입니다.
> 구현 코드: [`shipping_rush_agent.py`](./shipping_rush_agent.py)

---

## 📋 목차

0. [에이전트 개요 및 핵심 개념](#0-에이전트-개요-및-핵심-개념)
1. [핵심 상수 및 임계값](#1-핵심-상수-및-임계값)
2. [역할 선택 로직](#2-역할-선택-로직)
3. [Phase별 행동 로직](#3-phase별-행동-로직)
4. [헬퍼 함수](#4-헬퍼-함수)
5. [부록](#5-부록)

---

## 0. 에이전트 개요 및 핵심 개념

### 0.1 이 에이전트는 무엇인가?

**ShippingRushAgent**는 보드게임 **Puerto Rico**(3인 플레이)를 자동으로 플레이하는 **휴리스틱(규칙 기반) AI**입니다.
이름에서 알 수 있듯이, **상품을 생산하고 선적하여 VP를 빠르게 쌓는 전략**에 특화되어 있습니다.

PPO(강화학습) 에이전트의 성능을 평가하기 위한 **Baseline 상대** 중 하나이며,
특히 **선적 중심 전략의 강도**를 테스트하는 데 사용됩니다.

### 0.2 ActionValueAgent와의 비교

ShippingRushAgent는 같은 프로젝트의 ActionValueAgent와 **근본적으로 다른 의사결정 방식**을 사용합니다.

| 비교 항목 | ActionValueAgent | ShippingRushAgent |
|-----------|:---------------:|:-----------------:|
| **의사결정 방식** | `H(s) + bonus(a)` 수식 기반 | **고정 우선순위(priority score)** 기반 |
| **상태 평가** | VP 단위의 정교한 상태 평가 함수 | 없음 (상태 평가 없이 행동별 점수만 비교) |
| **전략 철학** | 범용적 균형 전략 | **선적(Shipping) 극대화** 특화 전략 |
| **상대 인식** | 없음 (자기 상태만 평가) | ✅ 있음 (상대 상품량, 게임 종료 예측) |
| **게임 종료 인식** | progress/decay로 점진적 반영 | 명시적 endgame 플래그로 전략 전환 |
| **행동 선택** | 가장 높은 action_value 선택 | 가장 높은 priority score 선택 |

### 0.3 Priority Score 시스템

ActionValueAgent가 `H(s) + bonus(a)`라는 수식으로 행동을 평가하는 반면,
ShippingRushAgent는 **Priority Score(우선순위 점수)** 시스템을 사용합니다.

```
┌──────────────────────────────────────────────────────────────────┐
│  Priority Score 시스템                                            │
│                                                                  │
│  1. 모든 행동의 기본 점수를 10.0으로 초기화                        │
│  2. Pass(패스)는 1.0, 건물 구매는 0.5로 억제                      │
│  3. 현재 게임 상태에 따라 조건별로 점수를 덮어씀                    │
│     (예: Captain 역할 = 140.0, Harbor 구매 = 225.0 등)            │
│  4. 미세한 랜덤 노이즈(0~0.1) 추가 (동점 방지)                    │
│  5. 불법 행동의 점수를 -∞로 설정                                   │
│  6. 가장 높은 점수의 행동을 선택                                   │
└──────────────────────────────────────────────────────────────────┘
```

> **핵심 차이**: ActionValueAgent는 모든 행동을 VP 단위로 **계산**하지만,
> ShippingRushAgent는 사전에 정의된 **고정 점수**를 조건에 따라 부여합니다.
> 따라서 더 단순하고 빠르지만, 미묘한 상황 판단은 부족할 수 있습니다.

### 0.4 의사결정 흐름

```
┌──────────────────────┐
│  게임 상태 입력       │
│  (obs_dict, mask)    │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  1. 상태 분석         │
│  - 내 자원/건물 확인  │
│  - 상대 상품량 확인   │
│  - 게임 종료 임박?    │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  2. Priority 할당     │
│  - 역할 선택 점수     │
│  - 농장/건물 점수     │
│  - 선적/판매 점수     │
│  - Mayor 배치 점수    │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  3. 노이즈 추가       │
│  + uniform(0, 0.1)   │
│  불법 행동 = -∞       │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  4. argmax(priority)  │
│  → 행동 선택          │
└──────────────────────┘
```

### 0.5 전략 요약: Shipping Rush란?

Shipping Rush 전략의 핵심은 다음과 같습니다:

1. **빠른 생산 체인 구축**: Corn, Indigo 등 생산이 쉬운 상품을 빠르게 생산
2. **선적 극대화**: Harbor, Wharf 건물로 선적 VP를 극대화
3. **Captain 역할 적극 선택**: 상품이 있으면 Captain을 높은 우선순위로 선택
4. **건물 VP보다 선적 VP 중시**: 고가 건물보다 선적 보조 건물(Harbor, Wharf, Warehouse)을 우선 구매
5. **게임 종료 감지**: VP 칩 부족/도시 포화 시 대형 건물로 전환

---

## 1. 핵심 상수 및 임계값

> ShippingRushAgent는 수식 계산 대신 **고정 점수**를 사용합니다.
> 아래 테이블은 각 상황에서 부여되는 점수 값과 그 의도를 정리합니다.

### 1.1 기본 Priority 점수

| 범주 | 기본 점수 | 설명 |
|------|:--------:|------|
| 모든 행동 초기값 | **10.0** | 기본 점수. 특별한 규칙이 없는 행동의 점수 |
| Pass (action 15) | **1.0** | 패스는 가장 낮은 우선순위 (다른 선택이 없을 때만) |
| 건물 구매 (action 16-38) | **0.5** | 기본적으로 억제. 명시적 규칙에 해당하는 건물만 높은 점수 부여 |

### 1.2 역할 선택 점수 범위

> 점수가 높을수록 해당 역할을 선택할 가능성이 높습니다.
> 조건에 따라 같은 역할이라도 다른 점수를 받습니다.

| 역할 | 점수 범위 | 최고 조건 |
|------|:--------:|----------|
| **Captain** | 70 ~ 210+ | 상품 ≥ 2 + Harbor + endgame + 상대 상품 많음 |
| **Builder** | 80 ~ 145 | Wharf 구매 가능할 때 최고 |
| **Craftsman** | 10 ~ 125 | 상품 0개 + 생산력 있을 때 최고 |
| **Settler** | 80 ~ 115 | 초반 (섬 타일 < 4) |
| **Mayor** | 90 ~ 110 | 빈 슬롯 있고 미배치 colonist 없을 때 |
| **Trader** | 75 ~ 105 | 더블론 부족 + Harbor 미보유 |
| **Prospector** | 20 | 고정 (항상 낮은 우선순위) |

### 1.3 게임 종료 임계값

| 조건 | 임계값 | 의미 |
|------|:------:|------|
| VP Critical | `vp_chips ≤ 15` | VP 칩이 15개 이하로 감소 → 게임 곧 종료 |
| City Critical | `max_city_fill ≥ 10` | 어떤 플레이어의 도시가 10/12칸 이상 → 게임 곧 종료 |
| Endgame | VP Critical **OR** City Critical | 둘 중 하나라도 해당되면 종료 임박으로 판단 |

### 1.4 건물 구매 우선순위 (고정 목록)

> ShippingRushAgent는 **선적 관련 건물**을 최우선으로 구매합니다.

| 순위 | 건물 | 점수 | 전략적 이유 |
|:----:|------|:----:|------------|
| 1 | **Wharf** | 230 | 개인 화물선. 어떤 상품이든 자유롭게 선적 가능 |
| 2 | **Harbor** | 225 | 선적 시 +1VP 추가. Shipping Rush의 핵심 건물 |
| 3 | **Large Warehouse** | 220 | 모든 종류 상품 보존. 선적 기회 극대화 |
| 4 | **Small Market** | 215 | 판매 시 +1더블론. 부족한 자금 보충용 |
| 5 | **Small Sugar Mill** | 210 | 저렴한 생산 건물. 빠른 생산 체인 구축 |
| 6 | **Small Indigo Plant** | 205 | 저렴한 생산 건물. 초기 생산력 확보 |
| 7 | **Small Warehouse** | 200 | 1종류 상품 보존. 최소 보험 |

> 게임 종료 임박 시(endgame 또는 city ≥ 8) **대형 건물**(Customs House, Fortress, Guildhall)도 구매 대상에 추가됩니다.

---

## 2. 역할 선택 로직

> 매 라운드 시작 시, ShippingRushAgent는 아래 규칙에 따라 각 역할에 점수를 부여하고
> 가장 높은 점수의 역할을 선택합니다.
> 모든 역할의 기본 점수는 10.0이며, 조건이 충족되면 더 높은 점수로 **덮어씁니다**.

### 2.1 역할별 Priority 규칙

#### ① Captain (action 5) — 선적

> **Shipping Rush 전략의 핵심 역할**. 상품을 배에 실어 VP를 직접 획득합니다.

```
if total_goods ≥ 2:
    base = 140.0
    if has_harbor:          base += 40.0   → 180.0
    if endgame:             base += 30.0   → 최대 210.0
    if opp_wants_captain:   base += 20.0   → 최대 230.0
    priority[Captain] = base

elif total_goods == 1:
    if has_harbor:  priority[Captain] = 100.0
    else:           priority[Captain] = 70.0
```

| 상황 | 점수 | 설명 |
|------|:----:|------|
| 상품 ≥ 2 (기본) | 140 | 선적 가치 충분 |
| + Harbor 활성 | 180 | 선적 시 추가 VP로 가치 상승 |
| + Harbor + endgame | 210 | 남은 VP 칩을 빨리 소진해야 함 |
| + Harbor + endgame + 상대 우위 | 230 | 상대보다 먼저 선적해야 유리 |
| 상품 1개 + Harbor | 100 | 적은 양이지만 Harbor 보너스 |
| 상품 1개 (Harbor 없음) | 70 | 최소한의 선적 가치 |

> **`opp_wants_captain`**: 상대 중 나보다 상품이 많은 플레이어가 있으면 `true`.
> 이 경우 Captain을 내가 먼저 선택해야 선적 순서에서 유리합니다.

#### ② Builder (action 2) — 건물 구매

> 선적 관련 핵심 건물(Wharf, Harbor)을 구매할 수 있을 때만 높은 점수를 받습니다.

```
if empty_city_slots > 0:
    if Wharf 미보유 AND doubloons ≥ Wharf 비용:
        priority[Builder] = 145.0     ← 최고 우선순위
    elif Harbor 미보유 AND doubloons ≥ Harbor 비용:
        priority[Builder] = 135.0
    elif Small Market 미활성 AND doubloons ≥ Small Market 비용:
        priority[Builder] = 80.0
```

| 조건 | 점수 | 설명 |
|------|:----:|------|
| Wharf 구매 가능 | 145 | Shipping Rush의 최고 가치 건물 |
| Harbor 구매 가능 | 135 | 선적 VP 보너스의 핵심 |
| Small Market 구매 가능 | 80 | 자금 보충용 (보조적) |
| 그 외 | 10 (기본) | 특별히 원하는 건물이 없으면 무시 |

#### ③ Craftsman (action 3) — 상품 생산

> 상품이 없을 때 생산을 위해 선택합니다. 생산력이 없으면 오히려 회피합니다.

```
if production_capacity > 0:
    if total_goods == 0:   priority[Craftsman] = 125.0  ← 빈 손일 때 적극 생산
    elif total_goods ≤ 2:  priority[Craftsman] = 100.0  ← 추가 생산 가치 있음
else:
    priority[Craftsman] = 10.0  ← 생산력 0이면 회피 ("자살 Craftsman" 방지)
```

> **"자살 Craftsman"**: 자신은 생산력이 없는데 Craftsman을 선택하면, 상대만 생산하여 나에게 불리합니다.

#### ④ Settler (action 0) — 농장 배치

```
if empty_island_slots > 0:
    if occupied_island < 4:  priority[Settler] = 115.0  ← 초반: 농장 확보 중요
    else:                    priority[Settler] = 80.0   ← 후반: 낮은 우선순위
```

#### ⑤ Mayor (action 1) — colonist 분배

```
if unplaced_colonists > 0:
    priority[Mayor] = 90.0    ← 미배치 colonist 활용
elif has_vacant_slots:
    priority[Mayor] = 110.0   ← 빈 슬롯은 있는데 colonist가 없음 → Mayor로 공급받기
```

> **`has_vacant_slots`**: 건물/농장에 빈 colonist 슬롯이 있는지 확인합니다.
> colonist가 0이어도 빈 슬롯이 있으면 Mayor를 선택하여 colonist 공급을 받는 것이 유리합니다.

#### ⑥ Trader (action 4) — 상품 판매

```
if total_goods > 0 AND trading_house_count < 4:
    if doubloons < 5 AND Harbor 미보유:
        priority[Trader] = 105.0   ← 돈이 부족하고 Harbor가 없으면 판매로 자금 확보
    else:
        priority[Trader] = 75.0    ← 그 외에는 낮은 우선순위
```

> Shipping Rush 전략에서 판매는 **보조적**입니다. 주요 목표는 선적이므로, 자금이 충분하거나 Harbor가 있으면 판매보다 선적을 우선합니다.

#### ⑦ Prospector (action 6, 7)

```
priority[Prospector_1] = 20.0  ← 항상 고정 (낮은 우선순위)
priority[Prospector_2] = 20.0
```

> 1더블론만 얻으므로, 다른 역할보다 항상 후순위입니다.

### 2.2 역할 선택 우선순위 비교 (예시)

```
예시 상황: 상품 3개, Harbor 활성, endgame, 더블론 8개, Wharf 미보유

  Captain:    140 + 40(Harbor) + 30(endgame) = 210
  Builder:    145 (Wharf 구매 가능)
  Craftsman:  10  (상품 3개 > 2 → 기본값 유지)
  Settler:    80  (occupied_island ≥ 4 가정)
  Mayor:      90  (미배치 colonist 가정)
  Trader:     75  (Harbor 보유이므로 낮은 점수)
  Prospector: 20

  → Captain(210) 선택!  상품이 많고 Harbor+endgame으로 선적 가치 극대화
```

---

## 3. Phase별 행동 로직

> 역할이 선택된 후, 해당 Phase에서 구체적으로 어떤 행동을 할지 결정하는 로직입니다.

### 3.1 Settler Phase — 농장 선택 (action 8-14)

> Shipping Rush 전략에서는 **생산이 쉽고 빠른 상품**의 농장을 선호합니다.

선호 농장 순서: `[Corn, Indigo, Sugar, Quarry]`

| 순위 | 농장 (action) | 기본 점수 | 전략적 이유 |
|:----:|-------------|:--------:|------------|
| 특별 | **Quarry** (13-14) | 160 | 건물 할인. 채석장은 항상 최우선 (base + 10.0) |
| 1 | **Corn** (10) | 150 | 건물 없이 바로 생산 가능. 가장 빠른 생산 체인 |
| 2 | **Indigo** (12) | 145 | 생산 건물이 저렴하고 접근이 쉬움 |
| 3 | **Sugar** (11) | 140 | 판매 시 더블론도 확보 가능 |
| 4 | 기타 | 10 (기본) | Coffee, Tobacco는 목록에 없음 → 낮은 우선순위 |

> **참고**: Shipping Rush는 Coffee/Tobacco 같은 고가 상품보다, 빠르게 대량 생산 가능한 Corn/Indigo를 선호합니다.

### 3.2 Builder Phase — 건물 구매 (action 16-38)

§1.4에서 설명한 고정 우선순위 목록에 따라 건물을 구매합니다.

```
일반 건물 우선순위 (점수: 230 ~ 200):
  Wharf(230) > Harbor(225) > Large Warehouse(220) > Small Market(215)
  > Small Sugar Mill(210) > Small Indigo Plant(205) > Small Warehouse(200)

게임 종료 임박 시 대형 건물 추가 (점수: 200 ~ 190):
  Customs House(200) > Fortress(195) > Guildhall(190)
```

> 건물 구매 기본 점수(0.5)가 매우 낮으므로, **위 목록에 없는 건물은 사실상 구매하지 않습니다**.

### 3.3 Captain Phase — 선적 최적화 (action 44-63, 74-78)

> **ShippingRushAgent의 가장 정교한 로직**입니다.
> VP를 극대화하는 최적의 선적 조합을 찾습니다.

#### 최적 선적 알고리즘 (`_get_best_shipping_action`)

모든 합법적인 선적 조합을 탐색하여 **VP 점수가 가장 높은 것**을 선택합니다.

```
for each ship (3척):
    for each good (5종):
        if mask[action] AND 해당 상품 보유 AND 배에 실을 수 있으면:
            load_amount = min(보유량, 배 여유공간)
            vp = load_amount
            if has_harbor:  vp += 1      ← Harbor 보너스
            score = vp × 10 + load_amount ← VP 우선, 동점이면 적재량 우선

for each good (Wharf, 5종):
    if has_wharf AND 미사용 AND mask[action]:
        vp = 보유량
        if has_harbor:  vp += 1
        score = vp × 10 + 보유량 + 5    ← Wharf는 약간의 가산점
```

**점수 계산 공식**:
$$\text{score} = VP \times 10 + \text{load\_amount} (+5 \text{ for Wharf})$$

> VP에 10을 곱하는 이유: VP가 가장 중요한 기준이므로 **가중치를 높여** VP가 높은 선택을 우선하고,
> 동일 VP일 때는 더 많은 양을 적재하는 것을 선호합니다.

#### 선적 Priority 부여

```
best_action의 priority = 320.0 + score    ← 최적 선적은 최고 우선순위
나머지 합법 선적 actions = 280.0           ← 차선은 그래도 높은 우선순위
```

```
예시: Indigo 3개 보유, Harbor 활성
  배1: 여유 2칸, 배2: 여유 4칸 (Indigo 선적 가능)

  배1: load=min(3,2)=2, vp=2+1(Harbor)=3, score=3×10+2=32
  배2: load=min(3,4)=3, vp=3+1(Harbor)=4, score=4×10+3=43  ← 승!

  best_action priority = 320 + 43 = 363
  → 배2에 Indigo 3개 선적 (4VP 획득)
```

### 3.4 Mayor Phase — colonist 배치 (action 120-125, 140-162)

> colonist를 어디에 배치할지 결정합니다.
> **선적 관련 건물에 최우선**으로 배치합니다.

#### 건물 배치 (action 140-162)

| 건물 분류 | 기본 점수 | 해당 건물 |
|----------|:--------:|----------|
| **선적 관련** | 260 | Harbor, Wharf, Small/Large Warehouse |
| **생산 건물** | 240 | Indigo/Sugar Plant, Tobacco Storage, Coffee Roaster |
| **판매 관련** | 235 | Small/Large Market, Office, Factory |
| **기타** | 230 | 나머지 건물 |

> 각 점수에 `uniform(0, 5.0)` 랜덤 노이즈가 추가됩니다.

#### 농장 배치 (action 120-125)

| 농장 타입 | 기본 점수 | 설명 |
|----------|:--------:|------|
| **Coffee, Tobacco, Sugar** | 230 | 고가 상품 생산 농장 우선 |
| **Quarry** | 225 | 건물 할인용 채석장 |
| **기타** (Corn, Indigo) | 220 | 저가 상품 농장 |

### 3.5 Trader Phase — 상품 판매 (action 39-43)

> 비싼 상품부터 우선 판매합니다.

| 상품 | action | 점수 | 판매 가격 |
|------|:------:|:----:|:--------:|
| Coffee | 39 | 132 | 4더블론 |
| Tobacco | 40 | 124 | 3더블론 |
| Corn | 41 | 100 | 0더블론 |
| Sugar | 42 | 116 | 2더블론 |
| Indigo | 43 | 108 | 1더블론 |

> 점수 = `100.0 + good_value × 8` (Coffee=4, Tobacco=3, Sugar=2, Indigo=1, Corn=0)

### 3.6 Craftsman Privilege — 보너스 상품 선택 (action 93-97)

> Craftsman을 선택한 플레이어는 추가 상품 1개를 선택할 수 있습니다.
> 비싼 상품을 우선합니다.

점수 = `80.0 + good_value × 5`

### 3.7 Captain Store Phase — 상품 저장 (action 64-68)

> Captain Phase 종료 후 남은 상품 중 보존할 상품을 선택합니다.

점수 = `55.0 + action_index` (인덱스가 클수록 약간 높은 점수, 사실상 균등)

---

## 4. 헬퍼 함수

> 의사결정 과정에서 사용되는 핵심 보조 함수들입니다.

### 4.1 `_get_game_progress()` — 게임 종료 예측

> 게임이 곧 끝날지 판단하여, 전략을 **선적 가속** 또는 **대형 건물 전환**으로 바꿉니다.

```python
def _get_game_progress(obs_dict, player_idx):
    vp_chips = global_state["vp_chips"]
    
    # 모든 플레이어의 도시 채움 상태 확인
    min_empty_city = 12
    for each player:
        empty = player["empty_city_spaces"]
        min_empty_city = min(min_empty_city, empty)
    max_city_fill = 12 - min_empty_city
    
    return {
        "vp_critical":  vp_chips <= 15,          # VP 칩 부족
        "city_critical": max_city_fill >= 10,     # 도시 거의 포화
        "endgame": vp_critical OR city_critical,  # 종료 임박
    }
```

| 반환값 | 조건 | 영향 |
|--------|------|------|
| `vp_critical` | VP 칩 ≤ 15 | Captain 점수 +30, 대형 건물 구매 활성화 |
| `city_critical` | 최대 도시 채움 ≥ 10/12 | 동일 |
| `endgame` | 둘 중 하나 | Captain/대형 건물 전략 전환 트리거 |

### 4.2 `_opponent_wants_captain()` — 상대 Captain 선호도 분석

> 상대가 나보다 상품을 많이 보유하고 있으면, Captain을 **먼저 선택**하는 것이 유리합니다.
> (Captain Phase에서 역할 선택자가 먼저 행동하므로)

```python
def _opponent_wants_captain(obs_dict, player_idx):
    my_goods = sum(my_state["goods"])
    for each opponent:
        if opponent_total_goods > my_goods:
            return True  # 상대가 더 많은 상품 보유
    return False
```

> 반환값이 `True`이면 Captain 역할 점수에 **+20.0** 가산됩니다.

### 4.3 `_get_best_shipping_action()` — 최적 선적 선택

> §3.3에서 설명한 알고리즘의 상세 구현입니다.
> 3척의 화물선 + Wharf를 탐색하여 VP를 극대화하는 선적을 찾습니다.

**입력:**
- `mask`: 합법 행동 마스크
- `goods`: 보유 상품 배열 [Coffee, Tobacco, Corn, Sugar, Indigo]
- `cargo_ships_good_onehot`: 각 배에 실린 상품 타입 (one-hot, 18차원)
- `cargo_ships_space`: 각 배의 남은 공간 (3차원)
- `has_harbor`, `has_wharf`: 건물 보유 여부

**출력:** `(best_action_id, best_score)`

**점수 공식:**
- 일반 화물선: $\text{score} = (load + harbor\_bonus) \times 10 + load$
- Wharf: $\text{score} = (qty + harbor\_bonus) \times 10 + qty + 5$

### 4.4 `_choose_mayor_strategy()` — Mayor colonist 배치

> §3.4에서 설명한 colonist 배치 우선순위를 구현합니다.
> 건물 분류(선적/생산/판매)에 따라 점수를 부여하고, 농장 타입별로도 별도 점수를 부여합니다.

### 4.5 `_is_active()` / `_has_building()` — 건물 상태 확인

```python
def _has_building(has_building_arr, b_type):
    """건물 보유 여부 (binary 벡터 조회)"""
    return has_building_arr[b_type.value] > 0

def _is_active(has_building_arr, building_colonists_arr, b_type):
    """건물 보유 AND colonist 배치됨 (활성화 상태)"""
    return has_building_arr[b_type.value] > 0 and building_colonists_arr[b_type.value] > 0
```

> `_has_building`: 건물을 소유하고 있는지 (비활성 포함)
> `_is_active`: 건물이 소유 **AND** colonist가 배치되어 효과가 발동되는지

---

## 5. 부록

### 5.1 Priority Score 범위 총정리

> 점수가 높을수록 우선 선택됩니다. 아래는 모든 행동의 점수 범위를 한눈에 보여줍니다.

| 점수 범위 | 행동 | 설명 |
|:---------:|------|------|
| **320+** | Captain 최적 선적 | 가장 높은 우선순위. VP 극대화 선적 |
| **280** | Captain 기타 선적 | 차선 선적 옵션 |
| **230~260** | Mayor colonist 배치 | 건물/농장에 colonist 배치 |
| **200~230** | 건물 구매 (목록 내) | 선적 관련 건물 우선 |
| **150~160** | 농장 선택 | Quarry > Corn > Indigo > Sugar |
| **70~230** | 역할 선택 | 상황에 따라 큰 편차 |
| **100~132** | 상품 판매 | 비싼 상품 우선 |
| **55~68** | 상품 저장 | 균등에 가까움 |
| **10** | 기본값 | 특별 규칙 없는 행동 |
| **1** | Pass | 최후의 수단 |
| **0.5** | 건물 구매 (목록 외) | 사실상 구매 안 함 |
| **-∞** | 불법 행동 | 선택 불가 |

### 5.2 설계 철학 요약

1. **선적 최우선**: Captain 역할과 선적 행동에 가장 높은 점수를 부여하여 VP 직접 획득을 극대화
2. **상대 인식**: 상대의 상품량과 게임 종료 임박 여부를 파악하여 점수를 동적으로 조정
3. **자살 행동 방지**: 생산력이 없으면 Craftsman 회피, 건물이 없으면 Builder 회피
4. **빠른 생산 체인**: Corn/Indigo 중심의 저비용 생산 → 대량 선적 전략
5. **게임 종료 적응**: endgame 감지 시 대형 건물로 전환하여 보너스 VP 확보
6. **단순성**: 복잡한 수식 대신 고정 점수 + 조건 분기로 빠르고 예측 가능한 의사결정

### 5.3 참고

- 구현 코드: [`shipping_rush_agent.py`](./shipping_rush_agent.py)
- 비교 에이전트 문서: [`action_value_agent_formula.md`](./action_value_agent_formula.md)
- 게임 상수 정의: [`configs/constants.py`](../configs/constants.py)
- 게임 규칙 원문: [`rule_documents/puerto-rico-rules-en.pdf`](../rule_documents/puerto-rico-rules-en.pdf)

---

*문서 버전: v1.0 (2026-04-27)*
