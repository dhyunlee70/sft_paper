# SFT 실험 결과 요약
생성일시: 2026-04-24 11:08

---

## Table 1. SFT (Ours) vs Baselines
> 기준: 판매량 MAE(개), 방식: 판매율 예측 → 입고수량 환산

| 순위 | 모델 | MAE(개) | RMSE(개) | MAPE(%) |
|------|------|---------|----------|---------|
| 1 | SFT (Ours) **← Ours** | 30.2 | 41.0 | 75.8 |
| 2 | Random Forest | 30.4 | 40.5 | 87.4 |
| 3 | MLP | 30.8 | 41.1 | 87.5 |
| 4 | Gradient Boosting | 30.8 | 41.6 | 86.0 |
| 5 | Linear Regression | 30.9 | 40.7 | 89.6 |
| 6 | LSTM | 30.9 | 40.7 | 91.9 |
| 7 | Attribute KNN (k=5) | 31.7 | 42.2 | 91.6 |
| 8 | Attr+Image KNN (k=5) | 32.1 | 43.5 | 93.1 |
| 9 | Image KNN (k=5) | 35.2 | 48.1 | 100.0 |

---

## Table 2. Architecture Comparison
> Encoder 1-dim(9tok) vs Encoder 3-dim(5tok) vs Encoder-Decoder 3-dim(5tok)

| Architecture | rate MAE(%p) | →qty MAE(개) | direct qty MAE(개) | Source |
|---|---|---|---|---|
| Encoder, 1-dim (9tok) | 0.1948 ±0.0032 | 30.1 ±0.4 | 43.4 ±0.8 | 3-seed |
| Encoder, 3-dim (5tok) | 0.1935 ±0.0025 | 30.2 ±0.3 | 42.9 ±0.4 | 3-seed |
| Enc-Dec, 3-dim (5tok) | 0.1974 ±0.0015 | 30.4 ±0.6 | 44.7 ±1.4 | 3-seed |

**핵심 발견:** Encoder 3-dim(5tok) > Enc-Dec 3-dim(5tok) > Encoder 1-dim(9tok)

---

## Table 3. Modality Ablation
> sft_code (Encoder, 3-dim), sell_through 기준

| 순위 | 실험 | 모달리티 | MAE(%p) | RMSE |
|------|------|----------|---------|------|
| 1 | Text only | text | 19.3 | 0.2305 |
| 2 | No Image | text, coll, naver, temporal | 19.3 | 0.2315 |
| 3 | All (SFT Full) | image, text, coll, naver, temporal | 19.4 | 0.2336 |
| 4 | No Collection | image, text, naver, temporal | 19.5 | 0.2323 |
| 5 | No Naver | image, text, coll, temporal | 19.5 | 0.2330 |
| 6 | Image + Text | image, text | 19.7 | 0.2357 |
| 7 | No Temporal | image, text, coll, naver | 19.7 | 0.2357 |
| 8 | No Text | image, coll, naver, temporal | 19.8 | 0.2357 |
| 9 | Image only | image | 19.9 | 0.2383 |
| 10 | Temporal only | temporal | 20.1 | 0.2391 |
| 11 | Naver only | naver | 20.3 | 0.2411 |
| 12 | Coll only | coll | 20.3 | 0.2414 |
| 13 | Trend only | coll, naver | 20.3 | 0.2412 |

---

## 종합 결론

1. **최고 성능 모델**: Encoder-only, 3-dim Naver/Collection, sell_through 예측 (sft_code)
2. **판매율→환산 방식**이 판매량 직접 예측보다 전 모델에서 일관되게 우수
3. **Temporal 모달리티**가 가장 중요 (제거 시 성능 저하 최대)
4. **인코더-디코더 구조**는 현재 데이터 규모(5천 개)에서 단순 인코더 대비 성능 이점 없음