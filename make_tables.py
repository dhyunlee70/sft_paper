"""
SFT 논문 결과 테이블 생성 스크립트
======================================
Table 1. SFT(Best) vs Baselines         — 판매량 MAE 기준
Table 2. Architecture Comparison         — Encoder 1-dim / Encoder 3-dim / Enc-Dec 3-dim
Table 3. Modality Ablation               — 5개 모달리티 조합별 성능

실행:
    python3 make_tables.py
"""
import os, math
import numpy as np
import pandas as pd
from datetime import datetime

BASE = os.path.expanduser("~/Library/CloudStorage/OneDrive-개인/phd_article")
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)

# ── 경로 정의 ─────────────────────────────────────────────────
PATHS = {
    "sft_model_results":     os.path.join(BASE, "SFT_model",        "results"),
    "sft_code_results":      os.path.join(BASE, "sft_code",         "results"),
    "sft_transformer_results": os.path.join(BASE, "sft_transformer", "results"),
    "ablation_csv":          os.path.join(BASE, "sft_code",         "results", "ablation_results.csv"),
    "baseline_csv":          os.path.join(BASE, "sft_code",         "results", "baseline_comparison.csv"),
}

# sft_code ablation이 없으면 articles/results_latest 사용
_abl = PATHS["ablation_csv"]
if not os.path.exists(_abl):
    _abl2 = os.path.join(BASE, "articles", "results_latest", "ablation_results.csv")
    if os.path.exists(_abl2):
        PATHS["ablation_csv"] = _abl2

# baseline이 없으면 SFT_model 사용
if not os.path.exists(PATHS["baseline_csv"]):
    PATHS["baseline_csv"] = os.path.join(BASE, "SFT_model", "results", "baseline_comparison.csv")


def load_multiseed(result_dir, target):
    path = os.path.join(result_dir, f"{target}_multiseed_summary.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    m = df[df["seed"] == "mean"]
    s = df[df["seed"] == "std"]
    if m.empty or s.empty:
        return None
    return (float(m["MAE"].values[0]), float(s["MAE"].values[0]),
            float(m["RMSE"].values[0]), float(s["RMSE"].values[0]),
            float(m["MAPE"].values[0]), float(s["MAPE"].values[0]))


def fmt_ms(mean, std, d=4):
    return f"{mean:.{d}f} ±{std:.{d}f}"

def fmt_ms1(mean, std):
    return f"{mean:.1f} ±{std:.1f}"


# ══════════════════════════════════════════════════════════════
# TABLE 1. SFT(Best) vs Baselines
# ══════════════════════════════════════════════════════════════
def make_table1():
    print("\n" + "═"*70)
    print("  TABLE 1. SFT (Best Model) vs Baselines")
    print("  기준: 판매량 MAE(개)  |  방식: 판매율→환산")
    print("═"*70)

    path = PATHS["baseline_csv"]
    if not os.path.exists(path):
        print(f"  [없음] {path}")
        return None

    df = pd.read_csv(path, encoding="utf-8-sig")
    # 판매율→환산 방식만 추출, 정렬
    df = df[df["approach"] == "판매율→환산"].copy()
    df = df.sort_values("MAE_qty").reset_index(drop=True)
    df["Rank"] = df.index + 1

    # SFT 3-seed 평균으로 교체 (더 정확)
    ms = load_multiseed(PATHS["sft_code_results"], "sell_through")
    if ms:
        # sell_through MAE는 판매율 기준 → 판매량 환산값은 predictions로 재계산
        sp = os.path.join(PATHS["sft_code_results"], "sell_through_predictions.csv")
        if os.path.exists(sp):
            sd = pd.read_csv(sp, encoding="utf-8-sig")
            qty_err = (sd["예측판매수량_추정"] - sd["실제판매수량"]).abs()
            sft_mae = round(qty_err.mean(), 1)
            sft_rmse = round(math.sqrt(((sd["예측판매수량_추정"] - sd["실제판매수량"])**2).mean()), 1)
            sft_mape = round((qty_err / sd["실제판매수량"].replace(0, float("nan"))).mean() * 100, 1)
            df.loc[df["model"] == "SFT (Ours)", "MAE_qty"]  = sft_mae
            df.loc[df["model"] == "SFT (Ours)", "RMSE_qty"] = sft_rmse
            df.loc[df["model"] == "SFT (Ours)", "MAPE_qty"] = sft_mape
        df = df.sort_values("MAE_qty").reset_index(drop=True)
        df["Rank"] = df.index + 1

    # 출력
    print(f"  {'순위':>4}  {'모델':<28}  {'MAE(개)':>10}  {'RMSE(개)':>10}  {'MAPE(%)':>10}")
    print("  " + "-"*66)
    for _, row in df.iterrows():
        marker = " ◀ Our Model" if row["model"] == "SFT (Ours)" else ""
        print(f"  {int(row['Rank']):>4}  {row['model']:<28}  "
              f"{row['MAE_qty']:>10.1f}  {row['RMSE_qty']:>10.1f}  {row['MAPE_qty']:>10.1f}{marker}")

    out_path = os.path.join(OUT, "table1_sft_vs_baselines.csv")
    df[["Rank","model","MAE_qty","RMSE_qty","MAPE_qty"]]\
      .rename(columns={"model":"Model","MAE_qty":"MAE(개)","RMSE_qty":"RMSE(개)","MAPE_qty":"MAPE(%)"})\
      .to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  저장: {out_path}")
    return df


# ══════════════════════════════════════════════════════════════
# TABLE 2. Architecture Comparison
# ══════════════════════════════════════════════════════════════
def make_table2():
    print("\n" + "═"*70)
    print("  TABLE 2. Architecture Comparison")
    print("  Encoder 1-dim(9tok) | Encoder 3-dim(5tok) | Enc-Dec 3-dim(5tok)")
    print("═"*70)

    ARCH = [
        ("Encoder, 1-dim (9tok)",    PATHS["sft_model_results"],       False),
        ("Encoder, 3-dim (5tok)",    PATHS["sft_code_results"],         True),
        ("Enc-Dec, 3-dim (5tok)",    PATHS["sft_transformer_results"],  True),
    ]
    rows = []
    for arch_name, result_dir, is_multiseed in ARCH:
        for target in ["sell_through", "sales_qty"]:
            unit = "%p" if target == "sell_through" else "개"
            ms = load_multiseed(result_dir, target) if is_multiseed else None

            if ms:
                mae_str  = fmt_ms(ms[0], ms[1], 4 if target=="sell_through" else 1)
                rmse_str = fmt_ms(ms[2], ms[3], 4 if target=="sell_through" else 1)
                mape_str = fmt_ms1(ms[4], ms[5])
                src = "3-seed"
            else:
                # single predictions
                pred_file = os.path.join(result_dir, f"{target}_predictions.csv")
                if not os.path.exists(pred_file):
                    mae_str = rmse_str = mape_str = "[없음]"; src = "-"
                else:
                    df = pd.read_csv(pred_file, encoding="utf-8-sig")
                    if target == "sell_through":
                        p, a = df["예측판매율"].values.astype(float), df["실제판매율"].values.astype(float)
                    else:
                        p, a = df["예측판매수량"].values.astype(float), df["실제판매수량"].values.astype(float)
                    err  = np.abs(p - a)
                    mae  = err.mean()
                    rmse = math.sqrt(((p-a)**2).mean())
                    mape = (err[a>0] / a[a>0]).mean() * 100
                    d = 4 if target=="sell_through" else 1
                    mae_str  = f"{mae:.{d}f}"
                    rmse_str = f"{rmse:.{d}f}"
                    mape_str = f"{mape:.1f}"
                    src = "single"

            rows.append({
                "Architecture": arch_name,
                "Target":  target,
                f"MAE({unit})": mae_str,
                f"RMSE({unit})": rmse_str,
                "MAPE(%)": mape_str,
                "Source":  src,
            })

    df_out = pd.DataFrame(rows)
    print(f"\n  {'구조':<26}  {'타깃':<14}  {'MAE':>20}  {'RMSE':>20}  {'MAPE(%)':>14}  {'출처'}")
    print("  " + "-"*100)
    for _, row in df_out.iterrows():
        target = row["Target"]
        unit = "%p" if target == "sell_through" else "개"
        print(f"  {row['Architecture']:<26}  {target:<14}  "
              f"{row[f'MAE({unit})']:>20}  {row[f'RMSE({unit})']:>20}  "
              f"{row['MAPE(%)']:>14}  {row['Source']}")

    out_path = os.path.join(OUT, "table2_architecture_comparison.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  저장: {out_path}")
    return df_out


# ══════════════════════════════════════════════════════════════
# TABLE 3. Modality Ablation
# ══════════════════════════════════════════════════════════════
def make_table3():
    print("\n" + "═"*70)
    print("  TABLE 3. Modality Ablation (sell_through, sft_code)")
    print("  5개 모달리티: Image / Text / Collection / Naver / Temporal")
    print("═"*70)

    path = PATHS["ablation_csv"]
    if not os.path.exists(path):
        print(f"  [없음] {path}")
        return None

    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.sort_values("MAE_rate").reset_index(drop=True)
    df["Rank"] = df.index + 1

    # 모달리티 플래그
    MODALITIES = ["image", "text", "coll", "naver", "temporal"]
    for m in MODALITIES:
        df[m.capitalize()] = df["사용모달리티"].apply(
            lambda x: "●" if m in str(x).lower() else "○")

    print(f"\n  {'순위':>4}  {'실험명':<22}  "
          f"{'Image':^7}  {'Text':^7}  {'Coll':^7}  {'Naver':^7}  {'Temp':^7}  "
          f"{'MAE(%p)':>9}  {'RMSE':>9}")
    print("  " + "-"*90)
    for _, row in df.iterrows():
        marker = " ◀ Full" if row["실험"] == "All (SFT Full)" else ""
        print(f"  {int(row['Rank']):>4}  {row['실험']:<22}  "
              f"{row['Image']:^7}  {row['Text']:^7}  {row['Coll']:^7}  "
              f"{row['Naver']:^7}  {row['Temporal']:^7}  "
              f"{row['MAE_pct']:>9.1f}  {row['RMSE_rate']:>9.4f}{marker}")

    cols = ["Rank","실험","사용모달리티","모달리티수","MAE_pct","RMSE_rate"]
    out_path = os.path.join(OUT, "table3_modality_ablation.csv")
    df[cols].rename(columns={"실험":"Experiment","사용모달리티":"Modalities",
                              "모달리티수":"#Mod","MAE_pct":"MAE(%p)","RMSE_rate":"RMSE"})\
            .to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  저장: {out_path}")
    return df


# ══════════════════════════════════════════════════════════════
# 요약 문서 생성
# ══════════════════════════════════════════════════════════════
def make_summary(t1, t2, t3):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# SFT 실험 결과 요약",
        f"생성일시: {ts}",
        "",
        "---",
        "",
        "## Table 1. SFT (Ours) vs Baselines",
        "> 기준: 판매량 MAE(개), 방식: 판매율 예측 → 입고수량 환산",
        "",
    ]
    if t1 is not None:
        lines.append("| 순위 | 모델 | MAE(개) | RMSE(개) | MAPE(%) |")
        lines.append("|------|------|---------|----------|---------|")
        for _, row in t1.iterrows():
            marker = " **← Ours**" if row["model"] == "SFT (Ours)" else ""
            lines.append(f"| {int(row['Rank'])} | {row['model']}{marker} | "
                         f"{row['MAE_qty']:.1f} | {row['RMSE_qty']:.1f} | {row['MAPE_qty']:.1f} |")

    lines += [
        "",
        "---",
        "",
        "## Table 2. Architecture Comparison",
        "> Encoder 1-dim(9tok) vs Encoder 3-dim(5tok) vs Encoder-Decoder 3-dim(5tok)",
        "",
    ]
    if t2 is not None:
        lines.append("| Architecture | Target | MAE | RMSE | MAPE(%) | Source |")
        lines.append("|---|---|---|---|---|---|")
        for _, row in t2.iterrows():
            target = row["Target"]
            unit = "%p" if target == "sell_through" else "개"
            lines.append(f"| {row['Architecture']} | {target} | "
                         f"{row[f'MAE({unit})']} | {row[f'RMSE({unit})']} | "
                         f"{row['MAPE(%)']} | {row['Source']} |")

    lines += [
        "",
        "**핵심 발견:** Encoder 3-dim(5tok) > Enc-Dec 3-dim(5tok) > Encoder 1-dim(9tok)",
        "",
        "---",
        "",
        "## Table 3. Modality Ablation",
        "> sft_code (Encoder, 3-dim), sell_through 기준",
        "",
    ]
    if t3 is not None:
        lines.append("| 순위 | 실험 | 모달리티 | MAE(%p) | RMSE |")
        lines.append("|------|------|----------|---------|------|")
        for _, row in t3.head(13).iterrows():
            lines.append(f"| {int(row['Rank'])} | {row['실험']} | "
                         f"{row['사용모달리티']} | {row['MAE_pct']:.1f} | {row['RMSE_rate']:.4f} |")

    lines += [
        "",
        "---",
        "",
        "## 종합 결론",
        "",
        "1. **최고 성능 모델**: Encoder-only, 3-dim Naver/Collection, sell_through 예측 (sft_code)",
        "2. **판매율→환산 방식**이 판매량 직접 예측보다 전 모델에서 일관되게 우수",
        "3. **Temporal 모달리티**가 가장 중요 (제거 시 성능 저하 최대)",
        "4. **인코더-디코더 구조**는 현재 데이터 규모(5천 개)에서 단순 인코더 대비 성능 이점 없음",
    ]

    out_path = os.path.join(OUT, "paper_results_summary.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  요약 문서 저장: {out_path}")


# ══════════════════════════════════════════════════════════════
def main():
    print("\n" + "█"*70)
    print("  SFT 논문 결과 테이블 생성")
    print("█"*70)
    t1 = make_table1()
    t2 = make_table2()
    t3 = make_table3()
    make_summary(t1, t2, t3)
    print("\n" + "█"*70)
    print(f"  완료! 결과 폴더: {OUT}")
    print("█"*70 + "\n")


if __name__ == "__main__":
    main()
