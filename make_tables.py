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

_ONEDRIVE_BASE = os.path.expanduser("~/Library/CloudStorage/OneDrive-개인/phd_article")

# Cowork 세션 환경 자동 감지: SFT_model과 articles가 /mnt/ 하위에 마운트된 경우
def _detect_base():
    # 환경변수 오버라이드 우선
    if os.environ.get("SFT_PAPER_BASE"):
        return os.environ["SFT_PAPER_BASE"]
    # Cowork session: 두 폴더가 /mnt/ 하위에 직접 마운트됨
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _mnt_candidate = os.path.join(_script_dir, "..")  # sft_paper → articles
    if os.path.exists(os.path.join(_mnt_candidate, "results_latest")):
        # articles 폴더 안에서 실행 중 → 부모를 가상 BASE로 사용
        return os.path.abspath(os.path.join(_mnt_candidate, ".."))
    return _ONEDRIVE_BASE

BASE = _detect_base()
_now = datetime.now()
_script_dir = os.path.dirname(os.path.abspath(__file__))
OUT        = os.path.join(_script_dir, "output")                                      # 최신본 (덮어쓰기)
OUT_STAMP  = os.path.join(_script_dir, "output", _now.strftime("%Y-%m-%d-%H%M"))      # 타임스탬프 백업
os.makedirs(OUT,       exist_ok=True)
os.makedirs(OUT_STAMP, exist_ok=True)

# ── 경로 정의 ─────────────────────────────────────────────────
# sft_code 실체: articles/results_latest (Encoder 3-dim 5tok, 3-seed)
# sft_transformer 결과: articles/results_latest에 없으므로 별도 경로 확인
_articles = os.path.join(BASE, "articles", "results_latest")
_sft_model = os.path.join(BASE, "SFT_model", "results")
_sft_transformer = os.path.join(BASE, "sft_transformer", "results")

PATHS = {
    "sft_model_results":       _sft_model,
    "sft_code_results":        _articles,          # articles/results_latest = sft_code 결과
    "sft_transformer_results": _sft_transformer,
    "ablation_csv":            os.path.join(_articles,  "ablation_results.csv"),
    "baseline_csv":            os.path.join(_sft_model, "baseline_comparison.csv"),
}

# ablation 3-seed 결과 탐색 순서:
#   1) sft_code/results/  (ablation_multiseed.py를 sft_code에서 실행한 결과)
#   2) articles/results_latest/
#   3) SFT_model/results/
_sft_code_results = os.path.join(BASE, "sft_code", "results")
PATHS["ablation_multiseed_csv"] = ""
for _abl_candidate in [
    os.path.join(_sft_code_results, "ablation_multiseed_results.csv"),
    os.path.join(_articles,         "ablation_multiseed_results.csv"),
    os.path.join(_sft_model,        "ablation_multiseed_results.csv"),
]:
    if os.path.exists(_abl_candidate):
        PATHS["ablation_multiseed_csv"] = _abl_candidate
        break


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
    # multiseed predictions 파일 탐색 (우선: _multiseed, 없으면 단일)
    for _sp_name in ["sell_through_predictions_multiseed.csv", "sell_through_predictions.csv"]:
        _sp = os.path.join(PATHS["sft_code_results"], _sp_name)
        if os.path.exists(_sp):
            break
    else:
        _sp = None
    if _sp:
        sd = pd.read_csv(_sp, encoding="utf-8-sig")
        # multiseed 파일은 seed별 rows 포함 → seed별 평균 계산
        if "seed" in sd.columns:
            # 각 style의 seed별 예측 평균으로 집계
            sd = sd.groupby("style", as_index=False).agg(
                입고수량=("입고수량", "first"),
                실제판매수량=("실제판매수량", "first"),
                예측판매수량_추정=("예측판매수량_추정", "mean"),
            )
        qty_err = (sd["예측판매수량_추정"] - sd["실제판매수량"]).abs()
        sft_mae  = round(qty_err.mean(), 1)
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
    out_df = df[["Rank","model","MAE_qty","RMSE_qty","MAPE_qty"]].copy()
    for col in ["MAE_qty","RMSE_qty","MAPE_qty"]:
        out_df[col] = out_df[col].apply(lambda x: round(float(x), 1))
    renamed = out_df.rename(columns={"model":"Model","MAE_qty":"MAE(개)","RMSE_qty":"RMSE(개)","MAPE_qty":"MAPE(%)"})
    renamed.to_csv(out_path, index=False, encoding="utf-8-sig")
    renamed.to_csv(os.path.join(OUT_STAMP, "table1_sft_vs_baselines.csv"), index=False, encoding="utf-8-sig")
    print(f"\n  저장: {out_path}")
    return df


# ══════════════════════════════════════════════════════════════
# TABLE 2. Architecture Comparison
# ══════════════════════════════════════════════════════════════
def _qty_from_sell_through(result_dir, is_multiseed):
    """sell_through 예측 파일에서 환산 판매량 MAE (mean, std) 계산.
    multiseed 파일이 있으면 seed별 MAE → mean/std, 아니면 (single, 0.0) 반환."""
    # 파일 탐색 순서: multiseed 우선
    ms_file = os.path.join(result_dir, "sell_through_predictions_multiseed.csv")
    sg_file = os.path.join(result_dir, "sell_through_predictions.csv")

    if is_multiseed and os.path.exists(ms_file):
        df = pd.read_csv(ms_file, encoding="utf-8-sig")
        if "seed" in df.columns:
            seed_maes, seed_rmses = [], []
            for seed, g in df.groupby("seed"):
                err = (g["예측판매수량_추정"] - g["실제판매수량"]).abs()
                seed_maes.append(err.mean())
                seed_rmses.append(math.sqrt(((g["예측판매수량_추정"] - g["실제판매수량"])**2).mean()))
            return (float(np.mean(seed_maes)), float(np.std(seed_maes)),
                    float(np.mean(seed_rmses)), float(np.std(seed_rmses)))
    if os.path.exists(sg_file):
        df = pd.read_csv(sg_file, encoding="utf-8-sig")
        err = (df["예측판매수량_추정"] - df["실제판매수량"]).abs()
        mae  = float(err.mean())
        rmse = float(math.sqrt(((df["예측판매수량_추정"] - df["실제판매수량"])**2).mean()))
        return (mae, 0.0, rmse, 0.0)
    return None


def make_table2():
    print("\n" + "═"*70)
    print("  TABLE 2. Architecture Comparison")
    print("  Encoder 1-dim(9tok) | Encoder 3-dim(5tok) | Enc-Dec 3-dim(5tok)")
    print("  ※ sell_through→qty: 판매율 예측 × 입고수량 환산  /  direct qty: 판매량 직접 예측")
    print("═"*70)

    ARCH = [
        ("Encoder, 1-dim (9tok)",    PATHS["sft_model_results"],       True),
        ("Encoder, 3-dim (5tok)",    PATHS["sft_code_results"],         True),
        ("Enc-Dec, 3-dim (5tok)",    PATHS["sft_transformer_results"],  True),
    ]
    rows = []
    for arch_name, result_dir, is_multiseed in ARCH:
        # ── sell_through rate MAE (%p) ──────────────────────────
        ms_rate = load_multiseed(result_dir, "sell_through") if is_multiseed else None
        if ms_rate:
            rate_mae_str  = fmt_ms(ms_rate[0], ms_rate[1], 4)
            rate_rmse_str = fmt_ms(ms_rate[2], ms_rate[3], 4)
            src = "3-seed"
        else:
            pred_f = os.path.join(result_dir, "sell_through_predictions.csv")
            if os.path.exists(pred_f):
                df = pd.read_csv(pred_f, encoding="utf-8-sig")
                p, a = df["예측판매율"].values.astype(float), df["실제판매율"].values.astype(float)
                err = np.abs(p - a)
                rate_mae_str  = f"{err.mean():.4f}"
                rate_rmse_str = f"{math.sqrt(((p-a)**2).mean()):.4f}"
                src = "single"
            else:
                rate_mae_str = rate_rmse_str = "[없음]"; src = "-"

        # ── sell_through → 환산 qty MAE (개) ───────────────────
        qty_conv = _qty_from_sell_through(result_dir, is_multiseed)
        if qty_conv:
            m, s, rm, rs = qty_conv
            qty_conv_mae_str  = fmt_ms(m, s, 1) if s > 0 else f"{m:.1f}"
            qty_conv_rmse_str = fmt_ms(rm, rs, 1) if rs > 0 else f"{rm:.1f}"
        else:
            qty_conv_mae_str = qty_conv_rmse_str = "[없음]"

        # ── direct sales_qty MAE (개) ───────────────────────────
        ms_qty = load_multiseed(result_dir, "sales_qty") if is_multiseed else None
        if ms_qty:
            qty_dir_mae_str  = fmt_ms(ms_qty[0], ms_qty[1], 1)
            qty_dir_rmse_str = fmt_ms(ms_qty[2], ms_qty[3], 1)
        else:
            pred_f2 = os.path.join(result_dir, "sales_qty_predictions.csv")
            if os.path.exists(pred_f2):
                df2 = pd.read_csv(pred_f2, encoding="utf-8-sig")
                p2, a2 = df2["예측판매수량"].values.astype(float), df2["실제판매수량"].values.astype(float)
                err2 = np.abs(p2 - a2)
                qty_dir_mae_str  = f"{err2.mean():.1f}"
                qty_dir_rmse_str = f"{math.sqrt(((p2-a2)**2).mean()):.1f}"
            else:
                qty_dir_mae_str = qty_dir_rmse_str = "[없음]"

        rows.append({
            "Architecture":        arch_name,
            "MAE_rate(%p)":        rate_mae_str,
            "RMSE_rate(%p)":       rate_rmse_str,
            "MAE_conv_qty(개)":    qty_conv_mae_str,
            "RMSE_conv_qty(개)":   qty_conv_rmse_str,
            "MAE_direct_qty(개)":  qty_dir_mae_str,
            "RMSE_direct_qty(개)": qty_dir_rmse_str,
            "Source":              src,
        })

    # [없음] 행은 기존 저장된 결과로 채우기 (다른 세션/머신에서 생성된 값 보존)
    existing_t2 = os.path.join(OUT, "table2_architecture_comparison.csv")
    if os.path.exists(existing_t2):
        ex = pd.read_csv(existing_t2, encoding="utf-8-sig")
        for i, row in enumerate(rows):
            arch = row["Architecture"]
            for col in ["MAE_rate(%p)","RMSE_rate(%p)","MAE_conv_qty(개)","RMSE_conv_qty(개)",
                        "MAE_direct_qty(개)","RMSE_direct_qty(개)"]:
                if str(row.get(col, "[없음]")).strip() in ("[없음]", "", "nan"):
                    m_ex = ex[ex["Architecture"] == arch]
                    if not m_ex.empty and col in m_ex.columns:
                        v = str(m_ex.iloc[0][col]).strip()
                        if v not in ("[없음]", "", "nan"):
                            rows[i][col] = v
                            rows[i]["Source"] = m_ex.iloc[0].get("Source", rows[i]["Source"])

    df_out = pd.DataFrame(rows)

    # 출력
    print(f"\n  {'구조':<26}  {'rate MAE(%p)':>18}  "
          f"{'→qty MAE(개)':>18}  {'direct qty MAE(개)':>20}  {'출처'}")
    print("  " + "-"*95)
    for _, row in df_out.iterrows():
        print(f"  {row['Architecture']:<26}  {str(row['MAE_rate(%p)']):>18}  "
              f"{str(row['MAE_conv_qty(개)']):>18}  {str(row['MAE_direct_qty(개)']):>20}  {row['Source']}")

    out_path = os.path.join(OUT, "table2_architecture_comparison.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    df_out.to_csv(os.path.join(OUT_STAMP, "table2_architecture_comparison.csv"), index=False, encoding="utf-8-sig")
    print(f"\n  저장: {out_path}")
    return df_out


# ══════════════════════════════════════════════════════════════
# TABLE 3. Modality Ablation
# ══════════════════════════════════════════════════════════════
def make_table3():
    print("\n" + "═"*70)
    print("  TABLE 3. Modality Ablation (sell_through, sft_code, Encoder 3-dim)")
    print("  5개 모달리티: Image / Text / Collection / Naver / Temporal")
    print("═"*70)

    # 3-seed 결과 우선, 없으면 single-seed fallback
    ms_path  = PATHS.get("ablation_multiseed_csv", "")
    s1_path  = PATHS["ablation_csv"]
    is_multiseed = False

    if os.path.exists(ms_path):
        df = pd.read_csv(ms_path, encoding="utf-8-sig")
        df = df.sort_values("MAE_mean").reset_index(drop=True)
        is_multiseed = True
        print("  ✅ 3-seed 평균 결과 사용")
    elif os.path.exists(s1_path):
        df = pd.read_csv(s1_path, encoding="utf-8-sig")
        df = df.sort_values("MAE_rate").reset_index(drop=True)
        # single-seed 컬럼 통일
        df["MAE_mean"]   = df["MAE_rate"]
        df["MAE_std"]    = 0.0
        df["MAE_pct"]    = df["MAE_pct"]
        df["RMSE_mean"]  = df["RMSE_rate"]
        df["RMSE_std"]   = 0.0
        print("  ⚠️  single-seed 결과 사용 (3-seed 실행 후 재생성 권장)")
    else:
        print(f"  [없음] ablation 결과 파일 없음")
        return None

    df["Rank"] = df.index + 1

    MODALITIES = ["image", "text", "coll", "naver", "temporal"]
    for m in MODALITIES:
        df[m.capitalize()] = df["사용모달리티"].apply(
            lambda x: "●" if m in str(x).lower() else "○")

    mae_header = "MAE(%p) ±std" if is_multiseed else "MAE(%p)"
    print(f"\n  {'순위':>4}  {'실험명':<22}  "
          f"{'Image':^7}  {'Text':^7}  {'Coll':^7}  {'Naver':^7}  {'Temp':^7}  "
          f"{mae_header:>16}  {'RMSE':>9}")
    print("  " + "-"*95)
    for _, row in df.iterrows():
        marker = " ◀ Full" if row["실험"] == "All (SFT Full)" else ""
        if is_multiseed:
            mae_str = f"{row['MAE_pct']:>5.1f} ±{row['MAE_pct_std']:>4.1f}"
        else:
            mae_str = f"{row['MAE_pct']:>5.1f}      "
        print(f"  {int(row['Rank']):>4}  {row['실험']:<22}  "
              f"{row['Image']:^7}  {row['Text']:^7}  {row['Coll']:^7}  "
              f"{row['Naver']:^7}  {row['Temporal']:^7}  "
              f"{mae_str:>16}  {row['RMSE_mean']:>9.4f}{marker}")

    # 저장
    save_cols = ["Rank","실험","사용모달리티","모달리티수","MAE_pct","RMSE_mean"]
    rename_map = {"실험":"Experiment","사용모달리티":"Modalities",
                  "모달리티수":"#Mod","MAE_pct":"MAE(%p)","RMSE_mean":"RMSE"}
    if is_multiseed:
        save_cols += ["MAE_pct_std","RMSE_std"]
        rename_map.update({"MAE_pct_std":"MAE_std(%p)","RMSE_std":"RMSE_std"})

    out_path = os.path.join(OUT, "table3_modality_ablation.csv")
    renamed_t3 = df[save_cols].rename(columns=rename_map)
    renamed_t3.to_csv(out_path, index=False, encoding="utf-8-sig")
    renamed_t3.to_csv(os.path.join(OUT_STAMP, "table3_modality_ablation.csv"), index=False, encoding="utf-8-sig")

    # 매트릭스 CSV — 컬럼명: capitalize() 기준 (Image, Text, Coll, Naver, Temporal)
    COL_LABELS = [("Image","image"),("Text","text"),("Coll","coll"),
                  ("Naver","naver"),("Temporal","temporal")]
    matrix_rows = []
    for _, row in df.iterrows():
        r = {"실험": row["실험"]}
        for label, _ in COL_LABELS:
            r[label] = row[label]
        r["MAE(%p)"] = row["MAE_pct"]
        if is_multiseed:
            r["MAE_std(%p)"] = row["MAE_pct_std"]
        r["RMSE"] = row["RMSE_mean"]
        matrix_rows.append(r)
    matrix_df = pd.DataFrame(matrix_rows)
    matrix_out = os.path.join(OUT, "table3_modality_matrix.csv")
    matrix_df.to_csv(matrix_out, index=False, encoding="utf-8-sig")
    matrix_df.to_csv(os.path.join(OUT_STAMP, "table3_modality_matrix.csv"), index=False, encoding="utf-8-sig")

    src_note = "3-seed" if is_multiseed else "single-seed"
    print(f"\n  저장: {out_path}  ({src_note})")
    print(f"  매트릭스: {matrix_out}")
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
        lines.append("| Architecture | rate MAE(%p) | →qty MAE(개) | direct qty MAE(개) | Source |")
        lines.append("|---|---|---|---|---|")
        for _, row in t2.iterrows():
            lines.append(f"| {row['Architecture']} | "
                         f"{row.get('MAE_rate(%p)', '-')} | "
                         f"{row.get('MAE_conv_qty(개)', '-')} | "
                         f"{row.get('MAE_direct_qty(개)', '-')} | "
                         f"{row.get('Source', '-')} |")

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
            rmse_val = row.get("RMSE_mean", row.get("RMSE_rate", 0))
            lines.append(f"| {int(row['Rank'])} | {row['실험']} | "
                         f"{row['사용모달리티']} | {row['MAE_pct']:.1f} | {float(rmse_val):.4f} |")

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
    content = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    with open(os.path.join(OUT_STAMP, "paper_results_summary.md"), "w", encoding="utf-8") as f:
        f.write(content)
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
