"""**Module - 설계 원칙**

(1) 단일 책임 원칙 (Single Responsibility Principle)

    각 함수는 “한 가지 역할”만 수행해야 함

(2) 입출력 명확화 (Explicit I/O)

    모든 모듈은 JSON-like 형태의 입력과 출력을 주고받음

(3) 효과 

    유지보수성 : 규정 개정 시 해당 모듈만 교체
    확장성 : Validator 모듈만 교체하면 법규별 자동화 가능
    자동화 : 가정검정 결과에 따라 사후검정 방식 자동 결정
    일관성 : 통계 보고서 표준화 (DataFrame 반환 + plot)
    재사용성 : pipeline 함수 하나로 전체 분석 수행 가능"""

import pandas as pd
from pandas import Series, DataFrame
import numpy as np

#df = pd.read_csv("/content/drive/MyDrive/pharma-data.csv")
#df.head()
#'['Distributor', 'Customer Name', 'City', 'Country', 'Latitude', 'Longitude', 'Channel', 'Sub-channel',
#'Product Name', 'Product Class', 'Quantity', 'Price', 'Sales', 'Month', 'Year', 'Name of Sales Rep',
#'Manager', 'Sales Team']

# ====================================================
# pharma_utils.py
# ====================================================
import pandas as pd
import numpy as np
from scipy.stats import (
    shapiro, levene, ttest_ind, pearsonr, spearmanr, f_oneway
)
from statsmodels.formula.api import ols
import statsmodels.api as sm
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns

# 1️⃣ 데이터 로딩
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


# 2️⃣ 정규성 검정
def test_normality(df, group_col, value_col):
    """샘플 수에 따라 Shapiro 또는 Kolmogorov-Smirnov Test 적용"""
    results = []
    for group, data in df.groupby(group_col):
        sample = data[value_col].dropna()
        if len(sample) < 50:
            stat, p = shapiro(sample)
            method = "Shapiro-Wilk"
        else:
            from scipy.stats import kstest, norm
            stat, p = kstest(sample, "norm", args=(sample.mean(), sample.std()))
            method = "Kolmogorov–Smirnov"
        results.append({"Group": group, "Method": method, "Statistic": stat, "p_value": p})
    return pd.DataFrame(results)


# 3️⃣ 등분산성 검정 (Levene)
def test_homoscedasticity(df, group_col, value_col):
    """Levene’s Test for Equality of Variances"""
    groups = [data[value_col].dropna() for _, data in df.groupby(group_col)]
    stat, p = levene(*groups)
    return {"Levene_stat": stat, "p_value": p, "equal_var": p >= 0.05}


# 4️⃣ t-test (독립표본)
def ttest_groups(df, group_col, value_col, alpha=0.05):
    """Mood Stabilizers vs Analgesics 등 두 그룹 비교"""
    groups = df[group_col].unique()
    if len(groups) != 2:
        raise ValueError("t-test는 두 그룹만 비교 가능합니다.")
    g1, g2 = [df[df[group_col] == g][value_col].dropna() for g in groups]
    stat, p = ttest_ind(g1, g2, equal_var=True)
    return {"t_stat": stat, "p_value": p, "significant": p < alpha}


# 5️⃣ 상관분석
def correlation_analysis(df, col1, col2, method="pearson"):
    """두 연속형 변수 간의 상관관계 분석"""
    if method == "pearson":
        r, p = pearsonr(df[col1], df[col2])
    elif method == "spearman":
        r, p = spearmanr(df[col1], df[col2])
    return {"method": method, "r": r, "p_value": p, "significant": p < 0.05}


# 6️⃣ One-way ANOVA
def one_way_anova(df, group_col, value_col):
    groups = [data[value_col].dropna() for _, data in df.groupby(group_col)]
    stat, p = f_oneway(*groups)
    return {"F": stat, "p_value": p, "significant": p < 0.05}


# 7️⃣ Two-way ANOVA
def two_way_anova(df, dv, between):
    formula = f"{dv} ~ {' + '.join(between)} + {'*'.join(between)}"
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table


# 8️⃣ 반복측정 ANOVA
def repeated_anova(df, dv, within, subject):
    result = pg.rm_anova(dv=dv, within=within, subject=subject, data=df, detailed=True)
    return result


# 9️⃣ 사후검정
def posthoc_analysis(df, group_col, value_col, equal_var=True):
    """Tukey HSD 또는 Games-Howell 자동선택"""
    if equal_var:
        return pg.pairwise_tukey(data=df, dv=value_col, between=group_col)
    else:
        return pg.pairwise_gameshowell(data=df, dv=value_col, between=group_col)
# ====================================================
# pharma_stats_pipeline.py
# ====================================================
from pharma_utils import (
    load_data, test_normality, test_homoscedasticity,
    ttest_groups, correlation_analysis, one_way_anova, two_way_anova,
    repeated_anova, posthoc_analysis, plot_group_comparison
)
import pandas as pd

def pharma_stats_pipeline(
    path, analysis_type="anova",
    group_col=None, value_col=None,
    factors=None, subject_col=None,
    col1=None, col2=None, alpha=0.05
):
    """
    통합 통계 분석 파이프라인
    ---------------------------
    analysis_type: "ttest", "correlation", "anova"
    group_col: 그룹 변수명
    value_col: 종속 변수명
    col1, col2: 상관분석 변수명
    factors: Two-way ANOVA 시 [factor1, factor2]
    subject_col: 반복측정 ANOVA 시 피험자 변수명
    """
    df = load_data(path)
    print(f"데이터 shape: {df.shape}\n")

    results = {}

    # 1️⃣ 가정 검정 (t-test & anova 공통)
    if analysis_type in ["ttest", "anova"]:
        print("=== [정규성 검정] ===")
        norm_result = test_normality(df, group_col, value_col)
        print(norm_result)
        results["normality"] = norm_result

        print("\n=== [등분산성 검정] ===")
        homo_result = test_homoscedasticity(df, group_col, value_col)
        print(homo_result)
        results["homoscedasticity"] = homo_result

    # 2️⃣ 분석 유형별 실행
    if analysis_type == "ttest":
        print("\n=== [t-test 결과] ===")
        res = ttest_groups(df, group_col, value_col)
        results["t_test"] = res
        print(res)

    elif analysis_type == "correlation":
        print("\n=== [상관분석 결과] ===")
        res = correlation_analysis(df, col1, col2, method="pearson")
        results["correlation"] = res
        print(res)

    elif analysis_type == "anova":
        print("\n=== [분산분석 결과] ===")
        if factors is None:
            res = one_way_anova(df, group_col, value_col)
        elif len(factors) == 2:
            res = two_way_anova(df, dv=value_col, between=factors)
        elif subject_col:
            res = repeated_anova(df, dv=value_col, within=group_col, subject=subject_col)
        else:
            raise ValueError("ANOVA 유형을 명확히 지정하세요.")
        results["anova"] = res
        print(res)

        # 3️⃣ 사후검정 (ANOVA만)
        print("\n=== [사후검정 결과] ===")
        posthoc = posthoc_analysis(df, group_col, value_col, equal_var=homo_result["equal_var"])
        print(posthoc)
        results["posthoc"] = posthoc

        # 4️⃣ 시각화
        plot_group_comparison(df, group_col, value_col, title=f"{group_col} vs {value_col}")

    else:
        raise ValueError("analysis_type은 'ttest', 'correlation', 'anova' 중 하나여야 합니다.")

    print("\n=== 분석 완료 ===")
    return results
