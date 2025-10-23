#  정규성 검정 (샘플 수에 따라 자동 선택)
def test_normality(data, label):
    """샘플 수에 따라 Shapiro 또는 K-S Test 자동 선택"""
    n = len(data)
    if n < 50:
        stat, p = shapiro(data)
        test = "Shapiro-Wilk"
    else:
        stat, p = kstest((data - np.mean(data)) / np.std(data, ddof=0), 'norm')
        test = "Kolmogorov-Smirnov"
    result = "정규성 만족" if p >= 0.05 else "정규성 위반"
    print(f"[정규성 검정] {label}: ({test}) p={p:.4f} → {result}")
    return p >= 0.05

# 그룹 분할 - 평균/중앙값 자동 선택)
def split_groups(df, base_class, compare_class, value_col):
    """ 왜도(skewness)에 따라 mean 또는 median 자동 기준 선택"""
    base = df[df["Product Class"] == base_class].groupby("City")[value_col].sum().reset_index()
    comp = df[df["Product Class"] == compare_class].groupby("City")[value_col].sum().reset_index()

    skew = base["Sales"].skew()
    if abs(skew) > 1:
        threshold = base["Sales"].median()
        method = "median"
    else:
        threshold = base["Sales"].mean()
        method = "mean"

    print(f"[INFO] {base_class} 기준값 선택: {method.upper()} (skew={skew:.2f}, 기준={threshold:.2f})")

    high_cities = base[base["Sales"] >= threshold]["City"]
    low_cities = base[base["Sales"] < threshold]["City"]

    high_sales = comp[comp["City"].isin(high_cities)]["Sales"]
    low_sales = comp[comp["City"].isin(low_cities)]["Sales"]

    print(f"[INFO] High region={len(high_sales)}개, Low region={len(low_sales)}개")
    return high_sales, low_sales, threshold

#  t-test 수행 (Levene 기반 자동 선택)

def perform_ttest(group1, group2, alpha=0.05):
    """등분산성 검정 후 Student / Welch 자동 선택"""
    # Levene’s Test
    stat_levene, p_levene = levene(group1, group2)
    equal_var = p_levene >= alpha
    levene_result = "등분산 만족 " if equal_var else "등분산 위반 "
    print(f"[Levene Test] p={p_levene:.4f} → {levene_result}")

    # T-test 선택
    t_stat, p_val = ttest_ind(group1, group2, equal_var=equal_var)
    test_type = "Student’s t-test" if equal_var else "Welch’s t-test"
    sig = "통계적으로 유의함" if p_val < alpha else "통계적으로 유의하지 않음"
    print(f"[{test_type}] t={t_stat:.3f}, p={p_val:.4f} → {sig}")

    return t_stat, p_val, test_type

# --------------------------------------------------------------
# 6️⃣ 전체 파이프라인
# --------------------------------------------------------------

    # 그룹 분할 (자동 기준 선택)
    high_sales, low_sales, threshold = split_groups(df, base_class, compare_class, value_col)

    # 정규성 검정
    test_normality(high_sales, "High region")
    test_normality(low_sales, "Low region")
