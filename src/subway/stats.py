"""역·방향 단위 대응 비교에 쓰는 통계 함수."""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def paired_summary(a: pd.Series, b: pd.Series, n_boot: int = 5000, seed: int = 0) -> dict:
    """같은 역·방향에서 잰 두 값(a, b)을 비교한다. 차이는 a - b.

    혼잡도 차이는 정규분포가 아니라서 Wilcoxon 부호순위 검정을 주 검정으로 쓰고,
    차이의 크기는 중앙값과 부트스트랩 95% 신뢰구간으로 보고한다.
    """
    pair = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    d = (pair["a"] - pair["b"]).to_numpy()

    rng = np.random.default_rng(seed)
    boot = np.median(rng.choice(d, size=(n_boot, len(d)), replace=True), axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    return {
        "n": len(d),
        "평균 차이": d.mean(),
        "중앙값 차이": np.median(d),
        "95% CI 하한": lo,
        "95% CI 상한": hi,
        "a가 더 낮은 비율": (d < 0).mean(),
        "효과크기 dz": d.mean() / d.std(ddof=1),
        "Wilcoxon p": stats.wilcoxon(d).pvalue,
    }


def compare_pairs(table: pd.DataFrame, pairs: list[tuple[str, str]], **kwargs) -> pd.DataFrame:
    """table의 열 쌍마다 paired_summary를 구하고 Holm 방법으로 다중비교를 보정한다."""
    rows = [{"비교": f"{a} - {b}", **paired_summary(table[a], table[b], **kwargs)} for a, b in pairs]
    out = pd.DataFrame(rows).set_index("비교")
    out["Holm 보정 p"] = multipletests(out["Wilcoxon p"], method="holm")[1]
    return out
