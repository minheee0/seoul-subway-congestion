"""EDA에서 확인한 규칙으로 혼잡도 데이터를 정제한다."""

import pandas as pd

from subway.config import STATION_ALIASES

# 성수·신도림은 본선과 지선 출발 행이 역명은 같고 역번호만 달라서 역번호까지 키로 쓴다.
SERIES_KEYS = ["day_type", "line", "station_code", "station", "direction"]


def find_structural_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """하루 종일 0인 역·방향 조합. 종착역에서 더 갈 곳이 없는 방향이라 관측값이 아니다."""
    zero_share = df.groupby(SERIES_KEYS)["congestion"].apply(lambda s: (s == 0).mean())
    return zero_share[zero_share == 1].reset_index()[SERIES_KEYS]


def clean_congestion(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. 종일 0인 종착역 방향을 제거한다.
    2. 나머지 0은 모두 첫차 전·막차 후 시간대라 미운행으로 보고 NaN으로 바꾼다.
       운행 시간 안에서는 0이 나오지 않는다(0이 아닌 최솟값 0.1%).
    3. 역명 표기를 역간 소요시간 데이터와 같은 기준으로 맞춘다(원래 이름은 station_raw에 남김).
    """
    zeros = find_structural_zeros(df)
    out = df.merge(zeros, on=SERIES_KEYS, how="left", indicator=True)
    out = out[out["_merge"] == "left_only"].drop(columns="_merge")

    out.loc[out["congestion"] == 0, "congestion"] = float("nan")

    out["station_raw"] = out["station"]
    out["station"] = out["station"].replace(STATION_ALIASES)
    return out.reset_index(drop=True)
