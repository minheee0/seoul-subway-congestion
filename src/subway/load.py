"""서울교통공사 혼잡도 CSV를 분석용 롱 포맷으로 불러온다."""

import re
import sys
from pathlib import Path

import pandas as pd

TIME_COL = re.compile(r"^(\d{1,2})시(\d{2})분$")

# 파일 버전마다 컬럼명이 조금씩 달라서 후보 이름으로 찾는다.
ID_COLUMNS = {
    "day_type": ["요일구분", "구분"],
    "line": ["호선"],
    "station_code": ["역번호"],
    "station": ["출발역", "역명"],
    "direction": ["상하구분"],
}


def read_csv_kr(path: str | Path) -> pd.DataFrame:
    """공공데이터 CSV는 UTF-8과 CP949가 섞여 있어 둘 다 시도한다."""
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"인코딩을 알 수 없음: {path}")


def _find_column(columns: list[str], candidates: list[str]) -> str:
    for name in candidates:
        if name in columns:
            return name
    raise KeyError(f"컬럼을 찾을 수 없음: {candidates} / 실제: {columns}")


def time_label_to_hour(label: str) -> float:
    """'07시30분' -> 7.5, '00시30분' -> 24.5 (운행일 기준으로 자정 이후를 이어 붙임)."""
    h, m = map(int, TIME_COL.match(label).groups())
    hour = h + m / 60
    return hour + 24 if h < 5 else hour


def load_congestion(path: str | Path) -> pd.DataFrame:
    """와이드 포맷(시간대별 컬럼)을 역·방향·시각 단위 롱 포맷으로 바꾼다.

    결측은 채우지 않고 NaN으로 둔다. 어떻게 처리할지는 EDA에서 패턴을 본 뒤 정한다.
    """
    df = read_csv_kr(path)
    df.columns = df.columns.str.strip()
    cols = list(df.columns)

    rename = {_find_column(cols, cands): new for new, cands in ID_COLUMNS.items()}
    time_cols = [c for c in cols if TIME_COL.match(c)]
    if not time_cols:
        raise KeyError(f"시간대 컬럼을 찾을 수 없음: {cols}")

    long = (
        df.rename(columns=rename)
        .melt(id_vars=list(rename.values()), value_vars=time_cols,
              var_name="time_label", value_name="congestion")
    )
    long["hour"] = long["time_label"].map(time_label_to_hour)
    long["line"] = long["line"].astype(str).str.extract(r"(\d+)", expand=False).astype(int)
    long["congestion"] = pd.to_numeric(long["congestion"], errors="coerce")
    long["station"] = long["station"].astype(str).str.strip()
    return long.drop(columns="time_label")


def load_travel_time(path: str | Path) -> pd.DataFrame:
    """역간 소요시간 CSV. 각 행은 '바로 앞 행의 역 -> 이 역' 구간이다(소요시간 00:00은 노선·지선의 시작점).

    역명은 config.STATION_ALIASES로 혼잡도 데이터와 같은 표기로 맞춘다.
    """
    from subway.config import STATION_ALIASES

    df = read_csv_kr(path).rename(columns={
        "연번": "seq", "호선": "line", "역명": "station", "소요시간": "travel_time",
        "역간거리(km)": "distance_km", "호선별누계(km)": "cum_km",
    })
    mm_ss = df["travel_time"].str.split(":", expand=True).astype(int)
    df["minutes"] = mm_ss[0] + mm_ss[1] / 60
    df["station"] = df["station"].str.strip().replace(STATION_ALIASES)
    return df.drop(columns="travel_time")


if __name__ == "__main__":
    data = load_congestion(sys.argv[1])
    print(data.head(10))
    print(f"\n행 {len(data):,}개 / 역 {data['station'].nunique()}개 / "
          f"결측 {data['congestion'].isna().mean():.1%}")
    print(data.groupby(["day_type", "direction"]).size())
