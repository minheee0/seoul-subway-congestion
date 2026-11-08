"""서울시 역사마스터(수도권 전체 노선)로 1~8호선 역의 환승 여부를 판별한다.

1~8호선 데이터 안에서 역명이 겹치는지만 보면 9호선·신분당선·공항철도 등과의 환승을 놓친다.
여기서는 역명(별칭 포함)이 같고 좌표가 MAX_TRANSFER_KM 이내인 역을 같은 환승역으로 본다.
이름만 같은 다른 역(2호선 신촌과 경의중앙선 신촌, 5호선 양평과 중앙선 양평)은 거리 조건으로 걸러진다.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from subway.config import STATION_ALIASES
from subway.load import read_csv_kr

MAX_TRANSFER_KM = 0.6

# 운영 구간 이름 -> 승객이 아는 노선 이름. 같은 노선의 구간끼리는 환승으로 세지 않는다.
SERVICE_LINE = {
    "경부선": "1호선", "경원선": "1호선", "경인선": "1호선", "장항선": "1호선",
    "일산선": "3호선",
    "과천선": "4호선", "안산선": "4호선", "진접선": "4호선",
    "7호선(인천)": "7호선",
    "별내선": "8호선",
    "9호선(연장)": "9호선",
    "분당선": "수인분당선", "수인선": "수인분당선",
    "중앙선": "경의중앙선",
    "신분당선(연장)": "신분당선", "신분당선(연장2)": "신분당선",
}

# 역사마스터의 '경원선' 중 이촌~청량리(역사_ID 1008~1014)는 경의중앙선이 다니는 구간이다.
GYEONGUI_JUNGANG_IDS = range(1008, 1015)

# 역사마스터에 빠져 있는 (역명, 노선). 좌표는 같은 이름의 다른 노선 역에서 가져온다.
MASTER_SUPPLEMENT = [
    ("까치산", "2호선"),
    ("창동", "1호선"),
    ("왕십리", "수인분당선"),
    ("청량리", "수인분당선"),
    ("상봉", "경춘선"),
]

# 노선마다 이름이 다른 환승역. 같은 키로 묶어서 비교하되, 역 이름은 노선별 이름을 그대로 쓴다.
MATCH_KEY = {
    "서울": "서울역",          # GTX-A
    "이수": "총신대입구",      # 7호선 이수 = 4호선 총신대입구(이수)
}


# 역사마스터에 옛 이름으로 남아 있는 역.
MASTER_RENAMES = {"뚝섬유원지": "자양"}


def _canonical(name: str) -> str:
    name = re.sub(r"\(.*?\)", "", str(name)).strip()
    name = MASTER_RENAMES.get(name, name)
    return STATION_ALIASES.get(name, name)


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def load_station_master(path: str | Path) -> pd.DataFrame:
    df = read_csv_kr(path).rename(columns={"역사_ID": "id", "역사명": "name_raw", "호선": "line_raw", "위도": "lat", "경도": "lon"})
    df["station"] = df["name_raw"].map(_canonical)
    df["service_line"] = df["line_raw"].replace(SERVICE_LINE)
    df.loc[(df["line_raw"] == "경원선") & df["id"].isin(GYEONGUI_JUNGANG_IDS), "service_line"] = "경의중앙선"

    extra = []
    for station, line in MASTER_SUPPLEMENT:
        src = df[df["station"] == station].iloc[0]
        extra.append({**src.to_dict(), "service_line": line, "line_raw": f"{line}(보충)"})
    df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)

    df["key"] = df["station"].replace(MATCH_KEY)
    return df[["station", "key", "service_line", "line_raw", "name_raw", "lat", "lon"]]


def transfer_table(master: pd.DataFrame, max_km: float = MAX_TRANSFER_KM) -> pd.DataFrame:
    """1~8호선 역마다 환승 가능한 노선 목록과 좌표를 만든다.

    반환 열: line(1~8), station, n_lines(자기 노선 포함), other_lines, is_transfer, lat, lon
    """
    rows = []
    core = master[master["service_line"].str.fullmatch(r"[1-8]호선")]
    for (service_line, station), grp in core.groupby(["service_line", "station"]):
        lat, lon = grp["lat"].mean(), grp["lon"].mean()
        same_name = master[master["key"] == grp["key"].iloc[0]]
        near = same_name[_haversine_km(lat, lon, same_name["lat"], same_name["lon"]) <= max_km]
        others = sorted(set(near["service_line"]) - {service_line})
        rows.append({
            "line": int(service_line[0]),
            "station": station,
            "n_lines": len(others) + 1,
            "other_lines": ", ".join(others),
            "is_transfer": int(bool(others)),
            "lat": lat,
            "lon": lon,
        })
    return pd.DataFrame(rows)


def far_same_name_pairs(master: pd.DataFrame, max_km: float = MAX_TRANSFER_KM) -> pd.DataFrame:
    """이름은 같지만 max_km보다 멀어서 환승역으로 묶지 않은 쌍. 판별 기준을 점검할 때 쓴다."""
    m = master.drop_duplicates(["key", "service_line"]).reset_index(drop=True)
    pairs = m.merge(m, on="key", suffixes=("_a", "_b")).rename(columns={"station_a": "station"})
    pairs = pairs[pairs["service_line_a"] < pairs["service_line_b"]]
    pairs["km"] = _haversine_km(pairs["lat_a"], pairs["lon_a"], pairs["lat_b"], pairs["lon_b"])
    return pairs.loc[pairs["km"] > max_km, ["station", "service_line_a", "service_line_b", "km"]].sort_values("km")
