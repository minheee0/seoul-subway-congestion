"""역간 소요시간으로 1~8호선 노선 그래프를 만들고, 경로를 따라 시각별 혼잡도를 계산한다.

그래프 규칙
- 소요시간 파일의 각 행은 '바로 윗행 역 -> 이 역' 구간이다. 소요시간 0은 노선의 시작점이다.
- 지선은 파일에서 본선 뒤에 이어 붙어 있어서, 지선 첫 역의 실제 앞 역을 BRANCH_ORIGIN으로 지정한다.
- 파일 순서 방향이 어떤 방향 이름(상선/하선, 내선/외선)인지는 종착역의 '종일 0' 패턴으로 확인했다(05 노트북).
- 6호선 응암순환(응암→역촌→불광→독바위→연신내→구산→응암)은 한 방향으로만 운행한다.

혼잡도는 '출발역에서 그 방향으로 떠나는 열차의 30분 평균'을 쓴다.
"""

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from subway.transfer import MATCH_KEY

DWELL_MIN = 0.5      # 역마다 정차 시간(가정). 소요시간 파일은 정차 시간을 뺀 표준 운행시간이다.
TRANSFER_MIN = 4.0   # 환승 이동·대기 시간(가정)

# 지선 첫 역 -> 실제로 갈라지는 본선 역
BRANCH_ORIGIN = {(2, "용답"): "성수", (2, "도림천"): "신도림", (5, "둔촌동"): "강동"}

# 파일 순서(정방향)의 방향 이름과 역방향 이름
FORWARD_DIR = {line: ("하선", "상선") for line in range(1, 9)}
FORWARD_DIR[2] = ("내선", "외선")
BRANCH_FORWARD_DIR = {(2, "성수"): ("외선", "내선")}  # 성수지선만 파일 순서가 외선

# 한 역에 역번호가 여러 개라 어느 행을 쓸지 정해야 하는 구간: (노선, 출발역, 다음 역) -> 역번호
EDGE_CODE = {
    (2, "성수", "용답"): 9002, (2, "성수", "뚝섬"): 9001, (2, "성수", "건대입구"): 211,
    (2, "신도림", "도림천"): 9003, (2, "신도림", "문래"): 234, (2, "신도림", "대림"): 234,
    (5, "강동", "길동"): 2549, (5, "강동", "둔촌동"): 9005,
    (6, "응암", "역촌"): 9006, (6, "응암", "새절"): 2611,
}
ONE_WAY_LOOP = {(6, "응암"), (6, "역촌"), (6, "불광"), (6, "독바위"), (6, "연신내"), (6, "구산")}


def build_graph(travel: pd.DataFrame) -> nx.DiGraph:
    """노드 = (노선, 역). 승차 구간 엣지는 kind='ride', 환승 엣지는 kind='transfer'."""
    g = nx.DiGraph()
    for line, seg in travel.sort_values("seq").groupby("line"):
        rows = seg.to_dict("records")
        segment_start = None
        for prev, cur in zip(rows, rows[1:]):
            if cur["minutes"] == 0:
                segment_start = cur["station"]
                continue
            origin = BRANCH_ORIGIN.get((line, cur["station"]), prev["station"])
            if (line, cur["station"]) in BRANCH_ORIGIN:
                segment_start = origin
            fwd, bwd = BRANCH_FORWARD_DIR.get((line, segment_start), FORWARD_DIR[line])
            u, v = (line, origin), (line, cur["station"])
            g.add_edge(u, v, kind="ride", line=line, minutes=cur["minutes"], direction=fwd)
            loop = (line, origin) in ONE_WAY_LOOP and (line, cur["station"]) in ONE_WAY_LOOP
            if not loop:
                g.add_edge(v, u, kind="ride", line=line, minutes=cur["minutes"], direction=bwd)

    by_key = {}
    for node in g.nodes:
        by_key.setdefault(MATCH_KEY.get(node[1], node[1]), []).append(node)
    for nodes in by_key.values():
        for a in nodes:
            for b in nodes:
                if a[0] != b[0]:
                    g.add_edge(a, b, kind="transfer", minutes=TRANSFER_MIN)

    for u, v, d in g.edges(data=True):
        d["cost"] = d["minutes"] + (DWELL_MIN if d["kind"] == "ride" else 0)
    return g


class CongestionLookup:
    """(노선, 역, 방향, 요일, 시각)의 혼잡도. 시각은 30분 슬롯으로 내린다."""

    def __init__(self, congestion: pd.DataFrame):
        df = congestion.dropna(subset=["congestion"])
        self.by_code = df.set_index(["day_type", "line", "station_code", "direction", "hour"])["congestion"].to_dict()
        self.by_name = (df.groupby(["day_type", "line", "station", "direction", "hour"])["congestion"].mean().to_dict())

    def get(self, day_type, line, station, next_station, direction, hour) -> float:
        slot = np.floor(hour * 2) / 2
        code = EDGE_CODE.get((line, station, next_station))
        if code is not None:
            value = self.by_code.get((day_type, line, code, direction, slot))
            if value is not None:
                return value
        return self.by_name.get((day_type, line, station, direction, slot), np.nan)


@dataclass
class Trip:
    path: list
    depart: float
    arrive: float
    legs: pd.DataFrame          # 승차 구간별 시각·혼잡도
    mean_congestion: float      # 승차 시간 가중 평균
    max_congestion: float
    transfers: int


def _endpoints(g: nx.DiGraph, station: str) -> list:
    nodes = [n for n in g.nodes if n[1] == station]
    if not nodes:
        raise KeyError(f"1~8호선에 없는 역: {station}")
    return nodes


def candidate_paths(g: nx.DiGraph, origin: str, dest: str, k: int = 3, slack_min: float = 10) -> list[list]:
    """최단 경로와, 그보다 slack_min분 이내로 더 걸리는 대안 경로를 최대 k개 돌려준다."""
    h = g.copy()
    h.add_edges_from(((("src", origin), n) for n in _endpoints(g, origin)), kind="virtual", cost=0, minutes=0)
    h.add_edges_from(((n, ("dst", dest)) for n in _endpoints(g, dest)), kind="virtual", cost=0, minutes=0)
    paths, best = [], None
    for p in nx.shortest_simple_paths(h, ("src", origin), ("dst", dest), weight="cost"):
        cost = nx.path_weight(h, p, "cost")
        best = best if best is not None else cost
        if cost > best + slack_min or len(paths) >= k:
            break
        core = p[1:-1]
        kinds = [g.edges[u, v]["kind"] for u, v in zip(core, core[1:])]
        # 출발·도착역에서 바로 환승하거나, 환승을 연달아 하는 경로(갈아탔다가 바로 돌아옴)는 같은 경로의 중복이다
        if kinds[0] == "transfer" or kinds[-1] == "transfer" or any(a == b == "transfer" for a, b in zip(kinds, kinds[1:])):
            continue
        paths.append(core)
    return paths


def simulate(g: nx.DiGraph, lookup: CongestionLookup, path: list, depart: float, day_type: str = "평일") -> Trip:
    t, legs, transfers = depart, [], 0
    for u, v in zip(path, path[1:]):
        e = g.edges[u, v]
        if e["kind"] == "transfer":
            transfers += 1
        else:
            c = lookup.get(day_type, e["line"], u[1], v[1], e["direction"], t)
            legs.append({"line": e["line"], "from": u[1], "to": v[1], "direction": e["direction"],
                         "board": t, "minutes": e["cost"], "congestion": c})
        t += e["cost"] / 60
    legs = pd.DataFrame(legs)
    valid = legs.dropna(subset=["congestion"])
    mean = np.average(valid["congestion"], weights=valid["minutes"]) if len(valid) else np.nan
    return Trip(path, depart, t, legs, mean, valid["congestion"].max(), transfers)


def departure_table(g, lookup, origin, dest, start, end, step_min=10, day_type="평일", k=3, slack_min=10) -> pd.DataFrame:
    """start~end 사이 출발 시각마다 후보 경로의 도착 시각과 혼잡도를 계산한다."""
    paths = candidate_paths(g, origin, dest, k=k, slack_min=slack_min)
    rows = []
    for depart in np.arange(start, end + 1e-9, step_min / 60):
        for i, p in enumerate(paths):
            trip = simulate(g, lookup, p, depart, day_type)
            rows.append({"출발": depart, "경로": i, "도착": trip.arrive, "소요(분)": (trip.arrive - depart) * 60,
                         "환승": trip.transfers, "평균 혼잡도": trip.mean_congestion, "최대 혼잡도": trip.max_congestion,
                         "노선": line_sequence(trip.legs)})
    return pd.DataFrame(rows)


def line_sequence(legs: pd.DataFrame) -> str:
    """승차 구간의 노선을 갈아탄 순서대로 보여준다. 예: '2호선 → 3호선 → 2호선'"""
    lines = legs["line"].tolist()
    runs = [l for i, l in enumerate(lines) if i == 0 or l != lines[i - 1]]
    return " → ".join(f"{l}호선" for l in runs)


def recommend(table: pd.DataFrame, arrive_by: float | None = None) -> pd.Series:
    """도착 기한 안에서 평균 혼잡도가 가장 낮은 (출발 시각, 경로). 동률이면 늦게 출발하는 쪽."""
    t = table if arrive_by is None else table[table["도착"] <= arrive_by]
    if t.empty:
        raise ValueError("기한 안에 도착하는 출발 시각이 없음")
    return t.sort_values(["평균 혼잡도", "출발"], ascending=[True, False]).iloc[0]


def fmt_time(hour: float) -> str:
    m = int(round(hour * 60))
    return f"{(m // 60) % 24:02d}:{m % 60:02d}"
