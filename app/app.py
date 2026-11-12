"""통학 경로 혼잡도 웹앱."""

from pathlib import Path

import streamlit as st

from subway.clean import clean_congestion
from subway.load import load_congestion, load_travel_time
from subway.route import CongestionLookup, build_graph

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


@st.cache_resource
def load():
    df = clean_congestion(load_congestion(RAW / "서울교통공사_지하철혼잡도정보_20260930.csv"))
    graph = build_graph(load_travel_time(RAW / "서울교통공사 역간거리 및 소요시간_240810.csv"))
    return graph, CongestionLookup(df)


st.title("🚇 통학 경로 혼잡도")
graph, lookup = load()
stations = sorted({n[1] for n in graph.nodes})

col1, col2 = st.columns(2)
origin = col1.selectbox("출발역", stations, index=stations.index("사당"))
dest = col2.selectbox("도착역", stations, index=stations.index("홍대입구"))
