"""통학 경로 혼잡도 웹앱."""

from datetime import time, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from subway.clean import clean_congestion
from subway.load import load_congestion, load_travel_time
from subway.route import (CongestionLookup, build_graph, candidate_paths, departure_table, fmt_time, recommend,
                          simulate)

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

col3, col4 = st.columns([1, 2])
day_type = col3.radio("요일", ["평일", "토요일", "일요일"], horizontal=True)
depart = col4.time_input("출발 시각", value=time(8, 0), step=timedelta(minutes=10))
depart_hour = depart.hour + depart.minute / 60
deadline = None
if st.checkbox("도착 기한 정하기"):
    arrive_by = st.time_input("이 시각까지 도착", value=time(9, 0), step=timedelta(minutes=10))
    deadline = arrive_by.hour + arrive_by.minute / 60

if origin == dest:
    st.warning("출발역과 도착역이 같아요.")
    st.stop()

table = departure_table(graph, lookup, origin, dest, depart_hour, depart_hour, day_type=day_type)
table["도착"] = table["도착"].map(fmt_time)
st.subheader(f"{depart.strftime('%H:%M')} 출발 경로 비교")
st.dataframe(table[["노선", "소요(분)", "환승", "도착", "평균 혼잡도", "최대 혼잡도"]].round(1), hide_index=True)

with st.expander("구간별 혼잡도 보기"):
    paths = candidate_paths(graph, origin, dest)
    tabs = st.tabs([f"경로 {i + 1}: {r}" for i, r in enumerate(table["노선"])])
    for tab, path in zip(tabs, paths):
        legs = simulate(graph, lookup, path, depart_hour, day_type).legs
        tab.dataframe(pd.DataFrame({
            "탑승": legs["board"].map(fmt_time),
            "노선": legs["line"].astype(str) + "호선",
            "구간": legs["from"] + " → " + legs["to"],
            "방향": legs["direction"],
            "혼잡도(%)": legs["congestion"].round(1),
        }), hide_index=True)

# 출발 시각을 앞뒤 1시간 범위에서 10분씩 바꿔 본다
start, end = max(5.5, depart_hour - 1), min(23.5, depart_hour + 1)
around = departure_table(graph, lookup, origin, dest, start, end, step_min=10, day_type=day_type)
around["출발 시각"] = pd.Timestamp("2000-01-01") + pd.to_timedelta(around["출발"], unit="h")

try:
    best = recommend(around, arrive_by=deadline)
    now = table.iloc[0]
    gap = now["평균 혼잡도"] - best["평균 혼잡도"]
    msg = f"**{fmt_time(best['출발'])} 출발, {best['노선']}** → {fmt_time(best['도착'])} 도착, 평균 혼잡도 **{best['평균 혼잡도']:.0f}%**"
    if gap >= 1:
        msg += f"  \n지금 고른 시각의 최단 경로({now['평균 혼잡도']:.0f}%)보다 {gap:.0f}%p 덜 붐벼요."
    st.success(msg, icon="✅")
except ValueError:
    st.warning("앞뒤 1시간 안에는 도착 기한을 맞추는 출발 시각이 없어요. 출발 시각을 앞당겨 보세요.")

ROUTE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # 경로 순서대로 고정
fig = go.Figure()
for i, (_, r) in enumerate(around.groupby("경로")):
    fig.add_trace(go.Scatter(
        x=r["출발 시각"], y=r["평균 혼잡도"], name=f"{r['노선'].iloc[0]} ({r['소요(분)'].iloc[0]:.0f}분)",
        mode="lines+markers", line=dict(width=2, color=ROUTE_COLORS[i]), marker=dict(size=8),
        hovertemplate="%{y:.0f}%<extra>%{fullData.name}</extra>",
    ))
selected = pd.Timestamp("2000-01-01") + pd.to_timedelta(depart_hour, unit="h")
fig.add_vline(x=selected, line=dict(width=1, dash="dot", color="gray"))
fig.update_layout(
    title="출발 시각에 따른 경로 평균 혼잡도", hovermode="x unified", height=360,
    xaxis=dict(title="출발 시각", tickformat="%H:%M"), yaxis=dict(title="평균 혼잡도(%)", rangemode="tozero"),
    legend=dict(orientation="h", y=-0.25), margin=dict(t=50, b=10),
)
st.plotly_chart(fig, use_container_width=True)
