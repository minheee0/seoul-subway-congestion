"""웹앱 배포용 데이터를 만든다.

원본 CSV는 저장소에 올리지 않으므로, 앱에 필요한 열만 정제해서 app/data/에 저장한다.
실행: python scripts/make_app_data.py
"""

from pathlib import Path

from subway.clean import clean_congestion
from subway.load import load_congestion, load_travel_time

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "app" / "data"

CONGESTION_COLS = ["day_type", "line", "station_code", "station", "direction", "hour", "congestion"]
TRAVEL_COLS = ["seq", "line", "station", "minutes"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    congestion = clean_congestion(load_congestion(RAW / "서울교통공사_지하철혼잡도정보_20260930.csv"))
    congestion = congestion.dropna(subset=["congestion"])[CONGESTION_COLS]
    congestion.to_csv(OUT / "congestion.csv", index=False)

    travel = load_travel_time(RAW / "서울교통공사 역간거리 및 소요시간_240810.csv")[TRAVEL_COLS]
    travel.to_csv(OUT / "travel_time.csv", index=False)

    for name, df in [("congestion", congestion), ("travel_time", travel)]:
        size = (OUT / f"{name}.csv").stat().st_size / 1024
        print(f"{name}.csv: {len(df):,}행, {size:,.0f} KB")


if __name__ == "__main__":
    main()
