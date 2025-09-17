import streamlit as st
import pandas as pd

# 온실가스 배출량 데이터 (길이 맞춤)
years = list(range(2002, 2023))
emissions = [
    574, 590, 596, 597, 599,
    609, 615, 630, 633, 680,
    710, 715, 723, 720, 719,
    730, 745, 780, 760, 720,
    740
]

if len(years) != len(emissions):
    st.error(f"데이터 길이가 맞지 않습니다. 연도 {len(years)}개, 데이터 {len(emissions)}개")
else:
    df = pd.DataFrame({"연도": years, "배출량": emissions})

    st.title("🇰🇷 한국 온실가스 배출량 대시보드")
    st.caption("단위: 백만 톤 CO₂eq.")

    # 연도 범위 선택
    year_range = st.slider("조회할 연도 범위 선택", min_value=2002, max_value=2022, value=(2002, 2022))
    filtered_df = df[(df["연도"] >= year_range[0]) & (df["연도"] <= year_range[1])]

    # 지표 카드
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("평균 배출량", f"{filtered_df['배출량'].mean():.1f}")
    with col2:
        max_year = filtered_df.loc[filtered_df['배출량'].idxmax(), '연도']
        st.metric("최대 배출량", f"{filtered_df['배출량'].max()} ({max_year})")
    with col3:
        change = ((filtered_df['배출량'].iloc[-1] - filtered_df['배출량'].iloc[0]) / filtered_df['배출량'].iloc[0]) * 100
        st.metric("변화율", f"{change:+.1f}%")

    # 라인 차트
    st.line_chart(filtered_df.set_index("연도")["배출량"])

    st.info("이 그래프는 한국의 연도별 온실가스 배출량 변화를 보여줍니다.\n연도 범위를 조정해 구간별 추세를 확인할 수 있습니다.")