# streamlit_app.py
"""
Streamlit 앱: 공개 데이터 대시보드 + 사용자 입력(프롬프트) 기반 대시보드
- 공개 데이터: Our World in Data CO2 (OWID) + NASA GISTEMP global temp (GISTEMP)
- 사용자 입력: 프롬프트 내 제공된 한글 설명(20년 온난화, 폭염일수 증가) + 업로드된 시각화 이미지 사용
- 코드 주석에 출처(URL) 명시

출처:
- OWID CO2 데이터 (CSV): https://github.com/owid/co2-data -> raw CSV URL used in code.
  (원본: https://ourworldindata.org/co2-and-other-greenhouse-gas-emissions)
  raw CSV: https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv
- NASA GISTEMP global annual/seasonal temperature: https://data.giss.nasa.gov/gistemp/
  GISTEMP raw CSV used: https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv
- KMA Open MET (참고): https://data.kma.go.kr/
"""

from __future__ import annotations
import io
import time
import requests
from typing import Tuple
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime

st.set_page_config(layout="wide", page_title="기후 대시보드 (공개 데이터 + 사용자 입력)", page_icon="🌍")

# --- 폰트 적용 시도 (Pretendard) ---
try:
    import matplotlib.font_manager as fm
    FONT_PATH = "/fonts/Pretendard-Bold.ttf"
    fm.fontManager.addfont(FONT_PATH)
    plt.rcParams['font.family'] = fm.FontProperties(fname=FONT_PATH).get_name()
except Exception:
    # 폰트가 없거나 추가 실패하면 기본 폰트 사용
    pass

# --- 유틸리티 함수: HTTP 요청 재시도 및 예시 데이터 fallback ---
def fetch_csv_with_retry(url: str, max_retries: int = 3, timeout: int = 15) -> Tuple[pd.DataFrame, str]:
    """
    주어진 URL에서 CSV를 시도해서 불러옴.
    실패 시 빈 DataFrame과 오류 메시지 반환.
    """
    last_err = ""
    for attempt in range(1, max_retries+1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            content = resp.content.decode('utf-8', errors='replace')
            df = pd.read_csv(io.StringIO(content))
            return df, ""
        except Exception as e:
            last_err = f"Attempt {attempt} failed: {e}"
            time.sleep(1)
    return pd.DataFrame(), f"모든 시도 실패: {last_err}"

@st.cache_data(show_spinner=False)
def load_owid_co2() -> Tuple[pd.DataFrame, str]:
    url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
    df, err = fetch_csv_with_retry(url)
    if df.empty:
        # 예시 데이터로 대체 (한국 CO2 간단 예시)
        years = list(range(2000, 2024))
        values = [570 + (i-2000)*3 + (np.random.rand()-0.4)*10 for i in range(len(years))]  # MtCO2 대충
        df = pd.DataFrame({
            "iso_code": ["KOR"] * len(years),
            "country": ["South Korea"] * len(years),
            "year": years,
            "co2": values
        })
        err = "OWID 데이터 로드 실패 — 예시 데이터로 대체됨."
    return df, err

@st.cache_data(show_spinner=False)
def load_gistemp_global() -> Tuple[pd.DataFrame, str]:
    """
    NASA GISTEMP provides a table CSV; we'll attempt to parse annual global mean (구조가 특이할 수 있음).
    raw URL: https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv
    """
    url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
    df, err = fetch_csv_with_retry(url)
    if df.empty:
        # 예시 글로벌 온도 이상치(연평균 이상온도) 예시
        years = list(range(2000, 2024))
        anomalies = [0.4 + 0.02*(y-2000) + (np.random.rand()-0.5)*0.05 for y in range(len(years))]
        df = pd.DataFrame({"Year": years, "Annual": anomalies})
        err = "GISTEMP 데이터 로드 실패 — 예시 데이터로 대체됨."
        return df, err

    # GISTEMP CSV 포맷: 첫 열 'Year', 마지막 열 'J-D' or 'Annual' 등. 파싱을 유연하게.
    try:
        # 파일에 헤더 주석이 있는 경우 첫 숫자행을 찾음
        # 판다스로 읽었을 때 year-like 칼럼 찾아 처리
        # 이미 df is a DataFrame; ensure Year and Annual columns exist.
        if "Year" in df.columns and ("J-D" in df.columns or "Annual" in df.columns):
            col = "Annual" if "Annual" in df.columns else "J-D"
            out = df[["Year", col]].rename(columns={col: "Annual"})
            out = out.dropna(subset=["Annual"])
            out['Annual'] = pd.to_numeric(out['Annual'], errors='coerce')
            out = out[out['Year'].apply(lambda x: str(x).isdigit())]
            out['Year'] = out['Year'].astype(int)
            return out, ""
        else:
            # 경우에 따라 df의 첫열이 Year로 잘 들어가지 않음 -> 시도 변환
            first_col = df.columns[0]
            # drop non-numeric rows:
            df2 = df[df[first_col].astype(str).str.match(r'^\d{4}$')]
            if df2.shape[0] > 0:
                # pick a sensible annual column (last numeric)
                numeric_cols = [c for c in df2.columns if df2[c].astype(str).str.replace('.','',1).str.isnumeric().all()]
                if len(numeric_cols) >= 2:
                    year_col = numeric_cols[0]
                    annual_col = numeric_cols[-1]
                    out = df2[[year_col, annual_col]]
                    out.columns = ["Year", "Annual"]
                    out['Year'] = out['Year'].astype(int)
                    out['Annual'] = pd.to_numeric(out['Annual'], errors='coerce')
                    return out, ""
    except Exception as e:
        return pd.DataFrame(), f"GISTEMP 파싱 중 오류: {e}"

    return pd.DataFrame(), "GISTEMP 포맷이 예상과 다름 — 파싱 실패"

# --- 데이터 불러오기 ---
owid_df, owid_err = load_owid_co2()
gistemp_df, gistemp_err = load_gistemp_global()

# 공공 데이터 전처리 (한국 필터링)
def prepare_korea_co2(df: pd.DataFrame) -> pd.DataFrame:
    """OWID에서 South Korea만 추출하고 표준화 (date,value)."""
    if df.empty:
        return pd.DataFrame()
    df_kor = df[df['country'].str.contains("Korea", case=False, na=False) | (df.get('iso_code') == 'KOR')].copy()
    if 'year' in df_kor.columns:
        df_kor = df_kor[['year', 'co2']].rename(columns={'year': 'date', 'co2': 'value'})
    elif 'Year' in df_kor.columns:
        df_kor = df_kor[['Year', 'co2']].rename(columns={'Year': 'date', 'co2': 'value'})
    else:
        # fallback: try to find year-like column and co2-like column
        possible_year = next((c for c in df_kor.columns if 'year' in c.lower() or c.lower().strip() == 'year'), None)
        possible_co2 = next((c for c in df_kor.columns if 'co2' in c.lower()), None)
        if possible_year and possible_co2:
            df_kor = df_kor[[possible_year, possible_co2]].rename(columns={possible_year: 'date', possible_co2: 'value'})
        else:
            return pd.DataFrame()
    df_kor['date'] = pd.to_numeric(df_kor['date'], errors='coerce')
    df_kor['value'] = pd.to_numeric(df_kor['value'], errors='coerce')
    df_kor = df_kor.dropna(subset=['date'])
    # 미래 데이터 제거 (오늘 자정 이후 데이터는 제거)
    current_year = datetime.now().year
    df_kor = df_kor[df_kor['date'] <= current_year]
    df_kor = df_kor.sort_values('date').drop_duplicates(subset=['date'])
    return df_kor

def prepare_global_temp(df: pd.DataFrame) -> pd.DataFrame:
    """GISTEMP에서 연평균 이상온도(Annual) 추출하여 표준화"""
    if df.empty:
        return pd.DataFrame()
    if 'Year' in df.columns:
        out = df[['Year', 'Annual']].rename(columns={'Year': 'date', 'Annual': 'value'})
        out['date'] = pd.to_numeric(out['date'], errors='coerce')
        out['value'] = pd.to_numeric(out['value'], errors='coerce')
        out = out.dropna(subset=['date'])
        current_year = datetime.now().year
        out = out[out['date'] <= current_year]
        out = out.sort_values('date').drop_duplicates(subset=['date'])
        return out
    # fallback: try to find numeric year col
    return pd.DataFrame()

korea_co2 = prepare_korea_co2(owid_df)
global_temp = prepare_global_temp(gistemp_df)

# --- 레이아웃: 사이드바 및 섹션 선택 ---
st.sidebar.title("대시보드 옵션")
section = st.sidebar.radio("섹션 선택", ["공개 데이터 대시보드", "사용자 입력(프롬프트) 대시보드", "원본 데이터 다운로드"])

st.title("기후 데이터 대시보드 🌍")
st.markdown("**설명:** 공개 데이터(OWID CO₂, NASA GISTEMP)를 우선 로드하고, 사용자가 제공한 프롬프트 텍스트와 이미지로 별도 분석을 생성합니다. 모든 라벨과 안내는 한국어입니다.")

# 공공 데이터 섹션
if section == "공개 데이터 대시보드":
    st.header("공개 데이터 대시보드")
    col1, col2 = st.columns([2,1])

    with col1:
        st.subheader("한국 연간 CO₂ 배출량 (Our World in Data)")
        if owid_err:
            st.warning("공개 데이터(OWID CO2) 로드 중 오류가 발생했습니다. 메시지: " + owid_err)
        if korea_co2.empty:
            st.error("한국 CO₂ 데이터가 준비되지 않았습니다. (대체 예시 사용 여부 확인)")
        else:
            fig = px.line(korea_co2, x='date', y='value', markers=True,
                          labels={'date': '연도', 'value': 'CO₂ 배출량 (톤)'},
                          title="대한민국 연간 CO₂ 배출량 추이 (OWID)")
            fig.update_layout(xaxis=dict(dtick=1))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**설명:** 데이터 출처: Our World in Data (owid-co2-data).")
        st.markdown("코드 주석에 원본 URL이 포함되어 있습니다.")

    with col2:
        st.subheader("글로벌 연평균 기온 이상치 (NASA GISTEMP)")
        if gistemp_err:
            st.warning("GISTEMP 데이터 로드/파싱 오류: " + gistemp_err)
        if global_temp.empty:
            st.error("GISTEMP 연평균 온도 데이터가 준비되지 않았습니다.")
        else:
            fig2 = px.line(global_temp, x='date', y='value', labels={'date':'연도','value':'연평균 이상온도 (℃)'},
                           title="전지구 연평균 온도 이상치 (GISTEMP)")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**참고:** 이상온도는 기준 기간 대비 차이(Anomaly)입니다. 출처: NASA GISTEMP.")

    st.markdown("---")
    st.subheader("간단 인사이트 요약")
    if not korea_co2.empty:
        recent = korea_co2.tail(3)
        st.write("최근 연도 데이터 (예시):")
        st.table(recent)
        change = korea_co2.iloc[-1]['value'] - korea_co2.iloc[0]['value']
        st.write(f"기간: {int(korea_co2['date'].min())} - {int(korea_co2['date'].max())} | 총 변화량: {change:.1f} (단위: CO₂)")
    else:
        st.write("한국 CO₂ 데이터가 없어 인사이트를 생성할 수 없습니다.")

# 사용자 입력(프롬프트) 섹션
if section == "사용자 입력(프롬프트) 대시보드":
    st.header("사용자 입력 기반 대시보드")
    st.markdown("프롬프트에서 제공된 설명을 바탕으로 '지난 20년간 한국 평균기온 상승'과 '폭염일수 증가'를 가상/재구성하여 시각화합니다.")
    st.markdown("앱 실행 중 파일 업로드를 요구하지 않으며, 대화에서 제공된 이미지가 자동으로 사용됩니다.")

    # 이미지 표시: 컨테이너에 업로드된 이미지 파일 사용 (개발자가 지정)
    try:
        img1 = "/mnt/data/스크린샷 2025-09-16 오전 11.10.59.png"
        img2 = "/mnt/data/국어 시각화 자료_지구가열.png"
        img3 = "/mnt/data/스크린샷 2025-09-16 오전 11.12.48.png"
        st.image([img1, img2, img3], caption=["온실가스·GDP 테이블 스냅샷", "지구가열 시각화", "온실가스 배출량 시계열"], use_column_width=True)
    except Exception:
        st.info("제공된 이미지가 없습니다 또는 경로를 찾을 수 없습니다. (개발자 경로 사용 중)")

    # 프롬프트 텍스트 기반 데이터 생성 (입력 섹션의 내용 활용)
    st.subheader("프롬프트 기반 재구성 데이터 (예시)")
    st.markdown("프롬프트에서 '지난 20년간 평균기온 약 +1.4℃ 상승', '폭염일수 1.5배 증가' 등의 기술을 기반으로 예시 데이터를 생성합니다.")

    # 생성: 2003-2022 (지난 20년)
    years = np.arange(2003, 2023)
    # avg temp baseline (2003): 가정 12.0 -> 2022에는 +1.4
    temps = 12.0 + (1.4 / (len(years)-1)) * (years - years[0]) + np.random.normal(0, 0.05, len(years))
    # 폭염일수: baseline 5일 -> 1.5배 증가 over period
    heatdays = 5.0 * (1 + (0.5/(len(years)-1)) * (years - years[0])) + np.random.poisson(1, len(years))*0.2

    user_df = pd.DataFrame({"연도": years, "평균기온(℃)": np.round(temps, 2), "폭염일수(일)": np.round(heatdays,1)})

    # 시각화: 온도 추세 (라인) + 폭염일수 (막대)
    fig = px.line(user_df, x='연도', y='평균기온(℃)', markers=True, title="(프롬프트 재구성) 지난 20년간 평균기온 추이")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.bar(user_df, x='연도', y='폭염일수(일)', title="(프롬프트 재구성) 지난 20년간 폭염일수 변화")
    st.plotly_chart(fig2, use_container_width=True)

    # 간단 통계
    st.subheader("요약 통계 (프롬프트 기반 데이터)")
    st.write(user_df.describe())

    # CSV 다운로드 버튼 (전처리된 표로 내보내기)
    csv_buffer = io.StringIO()
    user_df.to_csv(csv_buffer, index=False)
    st.download_button("프롬프트 기반 전처리 데이터 다운로드 (CSV)", csv_buffer.getvalue(), file_name="prompt_reconstruction_20yrs.csv", mime="text/csv")

    st.markdown("**제언(교육용 메시지 예시):** 기후변화는 이미 현실입니다. 청소년 대상 에너지 절약 챌린지(연간 전력 10% 절감 등)는 개인 단위로도 연간 CO₂ 절감(약 120~150kg) 효과를 냅니다. 데이터 기반 행동 목표를 세우세요.")

# 원본 데이터 다운로드 섹션
if section == "원본 데이터 다운로드":
    st.header("원본/전처리 데이터 다운로드")
    st.markdown("공개 데이터(OWID, GISTEMP) 및 프롬프트 기반 재구성 데이터의 전처리된 표를 CSV로 다운로드할 수 있습니다.")

    if not korea_co2.empty:
        buf = io.StringIO()
        korea_co2.to_csv(buf, index=False)
        st.download_button("한국 CO₂ (OWID) 전처리 CSV 다운로드", buf.getvalue(), file_name="korea_co2_owid_preprocessed.csv", mime="text/csv")

    if not global_temp.empty:
        buf2 = io.StringIO()
        global_temp.to_csv(buf2, index=False)
        st.download_button("Global Temp (GISTEMP) 전처리 CSV 다운로드", buf2.getvalue(), file_name="global_temp_gistemp_preprocessed.csv", mime="text/csv")

    # 프롬프트 데이터
    buf3 = io.StringIO()
    user_df.to_csv(buf3, index=False)
    st.download_button("프롬프트 재구성 데이터 CSV 다운로드", buf3.getvalue(), file_name="prompt_reconstruction_20yrs.csv", mime="text/csv")

st.markdown("---")
st.markdown("**참고/주의:**")
st.markdown("- 공개 데이터 로드 실패 시 예시 데이터를 자동 생성하여 보여줍니다. 상단 알림을 확인하세요.")
st.markdown("- 본 앱은 교육/시연 목적의 대시보드 템플릿이며, 실제 연구/정책 결정에는 원본 데이터를 직접 확인하세요.")
st.markdown("- 출처: OWID(owid-co2-data), NASA GISTEMP, KMA(Open MET) 등. (코드 상단 주석 참조)")
