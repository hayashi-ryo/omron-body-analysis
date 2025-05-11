import json
import os
from datetime import datetime, timedelta
from matplotlib import font_manager
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
st.set_page_config(page_title="オムロン体組成分析", layout="wide")


# --- 定数 ---
DATA_DIR = "data"
STATUS_FILE = "processed_files.json"
CACHE_DATA_FILE = "cached_merged_df.pkl"

TARGETS = {
    "体重(kg)": 60.0,
    "体脂肪(%)": 18.0,
    "骨格筋(%)": 37.0
}

# --- 日本語フォント設定（fonts-japanese-gothic.ttf を使用） ---
JP_FONT_PATH = "/usr/share/fonts/truetype/fonts-japanese-mincho.ttf"
if os.path.exists(JP_FONT_PATH):
    font_manager.fontManager.addfont(JP_FONT_PATH)
    font_name = font_manager.FontProperties(fname=JP_FONT_PATH).get_name()
    plt.rcParams["font.family"] = font_name
    st.write(f"使用フォント: {font_name}")
else:
    st.warning(f"日本語フォントが見つかりません（{JP_FONT_PATH}）。文字化けが発生する可能性があります。")


st.title("💪 オムロン体組成データの分析ツール")

# --- 処理済みファイル読み込み ---


def load_processed_files():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, "r") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):  # 古い形式の互換対応
                    return {}
                return data
            except Exception:
                return {}
    return {}


def save_processed_files(file_dict):
    with open(STATUS_FILE, "w") as f:
        json.dump(file_dict, f, ensure_ascii=False, indent=2)

# --- CSVファイルのマージ処理 ---


def load_and_merge_csv_files():
    processed = load_processed_files()
    merged_df = pd.DataFrame()
    updated_processed = processed.copy()
    new_data_loaded = False

    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith(".csv"):
            continue

        path = os.path.join(DATA_DIR, fname)
        mtime = os.path.getmtime(path)
        if fname in processed and processed[fname] == mtime:
            continue

        try:
            df = pd.read_csv(path)
            df["測定日"] = pd.to_datetime(df["測定日"])
            merged_df = pd.concat([merged_df, df], ignore_index=True)
            updated_processed[fname] = mtime
            new_data_loaded = True
        except Exception as e:
            st.error(f"{fname} の読み込み失敗: {e}")

    if new_data_loaded:
        merged_df = merged_df.sort_values("測定日")
        merged_df["日付のみ"] = merged_df["測定日"].dt.date
        deduped_df = merged_df.groupby("日付のみ", as_index=False).last()
        deduped_df = deduped_df.sort_values("測定日")
        deduped_df.to_pickle(CACHE_DATA_FILE)
        save_processed_files(updated_processed)
        return deduped_df, True
    elif os.path.exists(CACHE_DATA_FILE):
        df = pd.read_pickle(CACHE_DATA_FILE)
        return df, False
    else:
        return pd.DataFrame(), False


# --- メイン処理 ---
df, is_new_data = load_and_merge_csv_files()

if df.empty:
    st.error("有効なデータが見つかりません。CSVファイルを追加してください。")
    st.stop()

if is_new_data:
    st.success("新しいCSVファイルを読み込みました。")
else:
    st.info("CSVに変更はありませんが、以前のデータを使用して表示します。")

# --- 移動平均 ---
for col in ["体重(kg)", "体脂肪(%)", "骨格筋(%)"]:
    df[f"{col}_5avg"] = df[col].rolling(window=5).mean()

# --- 最新5日 vs 1ヶ月前比較 ---
latest_date = df["測定日"].max()
one_month_ago = latest_date - timedelta(days=30)

recent_5avg = df[df["測定日"] <= latest_date].tail(
    5)[["体重(kg)", "体脂肪(%)", "骨格筋(%)"]].mean()
past_5avg = df[df["測定日"] <= one_month_ago].tail(
    5)[["体重(kg)", "体脂肪(%)", "骨格筋(%)"]].mean()
diff_ratio = ((recent_5avg - past_5avg) / past_5avg * 100).round(2)

# --- グラフ描画 ---
st.subheader("📈 5日移動平均と目標値の比較")
for col in ["体重(kg)", "体脂肪(%)", "骨格筋(%)"]:
    st.markdown(f"### {col}")
    fig, ax = plt.subplots()
    ax.plot(df["測定日"], df[f"{col}_5avg"], label="5日平均")
    ax.axhline(TARGETS[col], color="red", linestyle="--",
               label=f"目標値: {TARGETS[col]}")
    ax.set_ylabel(col)
    ax.legend()
    st.pyplot(fig)

st.subheader("🔄 最新の1ヶ月前比較（5日平均）")
st.write(diff_ratio.to_frame(name="変化率(%)"))

st.subheader("📉 変化率の時系列グラフ")
for col in ["体重(kg)", "体脂肪(%)", "骨格筋(%)"]:
    changes = []
    dates = []

    for i in range(len(df)):
        current_date = df.iloc[i]["測定日"]
        one_month_before = current_date - timedelta(days=30)
        recent = df[df["測定日"] <= current_date].tail(5)
        past = df[df["測定日"] <= one_month_before].tail(5)

        if len(recent) < 5 or len(past) < 5:
            continue

        r = recent[col].mean()
        p = past[col].mean()
        if p != 0:
            rate = ((r - p) / p) * 100
            changes.append(rate)
            dates.append(current_date)

    if changes:
        fig, ax = plt.subplots()
        ax.plot(dates, changes, label=f"{col} の変化率")
        ax.axhline(0, color="gray", linestyle=":")
        ax.set_ylabel("変化率 (%)")
        ax.set_title(f"{col} の1ヶ月前比（5日平均）")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info(f"{col} の変化率グラフを出力するのに十分なデータがありません。")
