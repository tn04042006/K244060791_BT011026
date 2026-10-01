import streamlit as st
import pandas as pd
import plotly.express as px
from ucimlrepo import fetch_ucirepo


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Credit Approval Dashboard",
    page_icon="💳",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("💳 Credit Approval Dashboard")

st.markdown(
    """
    Dashboard trực quan hóa dữ liệu **Credit Approval**
    từ UCI Machine Learning Repository.
    """
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    # Fetch dataset từ UCI
    credit_approval = fetch_ucirepo(id=27)

    # Features
    X = credit_approval.data.features.copy()

    # Target
    y = credit_approval.data.targets.copy()

    # Ghép features + target
    df = pd.concat([X, y], axis=1)

    return df


try:
    df = load_data()

except Exception as e:
    st.error("Không thể tải dữ liệu từ UCI.")
    st.exception(e)
    st.stop()


# =========================================================
# COLUMN NAMES
# =========================================================

df.columns = [
    "A1", "A2", "A3", "A4",
    "A5", "A6", "A7", "A8",
    "A9", "A10", "A11", "A12",
    "A13", "A14", "A15", "A16"
]


# =========================================================
# DATA CLEANING
# =========================================================

# Missing values trong dataset được biểu diễn bằng ?
df = df.replace("?", pd.NA)


# Các biến dạng số
numeric_cols = [
    "A2",
    "A3",
    "A8",
    "A11",
    "A14",
    "A15"
]


# Chuyển sang numeric
for col in numeric_cols:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Bộ lọc")


categorical_cols = [
    "A1",
    "A4",
    "A5",
    "A6",
    "A7",
    "A9",
    "A10",
    "A12",
    "A13"
]


selected_category = st.sidebar.selectbox(
    "Chọn biến phân loại",
    categorical_cols
)


selected_numeric = st.sidebar.selectbox(
    "Chọn biến số",
    numeric_cols
)


st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Dataset:** Credit Approval

    **Nguồn:** UCI Machine Learning Repository

    **Số dòng:** 690

    **Số biến:** 16
    """
)


# =========================================================
# OVERVIEW
# =========================================================

st.header("📊 Tổng quan dữ liệu")


total_records = len(df)

approved = (
    df["A16"]
    .astype(str)
    .str.strip()
    .eq("+")
    .sum()
)

rejected = (
    df["A16"]
    .astype(str)
    .str.strip()
    .eq("-")
    .sum()
)


missing_total = int(df.isna().sum().sum())


col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric(
        "Tổng số hồ sơ",
        total_records
    )


with col2:
    st.metric(
        "Được duyệt",
        approved
    )


with col3:
    st.metric(
        "Bị từ chối",
        rejected
    )


with col4:
    st.metric(
        "Missing values",
        missing_total
    )


st.divider()


# =========================================================
# CHART 1 - APPROVAL DISTRIBUTION
# =========================================================

st.header("1️⃣ Phân bố kết quả xét duyệt")


approval_count = (
    df["A16"]
    .value_counts(dropna=False)
    .reset_index()
)

approval_count.columns = [
    "Approval",
    "Count"
]


fig1 = px.pie(
    approval_count,
    names="Approval",
    values="Count",
    hole=0.4,
    title="Tỷ lệ hồ sơ được duyệt và từ chối"
)


fig1.update_traces(
    textinfo="percent+label"
)


st.plotly_chart(
    fig1,
    use_container_width=True
)


# =========================================================
# CHART 2 - CATEGORY VS APPROVAL
# =========================================================

st.header(
    f"2️⃣ Kết quả xét duyệt theo {selected_category}"
)


category_df = (
    df.groupby(
        [selected_category, "A16"],
        dropna=False
    )
    .size()
    .reset_index(name="Count")
)


category_df[selected_category] = (
    category_df[selected_category]
    .astype(str)
)


fig2 = px.bar(
    category_df,
    x=selected_category,
    y="Count",
    color="A16",
    barmode="group",
    title=(
        f"Phân bố kết quả xét duyệt "
        f"theo {selected_category}"
    ),
    labels={
        "Count": "Số lượng",
        "A16": "Kết quả"
    }
)


st.plotly_chart(
    fig2,
    use_container_width=True
)


# =========================================================
# CHART 3 - HISTOGRAM
# =========================================================

st.header(
    f"3️⃣ Phân phối của biến {selected_numeric}"
)


fig3 = px.histogram(
    df,
    x=selected_numeric,
    color="A16",
    marginal="box",
    barmode="overlay",
    opacity=0.65,
    title=(
        f"Phân phối {selected_numeric} "
        f"theo kết quả xét duyệt"
    ),
    labels={
        selected_numeric: selected_numeric,
        "A16": "Kết quả"
    }
)


st.plotly_chart(
    fig3,
    use_container_width=True
)


# =========================================================
# CHART 4 - BOX PLOT
# =========================================================

st.header(
    f"4️⃣ Box Plot của {selected_numeric}"
)


fig4 = px.box(
    df,
    x="A16",
    y=selected_numeric,
    color="A16",
    points="outliers",
    title=(
        f"So sánh {selected_numeric} "
        f"giữa các nhóm xét duyệt"
    ),
    labels={
        "A16": "Kết quả xét duyệt",
        selected_numeric: selected_numeric
    }
)


st.plotly_chart(
    fig4,
    use_container_width=True
)


# =========================================================
# CHART 5 - SCATTER PLOT
# =========================================================

st.header("5️⃣ Mối quan hệ giữa hai biến số")


col_x, col_y = st.columns(2)


with col_x:

    x_variable = st.selectbox(
        "Chọn biến X",
        numeric_cols,
        index=0
    )


with col_y:

    y_variable = st.selectbox(
        "Chọn biến Y",
        numeric_cols,
        index=1
    )


scatter_df = df[
    [x_variable, y_variable, "A16"]
].dropna()


fig5 = px.scatter(
    scatter_df,
    x=x_variable,
    y=y_variable,
    color="A16",
    title=(
        f"Mối quan hệ giữa "
        f"{x_variable} và {y_variable}"
    ),
    labels={
        "A16": "Kết quả xét duyệt"
    },
    hover_data=[
        "A16"
    ]
)


st.plotly_chart(
    fig5,
    use_container_width=True
)


# =========================================================
# CHART 6 - MISSING VALUES
# =========================================================

st.header("6️⃣ Phân tích Missing Values")


missing_df = (
    df.isna()
    .sum()
    .reset_index()
)


missing_df.columns = [
    "Variable",
    "Missing"
]


missing_df = missing_df[
    missing_df["Missing"] > 0
]


if len(missing_df) > 0:

    fig6 = px.bar(
        missing_df,
        x="Variable",
        y="Missing",
        title="Số lượng Missing Values theo biến",
        labels={
            "Variable": "Biến",
            "Missing": "Số lượng Missing"
        }
    )

    st.plotly_chart(
        fig6,
        use_container_width=True
    )

else:

    st.success(
        "Dataset không có Missing Values."
    )


# =========================================================
# DATA TABLE
# =========================================================

st.header("📋 Dữ liệu")


with st.expander(
    "Nhấn để xem toàn bộ dữ liệu"
):

    st.dataframe(
        df,
        use_container_width=True,
        height=500
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Data source: UCI Machine Learning Repository - Credit Approval"
)
