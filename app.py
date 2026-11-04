from __future__ import annotations

import io
import re
from pathlib import Path

import duckdb
import pandas as pd
import polars as pl
import streamlit as st

APP_TITLE = "MP S&S Weekly Changes Report"
DB_PATH = Path("data/mp_ss_changes.duckdb")
SOURCE_TABLE = "changes"
MAPPING_TABLE = "module_client_mapping"

HIERARCHY = ["SG", "PG", "MODULE", "SUBCAT", "SEGMENT", "BRAND", "BRAND OWNER"]
REQUIRED_COLUMNS = ["NANKEY", "DATE"] + [f"{x} OLD" for x in HIERARCHY] + [f"{x} NEW" for x in HIERARCHY]
AOB_VALUES = {
    "", "0", "N/A", "NA", "NONE", "NULL", "NOT APPLICABLE", "UNCLASSIFIED",
    "BRAND DETAIL UNKNOWN", "BRAND OWNER DETAIL UNKNOWN", "BRAND NOT AVAILABLE",
    "BRAND NOT AVAILABLE (COMPANY NOT AVAILABLE)", "NO BRAND LISTED",
    "NO BRAND LISTED (NO COMPANY LISTED)", "NO COMPANY LISTED", "COMPANY NOT AVAILABLE",
}

st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
    :root {--navy:#0b1f3a;--panel:#102746;--blue:#2f86eb;--line:#6e8daf;}
    .stApp {background:#f3f6fa;}
    .block-container {padding-top:1.1rem; padding-bottom:2rem; max-width:1850px;}
    [data-testid="stSidebar"] {background:var(--navy); border-right:1px solid #38587b;}
    [data-testid="stSidebar"] * {color:#f7f9ff !important;}
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] input {background:#102746 !important; border-color:#607fa2 !important;}
    h1,h2,h3 {color:var(--navy);}
    [data-testid="stMetric"] {background:var(--panel); border:1px solid var(--line); padding:13px 15px;
        border-radius:8px; box-shadow:0 2px 7px #10274622;}
    [data-testid="stMetricLabel"] p,[data-testid="stMetricValue"] {color:white !important;}
    div[data-testid="stDataFrame"] {border:1px solid #9db2c9; border-radius:8px; overflow:hidden;
        box-shadow:0 2px 8px #10274618;}
    .stButton button,.stDownloadButton button {background:var(--blue); color:white; border:0;
        border-radius:6px; font-weight:650;}
    </style>
    """,
    unsafe_allow_html=True,
)


def canonical(value: object) -> str:
    return re.sub(r"[\s_]+", " ", str(value or "").replace("\u00a0", " ").strip().upper())


def clean_expr(column: str) -> pl.Expr:
    return (
        pl.col(column).cast(pl.Utf8, strict=False).fill_null("")
        .str.replace_all("\u00a0", " ")
        .str.replace_all(r"[\x00-\x1F\x7F]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def read_upload(upload, sheet_name: str | None = None) -> pl.DataFrame:
    extension = Path(upload.name).suffix.lower()
    source = io.BytesIO(upload.getvalue())
    if extension == ".csv":
        return pl.read_csv(source, infer_schema_length=10000, ignore_errors=True)
    if extension == ".parquet":
        return pl.read_parquet(source)
    if extension in {".xlsx", ".xlsm"}:
        return pl.read_excel(source, sheet_name=sheet_name or 0, engine="calamine")
    raise ValueError("Use CSV, Parquet, XLSX, or XLSM.")


def normalize_headers(df: pl.DataFrame) -> pl.DataFrame:
    rename = {column: canonical(column) for column in df.columns}
    if len(set(rename.values())) != len(rename):
        raise ValueError("Duplicate columns exist after header normalization.")
    return df.rename(rename)


def brand_status_expr(column: str) -> pl.Expr:
    normalized = clean_expr(column).str.to_uppercase()
    return pl.when(normalized.is_in(list(AOB_VALUES))).then(pl.lit("AOB")).otherwise(pl.lit("BRAND"))


def derive_data(df: pl.DataFrame) -> pl.DataFrame:
    df = normalize_headers(df)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    df = df.with_columns([clean_expr(column).alias(column) for column in df.columns if column != "DATE"])
    df = df.with_columns(
        pl.col("DATE").cast(pl.Utf8, strict=False).str.to_datetime(strict=False).alias("_DATE")
    )
    if df.select(pl.col("_DATE").is_null().sum()).item() == df.height:
        df = df.with_columns(
            (pl.datetime(1899, 12, 30) + pl.duration(days=pl.col("DATE").cast(pl.Float64, strict=False))).alias("_DATE")
        )
    if df.select(pl.col("_DATE").is_null().sum()).item() > 0:
        raise ValueError("Some DATE values cannot be converted. Correct the invalid dates and retry.")

    flags: list[pl.Expr] = []
    result_parts: list[pl.Expr] = []
    for level in HIERARCHY:
        changed = clean_expr(f"{level} OLD").str.to_uppercase() != clean_expr(f"{level} NEW").str.to_uppercase()
        flags.append(
            pl.when(changed).then(pl.lit("CHANGED")).otherwise(pl.lit("NO CHANGE")).alias(f"{level} FLAG")
        )
        result_parts.append(pl.when(changed).then(pl.lit(level)).otherwise(pl.lit(None, dtype=pl.Utf8)))

    df = df.with_columns(flags)
    df = df.with_columns(
        pl.concat_list(result_parts).list.drop_nulls().list.join(", ").alias("RESULT"),
        pl.sum_horizontal([(pl.col(f"{level} FLAG") == "CHANGED").cast(pl.Int8) for level in HIERARCHY]).alias("CHANGE COUNT"),
        brand_status_expr("BRAND OLD").alias("BRAND STATUS OLD"),
        brand_status_expr("BRAND NEW").alias("BRAND STATUS NEW"),
    )
    week_start = pl.col("_DATE").dt.date() - pl.duration(days=pl.col("_DATE").dt.weekday() - 1)
    return df.with_columns(
        pl.when(pl.col("CHANGE COUNT") > 0).then(pl.lit("CHANGES")).otherwise(pl.lit("NO CHANGES")).alias("CHANGE LEVEL STATUS"),
        pl.when(pl.col("RESULT") == "").then(pl.lit("NO CHANGES")).otherwise(pl.col("RESULT")).alias("RESULT"),
        (pl.col("BRAND STATUS OLD") + pl.lit(" TO ") + pl.col("BRAND STATUS NEW")).alias("BRAND TRANSITION"),
        pl.col("_DATE").dt.year().alias("YEAR"),
        pl.col("_DATE").dt.strftime("%b %Y").alias("MONTH"),
        pl.col("_DATE").dt.strftime("%Y%m").cast(pl.Int32).alias("MONTH SORT"),
        week_start.alias("WEEK START"),
        (pl.lit("Wk_") + pl.col("_DATE").dt.week().cast(pl.Utf8)).alias("WEEK"),
        pl.col("_DATE").dt.date().alias("DATE"),
    ).drop("_DATE")


def connect_db() -> duckdb.DuckDBPyConnection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DB_PATH))


def table_exists(connection, table_name: str) -> bool:
    return bool(connection.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name=?", [table_name]
    ).fetchone()[0])


def replace_table(connection, table_name: str, df: pl.DataFrame) -> None:
    connection.register("incoming_arrow", df.to_arrow())
    connection.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM incoming_arrow')
    connection.unregister("incoming_arrow")


def append_new_rows(connection, df: pl.DataFrame) -> tuple[int, int]:
    connection.register("incoming_arrow", df.to_arrow())
    if not table_exists(connection, SOURCE_TABLE):
        connection.execute(f'CREATE TABLE "{SOURCE_TABLE}" AS SELECT * FROM incoming_arrow')
        inserted = df.height
    else:
        before = connection.execute(f'SELECT count(*) FROM "{SOURCE_TABLE}"').fetchone()[0]
        connection.execute(f'''
            INSERT INTO "{SOURCE_TABLE}" BY NAME
            SELECT incoming.*
            FROM incoming_arrow incoming
            WHERE NOT EXISTS (
                SELECT 1 FROM "{SOURCE_TABLE}" existing
                WHERE cast(existing.NANKEY AS varchar)=cast(incoming.NANKEY AS varchar)
                  AND existing.DATE=incoming.DATE
            )
        ''')
        after = connection.execute(f'SELECT count(*) FROM "{SOURCE_TABLE}"').fetchone()[0]
        inserted = after - before
    connection.unregister("incoming_arrow")
    return inserted, df.height - inserted


def query_df(connection, sql: str, parameters: list | None = None) -> pd.DataFrame:
    return connection.execute(sql, parameters or []).fetchdf()


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def available_values(connection, column: str, mapping: bool = False) -> list[str]:
    table = MAPPING_TABLE if mapping else SOURCE_TABLE
    if column == "WEEK":
        sql = f'SELECT WEEK FROM "{table}" WHERE WEEK IS NOT NULL GROUP BY WEEK ORDER BY min("WEEK START")'
    elif column == "MONTH":
        sql = f'SELECT MONTH FROM "{table}" WHERE MONTH IS NOT NULL GROUP BY MONTH ORDER BY min("MONTH SORT")'
    else:
        sql = f'''SELECT DISTINCT "{column}" FROM "{table}"
                  WHERE "{column}" IS NOT NULL AND trim(cast("{column}" AS varchar))<>'' ORDER BY 1'''
    return [str(row[0]) for row in connection.execute(sql).fetchall()]


def sidebar_filters(connection, key_prefix: str) -> dict[str, list[str]]:
    st.sidebar.subheader("Filters")
    selections: dict[str, list[str]] = {}
    for column, label in [
        ("WEEK", "Week"), ("SG OLD", "SG"), ("PG OLD", "PG"),
        ("MODULE OLD", "Module"), ("MONTH", "Month"),
    ]:
        selections[column] = st.sidebar.multiselect(
            label, available_values(connection, column), key=f"{key_prefix}_{canonical(column)}"
        )
    if table_exists(connection, MAPPING_TABLE):
        selections["CLIENT"] = st.sidebar.multiselect(
            "Client", available_values(connection, "CLIENT", mapping=True), key=f"{key_prefix}_CLIENT"
        )
    return selections


def filtered_source_sql(connection, selections: dict[str, list[str]]) -> tuple[str, list]:
    clauses = ["TRUE"]
    parameters: list = []
    for column, values in selections.items():
        if values:
            reference = "mapping.CLIENT" if column == "CLIENT" else f'source."{column}"'
            clauses.append(f'{reference} IN ({",".join(["?"] * len(values))})')
            parameters.extend(values)
    if table_exists(connection, MAPPING_TABLE):
        join = f'''LEFT JOIN "{MAPPING_TABLE}" mapping
                   ON upper(trim(source."MODULE OLD"))=upper(trim(mapping.MODULE))'''
        client = ", mapping.CLIENT"
    else:
        join = ""
        client = ", NULL::VARCHAR AS CLIENT"
    sql = f'''SELECT source.* {client}
              FROM "{SOURCE_TABLE}" source {join}
              WHERE {" AND ".join(clauses)}'''
    return sql, parameters


def week_controls(connection, key_prefix: str) -> tuple[str, str]:
    weeks = query_df(
        connection,
        f'SELECT WEEK, min("WEEK START") AS WS FROM "{SOURCE_TABLE}" GROUP BY WEEK ORDER BY WS',
    )["WEEK"].tolist()
    left, right = st.columns(2)
    base_week = left.selectbox("Wk_Base", ["ALL"] + weeks, key=f"{key_prefix}_base")
    target_week = right.selectbox("Wk_Targeted", ["ALL"] + weeks, key=f"{key_prefix}_target")
    return base_week, target_week


def require_source(connection) -> None:
    if not table_exists(connection, SOURCE_TABLE):
        st.warning("No database is loaded. Open Data Management and complete the initial load.")
        st.stop()


def prepare_display(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    for column in ["WEEK START", "DATE"]:
        if column in output.columns:
            converted = pd.to_datetime(output[column], errors="coerce")
            output[column] = converted.apply(
                lambda value: f"{value.month}/{value.day}/{value.year}" if pd.notna(value) else ""
            )
    return output


def show_table(df: pd.DataFrame, negative_columns: list[str] | None = None, height: int = 620, key: str = "table") -> None:
    display_df = prepare_display(df)
    if display_df.empty:
        st.info("No records match the selected filters.")
        return

    page_size = 400
    page_count = max(1, (len(display_df) + page_size - 1) // page_size)
    if page_count > 1:
        page = st.number_input(
            "Table page", min_value=1, max_value=page_count, value=1, step=1, key=f"{key}_page"
        )
        first = (page - 1) * page_size
        last = min(page * page_size, len(display_df))
        st.caption(f"Showing rows {first + 1:,} to {last:,} of {len(display_df):,}")
        view = display_df.iloc[first:last].copy()
    else:
        view = display_df.copy()

    formats = {
        column: "{:,.0f}" for column in view.select_dtypes(include="number").columns
        if not column.endswith("PCT")
    }
    formats.update({column: "{:.2f}%" for column in view.columns if column.endswith("PCT")})
    styled = view.style.format(formats)
    for column in negative_columns or []:
        if column in view.columns:
            styled = styled.map(
                lambda value: "background-color:#fde8e8;color:#9c1c1c;font-weight:600"
                if isinstance(value, (int, float)) and value < 0 else "",
                subset=[column],
            )
    st.dataframe(styled, use_container_width=True, hide_index=True, height=height)


def aggregate_metrics(connection, source_sql: str, parameters: list, week: str | None = None) -> pd.Series:
    week_clause = "" if not week or week == "ALL" else "WHERE WEEK=?"
    query_parameters = parameters + ([] if not week or week == "ALL" else [week])
    result = query_df(connection, f'''
        WITH filtered AS ({source_sql})
        SELECT count(*) AS TOTAL_ITEMS,
               sum(CASE WHEN "CHANGE LEVEL STATUS"='CHANGES' THEN 1 ELSE 0 END) AS CHANGES,
               sum(CASE WHEN "CHANGE LEVEL STATUS"='NO CHANGES' THEN 1 ELSE 0 END) AS NO_CHANGES,
               sum(CASE WHEN "BRAND TRANSITION"='AOB TO BRAND' THEN 1 ELSE 0 END) AS AOB_TO_BRAND,
               sum(CASE WHEN "BRAND TRANSITION"='BRAND TO AOB' THEN 1 ELSE 0 END) AS BRAND_TO_AOB
        FROM filtered {week_clause}
    ''', query_parameters).fillna(0)
    return result.iloc[0]


def detail_sql(source_sql: str, dimension: str, condition: str = "") -> str:
    return f'''
        WITH filtered AS ({source_sql})
        SELECT "{dimension}" AS DIMENSION, WEEK, "WEEK START",
               count(*) AS TOTAL_ITEMS,
               sum(CASE WHEN "CHANGE LEVEL STATUS"='CHANGES' THEN 1 ELSE 0 END) AS CHANGES,
               sum(CASE WHEN "CHANGE LEVEL STATUS"='NO CHANGES' THEN 1 ELSE 0 END) AS NO_CHANGES,
               sum(CASE WHEN "SG FLAG"='CHANGED' THEN 1 ELSE 0 END) AS SG,
               sum(CASE WHEN "PG FLAG"='CHANGED' THEN 1 ELSE 0 END) AS PG,
               sum(CASE WHEN "MODULE FLAG"='CHANGED' THEN 1 ELSE 0 END) AS MOD,
               sum(CASE WHEN "SUBCAT FLAG"='CHANGED' THEN 1 ELSE 0 END) AS SUBCAT,
               sum(CASE WHEN "SEGMENT FLAG"='CHANGED' THEN 1 ELSE 0 END) AS SEGMENT,
               sum(CASE WHEN "BRAND FLAG"='CHANGED' THEN 1 ELSE 0 END) AS BRAND,
               sum(CASE WHEN "BRAND OWNER FLAG"='CHANGED' THEN 1 ELSE 0 END) AS BRAND_OWNER,
               sum(CASE WHEN "BRAND FLAG"='CHANGED' THEN 1 ELSE 0 END) AS BRAND_NO,
               round(100.0*sum(CASE WHEN "BRAND FLAG"='CHANGED' THEN 1 ELSE 0 END)/nullif(count(*),0),2) AS BRAND_PCT,
               sum(CASE WHEN "BRAND TRANSITION" IN ('AOB TO BRAND','BRAND TO AOB') THEN 1 ELSE 0 END) AS AOB_NO,
               round(100.0*sum(CASE WHEN "BRAND TRANSITION" IN ('AOB TO BRAND','BRAND TO AOB') THEN 1 ELSE 0 END)/nullif(count(*),0),2) AS AOB_PCT
        FROM filtered {condition}
        GROUP BY 1,2,3
        ORDER BY 3,1
    '''


def page_high_level() -> None:
    st.title("High Level Module Insights")
    connection = connect_db()
    require_source(connection)
    selections = sidebar_filters(connection, "high")
    source_sql, parameters = filtered_source_sql(connection, selections)
    base_week, target_week = week_controls(connection, "high")

    metric_week = target_week if target_week != "ALL" else None
    metric_values = aggregate_metrics(connection, source_sql, parameters, metric_week)
    metric_columns = st.columns(5)
    for column, label, container in zip(
        ["TOTAL_ITEMS", "CHANGES", "NO_CHANGES", "AOB_TO_BRAND", "BRAND_TO_AOB"],
        ["Total Items", "Changes", "No Changes", "AOB to Brand", "Brand to AOB"],
        metric_columns,
    ):
        container.metric(label, f"{int(metric_values[column]):,}")

    if base_week == "ALL" and target_week == "ALL":
        history = query_df(connection, f'''
            WITH filtered AS ({source_sql})
            SELECT "MODULE OLD" AS MODULE, WEEK, "WEEK START",
                   count(*) AS TOTAL_ITEMS,
                   sum(CASE WHEN "CHANGE LEVEL STATUS"='CHANGES' THEN 1 ELSE 0 END) AS CHANGES,
                   sum(CASE WHEN "CHANGE LEVEL STATUS"='NO CHANGES' THEN 1 ELSE 0 END) AS NO_CHANGES,
                   sum(CASE WHEN "BRAND TRANSITION"='AOB TO BRAND' THEN 1 ELSE 0 END) AS AOB_TO_BRAND,
                   sum(CASE WHEN "BRAND TRANSITION"='BRAND TO AOB' THEN 1 ELSE 0 END) AS BRAND_TO_AOB
            FROM filtered GROUP BY 1,2,3 ORDER BY 3,1
        ''', parameters)
        history["NET_AOB_CHANGE"] = history["AOB_TO_BRAND"] - history["BRAND_TO_AOB"]
        st.subheader("Module history: first available week to latest")
        show_table(history, ["NET_AOB_CHANGE"], key="high_history")
        st.download_button("Download module history CSV", csv_bytes(prepare_display(history)), "module_history.csv", "text/csv")

    elif base_week != "ALL" and target_week != "ALL":
        selected = query_df(connection, f'''
            WITH filtered AS ({source_sql})
            SELECT "MODULE OLD" AS MODULE, WEEK,
                   count(*) AS TOTAL_ITEMS,
                   sum(CASE WHEN "CHANGE LEVEL STATUS"='CHANGES' THEN 1 ELSE 0 END) AS CHANGES,
                   sum(CASE WHEN "CHANGE LEVEL STATUS"='NO CHANGES' THEN 1 ELSE 0 END) AS NO_CHANGES,
                   sum(CASE WHEN "BRAND TRANSITION"='AOB TO BRAND' THEN 1 ELSE 0 END) AS AOB_TO_BRAND,
                   sum(CASE WHEN "BRAND TRANSITION"='BRAND TO AOB' THEN 1 ELSE 0 END) AS BRAND_TO_AOB
            FROM filtered
            WHERE WEEK IN (?,?)
            GROUP BY 1,2
        ''', parameters + [base_week, target_week])

        base_modules = set(selected.loc[selected["WEEK"] == base_week, "MODULE"].dropna())
        target_modules = set(selected.loc[selected["WEEK"] == target_week, "MODULE"].dropna())
        universe = sorted(base_modules | target_modules)
        affected_modules = sorted(target_modules)
        unaffected_modules = sorted(base_modules - target_modules)

        status_filter = st.selectbox(
            "Module availability in targeted week",
            ["All modules", "Affected: module available in targeted week", "Unaffected: module absent in targeted week"],
            key="high_availability_status",
        )
        if status_filter.startswith("Affected"):
            available_modules = affected_modules
        elif status_filter.startswith("Unaffected"):
            available_modules = unaffected_modules
        else:
            available_modules = universe

        module_filter = st.multiselect(
            "Filter modules", available_modules, key="high_module_filter"
        )
        modules_to_show = module_filter if module_filter else available_modules

        measures = ["TOTAL_ITEMS", "CHANGES", "NO_CHANGES", "AOB_TO_BRAND", "BRAND_TO_AOB"]
        output_rows = []
        for module in modules_to_show:
            module_data = selected[selected["MODULE"] == module]
            base_data = module_data[module_data["WEEK"] == base_week]
            target_data = module_data[module_data["WEEK"] == target_week]
            base_values = {measure: int(base_data.iloc[0][measure]) if not base_data.empty else 0 for measure in measures}
            target_values = {measure: int(target_data.iloc[0][measure]) if not target_data.empty else 0 for measure in measures}
            output_rows.append({"MODULE OLD": module, "WEEK / MEASURE": base_week, **base_values})
            output_rows.append({"MODULE OLD": module, "WEEK / MEASURE": target_week, **target_values})
            output_rows.append({
                "MODULE OLD": module,
                "WEEK / MEASURE": "NET CHANGE",
                **{measure: target_values[measure] - base_values[measure] for measure in measures},
            })

        comparison = pd.DataFrame(output_rows)
        st.subheader(f"Module comparison: {base_week} vs {target_week}")
        negative_columns = measures
        show_table(comparison, negative_columns, key="high_week_comparison")
        st.download_button(
            "Download module comparison CSV",
            csv_bytes(comparison),
            "module_week_comparison.csv",
            "text/csv",
        )
    else:
        st.info("Select both Wk_Base and Wk_Targeted, or keep both as ALL.")
    connection.close()


def page_dashboard() -> None:
    st.title("Dashboard")
    connection = connect_db()
    require_source(connection)

    selections = sidebar_filters(connection, "dashboard")
    source_sql, parameters = filtered_source_sql(connection, selections)
    base_week, target_week = week_controls(connection, "dashboard")

    grouping = st.selectbox(
        "Display rows by",
        ["MODULE OLD", "PG OLD", "SG OLD", "CLIENT"],
        format_func=lambda value: value.replace(" OLD", ""),
    )
    dimension = "CLIENT" if grouping == "CLIENT" else grouping
    dimension_label = "MODULE" if dimension == "MODULE OLD" else dimension.replace(" OLD", "")

    # The master universe ignores only the sidebar Week selection. All calculations
    # and availability checks use OLD hierarchy fields.
    universe_filters = {key: value for key, value in selections.items() if key != "WEEK"}
    universe_sql, universe_parameters = filtered_source_sql(connection, universe_filters)
    universe_df = query_df(
        connection,
        f'''WITH universe AS ({universe_sql})
            SELECT DISTINCT "{dimension}" AS DIMENSION
            FROM universe
            WHERE "{dimension}" IS NOT NULL
              AND trim(cast("{dimension}" AS varchar))<>''
            ORDER BY 1''',
        universe_parameters,
    )
    universe_items = sorted(universe_df["DIMENSION"].dropna().tolist())
    sidebar_weeks = selections.get("WEEK", [])

    measures = [
        "SG", "PG", "MOD", "SUBCAT", "SEGMENT", "BRAND", "BRAND_OWNER",
        "BRAND_NO", "BRAND_PCT", "AOB_NO", "AOB_PCT",
    ]

    if base_week != "ALL" and target_week != "ALL":
        weekly = query_df(
            connection,
            detail_sql(universe_sql, dimension, "WHERE WEEK IN (?,?)"),
            universe_parameters + [base_week, target_week],
        )
        available_in_target = set(
            weekly.loc[weekly["WEEK"] == target_week, "DIMENSION"].dropna()
        )
        status_filter = st.selectbox(
            "Module availability status",
            ["All", "Changes", "No Changes"],
            help=(
                "Changes: OLD module/parameter is available in Wk_Targeted. "
                "No Changes: OLD module/parameter is absent in Wk_Targeted."
            ),
            key="dashboard_availability_status",
        )
        if status_filter == "Changes":
            eligible = sorted(available_in_target)
        elif status_filter == "No Changes":
            eligible = sorted(set(universe_items) - available_in_target)
        else:
            eligible = universe_items

        selected_items = st.multiselect(
            f"Filter {dimension_label}", eligible, key="dashboard_dimension_filter"
        )
        items_to_show = selected_items if selected_items else eligible
        rows = []
        for item in items_to_show:
            item_data = weekly[weekly["DIMENSION"] == item]
            base_data = item_data[item_data["WEEK"] == base_week]
            target_data = item_data[item_data["WEEK"] == target_week]
            base_values = {
                measure: float(base_data.iloc[0][measure]) if not base_data.empty else 0
                for measure in measures
            }
            target_values = {
                measure: float(target_data.iloc[0][measure]) if not target_data.empty else 0
                for measure in measures
            }
            rows.append({dimension_label: item, "WEEK / MEASURE": base_week, **base_values})
            rows.append({dimension_label: item, "WEEK / MEASURE": target_week, **target_values})
            rows.append({
                dimension_label: item,
                "WEEK / MEASURE": "NET CHANGE",
                **{measure: target_values[measure] - base_values[measure] for measure in measures},
            })
        detail = pd.DataFrame(rows, columns=[dimension_label, "WEEK / MEASURE", *measures])
        title = f"Parameter comparison: {base_week} vs {target_week}"
        negative_columns = measures

    else:
        if sidebar_weeks:
            title = (
                f"Selected week: {sidebar_weeks[0]}"
                if len(sidebar_weeks) == 1
                else "Selected weeks consolidated: " + ", ".join(sidebar_weeks)
            )
        else:
            title = "All available weeks consolidated: Wk_1 to latest"

        # One output row per OLD module/selected OLD parameter.
        consolidated = query_df(
            connection,
            f'''WITH filtered AS ({source_sql})
                SELECT "{dimension}" AS DIMENSION,
                       sum(CASE WHEN "SG FLAG"='CHANGED' THEN 1 ELSE 0 END) AS SG,
                       sum(CASE WHEN "PG FLAG"='CHANGED' THEN 1 ELSE 0 END) AS PG,
                       sum(CASE WHEN "MODULE FLAG"='CHANGED' THEN 1 ELSE 0 END) AS MOD,
                       sum(CASE WHEN "SUBCAT FLAG"='CHANGED' THEN 1 ELSE 0 END) AS SUBCAT,
                       sum(CASE WHEN "SEGMENT FLAG"='CHANGED' THEN 1 ELSE 0 END) AS SEGMENT,
                       sum(CASE WHEN "BRAND FLAG"='CHANGED' THEN 1 ELSE 0 END) AS BRAND,
                       sum(CASE WHEN "BRAND OWNER FLAG"='CHANGED' THEN 1 ELSE 0 END) AS BRAND_OWNER,
                       sum(CASE WHEN "BRAND FLAG"='CHANGED' THEN 1 ELSE 0 END) AS BRAND_NO,
                       round(100.0*sum(CASE WHEN "BRAND FLAG"='CHANGED' THEN 1 ELSE 0 END)/nullif(count(*),0),2) AS BRAND_PCT,
                       sum(CASE WHEN "BRAND TRANSITION" IN ('AOB TO BRAND','BRAND TO AOB') THEN 1 ELSE 0 END) AS AOB_NO,
                       round(100.0*sum(CASE WHEN "BRAND TRANSITION" IN ('AOB TO BRAND','BRAND TO AOB') THEN 1 ELSE 0 END)/nullif(count(*),0),2) AS AOB_PCT
                FROM filtered
                GROUP BY 1
                ORDER BY 1''',
            parameters,
        )
        available_items = set(consolidated["DIMENSION"].dropna())
        status_filter = st.selectbox(
            "Module availability status",
            ["All", "Changes", "No Changes"],
            help=(
                "Changes: OLD module/parameter is available in the selected week(s). "
                "No Changes: OLD module/parameter is absent in the selected week(s)."
            ),
            key="dashboard_availability_status",
        )
        if status_filter == "Changes":
            eligible = sorted(available_items)
        elif status_filter == "No Changes":
            eligible = sorted(set(universe_items) - available_items)
        else:
            eligible = universe_items

        selected_items = st.multiselect(
            f"Filter {dimension_label}", eligible, key="dashboard_dimension_filter"
        )
        items_to_show = selected_items if selected_items else eligible
        lookup = consolidated.set_index("DIMENSION").to_dict("index") if not consolidated.empty else {}
        rows = []
        for item in items_to_show:
            values = lookup.get(item, {})
            rows.append({
                dimension_label: item,
                **{measure: float(values.get(measure, 0) or 0) for measure in measures},
            })
        detail = pd.DataFrame(rows, columns=[dimension_label, *measures])
        negative_columns = []

    st.subheader(title)
    show_table(detail, negative_columns, key="dashboard_main")
    st.download_button(
        "Download dashboard CSV",
        csv_bytes(prepare_display(detail)),
        "dashboard.csv",
        "text/csv",
    )
    connection.close()


def page_mapping() -> None:
    st.title("Module-Client Mapping")
    connection = connect_db()
    if not table_exists(connection, MAPPING_TABLE):
        st.warning("No mapping is loaded. Open Data Management and upload the mapping sheet.")
        st.stop()
    modules = st.multiselect("Module", available_values(connection, "MODULE", mapping=True))
    clients = st.multiselect("Client", available_values(connection, "CLIENT", mapping=True))
    clauses = ["TRUE"]
    parameters: list = []
    if modules:
        clauses.append(f'MODULE IN ({",".join(["?"] * len(modules))})')
        parameters.extend(modules)
    if clients:
        clauses.append(f'CLIENT IN ({",".join(["?"] * len(clients))})')
        parameters.extend(clients)
    mapping = query_df(
        connection,
        f'SELECT MODULE, CLIENT FROM "{MAPPING_TABLE}" WHERE {" AND ".join(clauses)} ORDER BY MODULE, CLIENT',
        parameters,
    )
    show_table(mapping, height=700, key="mapping")
    st.download_button("Download mapping CSV", csv_bytes(mapping), "module_client_mapping.csv", "text/csv")
    connection.close()


def page_data_management() -> None:
    st.title("Data Management")
    connection = connect_db()
    if table_exists(connection, SOURCE_TABLE):
        status = connection.execute(f'''
            SELECT count(*), count(DISTINCT WEEK), min("WEEK START"), max("WEEK START")
            FROM "{SOURCE_TABLE}"
        ''').fetchone()
        first_date = "" if status[2] is None else f"{status[2].month}/{status[2].day}/{status[2].year}"
        latest_date = "" if status[3] is None else f"{status[3].month}/{status[3].day}/{status[3].year}"
        cards = st.columns(4)
        cards[0].metric("Rows", f"{status[0]:,}")
        cards[1].metric("Weeks", int(status[1]))
        cards[2].metric("First date", first_date)
        cards[3].metric("Latest date", latest_date)

    with st.expander("Upload or update source data", expanded=not table_exists(connection, SOURCE_TABLE)):
        upload = st.file_uploader("Consolidation data", type=["csv", "parquet", "xlsx", "xlsm"], key="source_upload")
        sheet_name = st.text_input("Source sheet", "Q1 Consolidation data")
        mode = st.radio("Load mode", ["Append new NANKEY + DATE", "Replace complete database"])
        if st.button("Process source data", type="primary", disabled=upload is None):
            try:
                with st.status("Processing source data", expanded=True) as status_box:
                    raw = read_upload(upload, sheet_name if Path(upload.name).suffix.lower() in {".xlsx", ".xlsm"} else None)
                    st.write(f"Read {raw.height:,} rows")
                    processed = derive_data(raw)
                    if mode.startswith("Replace"):
                        replace_table(connection, SOURCE_TABLE, processed)
                        inserted, skipped = processed.height, 0
                    else:
                        inserted, skipped = append_new_rows(connection, processed)
                    status_box.update(label=f"Complete: {inserted:,} inserted; {skipped:,} skipped", state="complete")
                st.rerun()
            except Exception as error:
                st.exception(error)

    with st.expander("Upload or refresh Module and Client mapping"):
        mapping_upload = st.file_uploader("Mapping file", type=["csv", "parquet", "xlsx", "xlsm"], key="mapping_upload")
        mapping_sheet = st.text_input("Mapping sheet", "Module and Client mapping")
        if st.button("Replace mapping", disabled=mapping_upload is None):
            try:
                mapping = normalize_headers(read_upload(
                    mapping_upload,
                    mapping_sheet if Path(mapping_upload.name).suffix.lower() in {".xlsx", ".xlsm"} else None,
                ))
                if not {"MODULE", "CLIENT"}.issubset(mapping.columns):
                    raise ValueError("Mapping requires MODULE and CLIENT columns.")
                mapping = mapping.select(
                    clean_expr("MODULE").alias("MODULE"), clean_expr("CLIENT").alias("CLIENT")
                ).filter((pl.col("MODULE") != "") & (pl.col("CLIENT") != "")).unique()
                replace_table(connection, MAPPING_TABLE, mapping)
                st.success(f"Saved {mapping.height:,} unique Module-Client combinations.")
            except Exception as error:
                st.exception(error)
    connection.close()


PAGES = {
    "High Level Insights": page_high_level,
    "Dashboard": page_dashboard,
    "Module-Client Mapping": page_mapping,
    "Data Management": page_data_management,
}
st.sidebar.title("MP S&S REPORT")
selected_page = st.sidebar.radio("Page", list(PAGES), index=0)
PAGES[selected_page]()
