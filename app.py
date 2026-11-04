from __future__ import annotations

import io
import re
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import polars as pl
import streamlit as st

APP_TITLE = "MP S&S Weekly Changes Report"
DB_PATH = Path("data/mp_ss_changes.duckdb")
SOURCE_TABLE = "changes"
MAPPING_TABLE = "module_client_mapping"
HIERARCHY = ["SG", "PG", "MODULE", "SUBCAT", "SEGMENT", "BRAND", "BRAND OWNER"]
REQUIRED = ["NANKEY", "DATE"] + [f"{x} OLD" for x in HIERARCHY] + [f"{x} NEW" for x in HIERARCHY]
AOB_VALUES = {
    "", "0", "N/A", "NA", "NONE", "NULL", "NOT APPLICABLE", "UNCLASSIFIED",
    "BRAND DETAIL UNKNOWN", "BRAND OWNER DETAIL UNKNOWN", "BRAND NOT AVAILABLE",
    "BRAND NOT AVAILABLE (COMPANY NOT AVAILABLE)", "NO BRAND LISTED",
    "NO BRAND LISTED (NO COMPANY LISTED)", "NO COMPANY LISTED", "COMPANY NOT AVAILABLE",
}

st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background:#f5f8fc;border:1px solid #d9e2f2;padding:12px;border-radius:8px;}
</style>
""", unsafe_allow_html=True)


def canon(x) -> str:
    return re.sub(r"[\s_]+", " ", str(x or "").replace("\u00a0", " ").strip().upper())


def clean_expr(c: str) -> pl.Expr:
    return (pl.col(c).cast(pl.Utf8, strict=False).fill_null("")
            .str.replace_all("\u00a0", " ").str.replace_all(r"[\x00-\x1F\x7F]", " ")
            .str.replace_all(r"\s+", " ").str.strip_chars())


def read_upload(upload, sheet: str | None = None) -> pl.DataFrame:
    ext = Path(upload.name).suffix.lower()
    data = upload.getvalue()
    if ext == ".csv":
        return pl.read_csv(io.BytesIO(data), infer_schema_length=10000, ignore_errors=True)
    if ext == ".parquet":
        return pl.read_parquet(io.BytesIO(data))
    if ext in {".xlsx", ".xlsm"}:
        return pl.read_excel(io.BytesIO(data), sheet_name=sheet or 0, engine="calamine")
    raise ValueError("Supported inputs: CSV, Parquet, XLSX, XLSM.")


def normalize_headers(df: pl.DataFrame) -> pl.DataFrame:
    rename = {c: canon(c) for c in df.columns}
    if len(set(rename.values())) != len(rename):
        raise ValueError("Duplicate columns exist after header normalization.")
    return df.rename(rename)


def brand_status(c: str) -> pl.Expr:
    return pl.when(clean_expr(c).str.to_uppercase().is_in(list(AOB_VALUES))).then(pl.lit("AOB")).otherwise(pl.lit("BRAND"))


def derive(df: pl.DataFrame) -> pl.DataFrame:
    df = normalize_headers(df)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    text_cols = [c for c in df.columns if c != "DATE"]
    df = df.with_columns([clean_expr(c).alias(c) for c in text_cols])
    # Try normal date strings first, then Excel serial numbers.
    df = df.with_columns(pl.col("DATE").cast(pl.Utf8, strict=False).str.to_datetime(strict=False).alias("_DATE"))
    if df.select(pl.col("_DATE").is_null().sum()).item() == df.height:
        df = df.with_columns((pl.datetime(1899, 12, 30) + pl.duration(days=pl.col("DATE").cast(pl.Float64, strict=False))).alias("_DATE"))
    if df.select(pl.col("_DATE").is_null().sum()).item() > 0:
        raise ValueError("Some DATE values cannot be converted. Correct the invalid dates and retry.")

    flags, parts = [], []
    for level in HIERARCHY:
        changed = clean_expr(f"{level} OLD").str.to_uppercase() != clean_expr(f"{level} NEW").str.to_uppercase()
        flags.append(pl.when(changed).then(pl.lit("CHANGED")).otherwise(pl.lit("NO CHANGE")).alias(f"{level} FLAG"))
        parts.append(pl.when(changed).then(pl.lit(level)).otherwise(pl.lit(None, dtype=pl.Utf8)))
    df = df.with_columns(flags)
    df = df.with_columns(
        pl.concat_list(parts).list.drop_nulls().list.join(", ").alias("RESULT"),
        pl.sum_horizontal([(pl.col(f"{x} FLAG") == "CHANGED").cast(pl.Int8) for x in HIERARCHY]).alias("CHANGE COUNT"),
        brand_status("BRAND OLD").alias("BRAND STATUS OLD"),
        brand_status("BRAND NEW").alias("BRAND STATUS NEW"),
    )
    monday = pl.col("_DATE").dt.date() - pl.duration(days=pl.col("_DATE").dt.weekday() - 1)
    return df.with_columns(
        pl.when(pl.col("CHANGE COUNT") > 0).then(pl.lit("CHANGES")).otherwise(pl.lit("NO CHANGES")).alias("CHANGE LEVEL STATUS"),
        pl.when(pl.col("RESULT") == "").then(pl.lit("NO CHANGES")).otherwise(pl.col("RESULT")).alias("RESULT"),
        (pl.col("BRAND STATUS OLD") + pl.lit(" TO ") + pl.col("BRAND STATUS NEW")).alias("BRAND TRANSITION"),
        pl.col("_DATE").dt.year().alias("YEAR"),
        pl.col("_DATE").dt.strftime("%b %Y").alias("MONTH"),
        pl.col("_DATE").dt.strftime("%Y%m").cast(pl.Int32).alias("MONTH SORT"),
        monday.alias("WEEK START"),
        (pl.lit("Wk_") + pl.col("_DATE").dt.week().cast(pl.Utf8)).alias("WEEK"),
        pl.col("_DATE").dt.date().alias("DATE"),
    ).drop("_DATE")


def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DB_PATH))


def table_exists(con, name: str) -> bool:
    return bool(con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name=?", [name]).fetchone()[0])


def replace_table(con, name: str, df: pl.DataFrame):
    con.register("incoming", df.to_arrow())
    con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM incoming')
    con.unregister("incoming")


def append_new_only(con, df: pl.DataFrame) -> tuple[int, int]:
    con.register("incoming", df.to_arrow())
    if not table_exists(con, SOURCE_TABLE):
        con.execute(f'CREATE TABLE "{SOURCE_TABLE}" AS SELECT * FROM incoming')
        inserted = df.height
    else:
        before = con.execute(f'SELECT count(*) FROM "{SOURCE_TABLE}"').fetchone()[0]
        # Existing business key: NANKEY + DATE. Prevents reloading an already stored record.
        con.execute(f'''
            INSERT INTO "{SOURCE_TABLE}" BY NAME
            SELECT i.* FROM incoming i
            WHERE NOT EXISTS (
                SELECT 1 FROM "{SOURCE_TABLE}" c
                WHERE cast(c.NANKEY as varchar)=cast(i.NANKEY as varchar) AND c.DATE=i.DATE
            )
        ''')
        after = con.execute(f'SELECT count(*) FROM "{SOURCE_TABLE}"').fetchone()[0]
        inserted = after - before
    con.unregister("incoming")
    return inserted, df.height - inserted


def db_status(con) -> dict:
    if not table_exists(con, SOURCE_TABLE):
        return {"rows": 0, "weeks": 0, "latest": None, "first": None}
    r = con.execute(f'SELECT count(*), count(distinct WEEK), min("WEEK START"), max("WEEK START") FROM "{SOURCE_TABLE}"').fetchone()
    latest = con.execute(f'SELECT WEEK FROM "{SOURCE_TABLE}" ORDER BY "WEEK START" DESC LIMIT 1').fetchone()
    return {"rows": r[0], "weeks": r[1], "first": r[2], "latest_date": r[3], "latest": latest[0] if latest else None}


def qdf(con, sql: str, params=None) -> pd.DataFrame:
    return con.execute(sql, params or []).fetchdf()


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def all_values(con, col: str, mapping=False) -> list[str]:
    table = MAPPING_TABLE if mapping else SOURCE_TABLE
    return [str(x[0]) for x in con.execute(f'SELECT DISTINCT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL AND trim(cast("{col}" as varchar))<>\'\' ORDER BY 1').fetchall()]


def filter_panel(con, prefix: str):
    st.sidebar.markdown("### Filters")
    filters = {}
    specs = [("WEEK","Week"),("SG NEW","SG"),("PG NEW","PG"),("MODULE NEW","Module"),("MONTH","Month")]
    for col, label in specs:
        filters[col] = st.sidebar.multiselect(label, all_values(con, col), key=f"{prefix}_{canon(col)}")
    if table_exists(con, MAPPING_TABLE):
        filters["CLIENT"] = st.sidebar.multiselect("Client", all_values(con, "CLIENT", True), key=f"{prefix}_CLIENT")
    return filters


def qualified_where(filters):
    clauses, params = ["TRUE"], []
    for col, vals in filters.items():
        if vals:
            ref = "m.CLIENT" if col == "CLIENT" else f'c."{col}"'
            clauses.append(f'{ref} IN ({",".join(["?"]*len(vals))})')
            params.extend(vals)
    return " AND ".join(clauses), params


def base_sql(con, filters):
    has_map = table_exists(con, MAPPING_TABLE)
    where, params = qualified_where(filters)
    join = f'LEFT JOIN "{MAPPING_TABLE}" m ON upper(trim(c."MODULE NEW"))=upper(trim(m.MODULE))' if has_map else ""
    client = ", m.CLIENT" if has_map else ", NULL::VARCHAR AS CLIENT"
    return f'SELECT c.* {client} FROM "{SOURCE_TABLE}" c {join} WHERE {where}', params


def comparison_controls(con, key):
    weeks = qdf(con, f'SELECT WEEK, min("WEEK START") AS WS FROM "{SOURCE_TABLE}" GROUP BY WEEK ORDER BY WS')
    options = weeks["WEEK"].tolist()
    c1, c2 = st.columns(2)
    base = c1.selectbox("Wk_Base", ["ALL"] + options, index=0, key=f"{key}_base")
    target = c2.selectbox("Wk_Targeted", ["ALL"] + options, index=0, key=f"{key}_target")
    return base, target


def metric_frame(con, base, params, week=None):
    extra = '' if not week or week == "ALL" else ' WHERE WEEK=? '
    p = params + ([] if not week or week == "ALL" else [week])
    return qdf(con, f'''
        WITH x AS ({base})
        SELECT count(*) TOTAL_ITEMS,
               sum(CASE WHEN "CHANGE LEVEL STATUS"='CHANGES' THEN 1 ELSE 0 END) CHANGES,
               sum(CASE WHEN "CHANGE LEVEL STATUS"='NO CHANGES' THEN 1 ELSE 0 END) NO_CHANGES,
               sum(CASE WHEN "BRAND TRANSITION"='AOB TO BRAND' THEN 1 ELSE 0 END) AOB_TO_BRAND,
               sum(CASE WHEN "BRAND TRANSITION"='BRAND TO AOB' THEN 1 ELSE 0 END) BRAND_TO_AOB
        FROM x {extra}
    ''', p).fillna(0)


def page_data_management():
    st.title("Data Management")
    st.caption("Database upload, append, mapping refresh, and database status are isolated on this page. Dashboard users do not need to upload again after the database is created.")
    con = connect()
    status = db_status(con)
    a,b,c,d = st.columns(4)
    a.metric("Database rows", f"{status['rows']:,}")
    b.metric("Weeks stored", f"{status['weeks']:,}")
    c.metric("Latest week", status.get("latest") or "No data")
    d.metric("Latest week start", str(status.get("latest_date") or "No data"))
    if status["rows"]:
        st.success(f"Database is available from {status['first']} through {status['latest_date']}. Opening the app again will use this saved database automatically.")

    with st.expander("Upload or update source data", expanded=not bool(status["rows"])):
        source = st.file_uploader("Consolidation data", type=["csv","parquet","xlsx","xlsm"], key="source")
        sheet = st.text_input("Source sheet", "Q1 Consolidation data")
        mode = st.radio("Load mode", ["Append only new NANKEY + DATE", "Replace complete database"], horizontal=True)
        if st.button("Process source data", type="primary", disabled=source is None):
            try:
                with st.status("Processing source data", expanded=True) as s:
                    raw = read_upload(source, sheet if Path(source.name).suffix.lower() in {".xlsx",".xlsm"} else None)
                    st.write(f"Read {raw.height:,} rows")
                    processed = derive(raw)
                    if mode.startswith("Replace"):
                        replace_table(con, SOURCE_TABLE, processed)
                        inserted, skipped = processed.height, 0
                    else:
                        inserted, skipped = append_new_only(con, processed)
                    s.update(label=f"Complete: {inserted:,} inserted; {skipped:,} overlapping rows skipped", state="complete")
                st.rerun()
            except Exception as e:
                st.exception(e)

    with st.expander("Upload or refresh Module and Client mapping"):
        mapping = st.file_uploader("Mapping file", type=["csv","parquet","xlsx","xlsm"], key="mapping")
        sheet = st.text_input("Mapping sheet", "Module and Client mapping")
        if st.button("Replace mapping", disabled=mapping is None):
            try:
                mp = normalize_headers(read_upload(mapping, sheet if Path(mapping.name).suffix.lower() in {".xlsx",".xlsm"} else None))
                if not {"MODULE","CLIENT"}.issubset(mp.columns):
                    raise ValueError("Mapping requires MODULE and CLIENT columns.")
                mp = mp.select(clean_expr("MODULE").alias("MODULE"), clean_expr("CLIENT").alias("CLIENT")).filter((pl.col("MODULE")!="")&(pl.col("CLIENT")!="")).unique()
                replace_table(con, MAPPING_TABLE, mp)
                st.success(f"Mapping saved: {mp.height:,} unique Module-Client combinations.")
            except Exception as e:
                st.exception(e)
    con.close()


def require_data(con):
    if not table_exists(con, SOURCE_TABLE):
        st.warning("No database is loaded. Open Data Management and complete the initial load.")
        st.stop()


def page_high_level():
    st.title("High Level Module Insights")
    con = connect(); require_data(con)
    filters = filter_panel(con, "hl")
    base, params = base_sql(con, filters)
    wk_base, wk_target = comparison_controls(con, "hl")

    # Neutral state shows all historical data. Selecting both weeks adds the comparison table.
    current = metric_frame(con, base, params, wk_target if wk_target != "ALL" else None).iloc[0]
    cols = st.columns(5)
    labels = ["TOTAL_ITEMS","CHANGES","NO_CHANGES","AOB_TO_BRAND","BRAND_TO_AOB"]
    names = ["Total Items","Changes","No Changes","AOB to Brand","Brand to AOB"]
    for col, lab, name in zip(cols, labels, names): col.metric(name, f"{int(current[lab]):,}")

    trend = qdf(con, f'''
        WITH x AS ({base})
        SELECT "WEEK START", WEEK,
               count(*) TOTAL_ITEMS,
               sum(CASE WHEN "CHANGE LEVEL STATUS"='CHANGES' THEN 1 ELSE 0 END) CHANGES,
               sum(CASE WHEN "CHANGE LEVEL STATUS"='NO CHANGES' THEN 1 ELSE 0 END) NO_CHANGES,
               sum(CASE WHEN "BRAND TRANSITION"='AOB TO BRAND' THEN 1 ELSE 0 END) AOB_TO_BRAND,
               sum(CASE WHEN "BRAND TRANSITION"='BRAND TO AOB' THEN 1 ELSE 0 END) BRAND_TO_AOB
        FROM x GROUP BY 1,2 ORDER BY 1
    ''', params)
    long = trend.melt(id_vars=["WEEK START","WEEK"], value_vars=labels, var_name="MEASURE", value_name="VALUE")
    fig = px.line(long, x="WEEK START", y="VALUE", color="MEASURE", markers=True, title="History from first available week to latest week")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(trend, use_container_width=True, hide_index=True)
    st.download_button("Download high-level history CSV", csv_bytes(trend), "high_level_history.csv", "text/csv")

    if wk_base != "ALL" and wk_target != "ALL":
        b = metric_frame(con, base, params, wk_base).iloc[0]
        t = metric_frame(con, base, params, wk_target).iloc[0]
        comp = pd.DataFrame({"MEASURE": names, wk_base:[int(b[x]) for x in labels], wk_target:[int(t[x]) for x in labels]})
        comp["NET CHANGE"] = comp[wk_target] - comp[wk_base]
        st.subheader(f"{wk_base} vs {wk_target}")
        st.dataframe(comp, use_container_width=True, hide_index=True)
        st.download_button("Download week comparison CSV", csv_bytes(comp), "high_level_week_comparison.csv", "text/csv")
    con.close()


def detailed_query(base, group_col, week_condition=""):
    return f'''
        WITH x AS ({base})
        SELECT "{group_col}" AS DIMENSION, WEEK, "WEEK START",
               count(*) AS TOTAL_ITEMS,
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
               round(100.0*sum(CASE WHEN "BRAND TRANSITION" IN ('AOB TO BRAND','BRAND TO AOB') THEN 1 ELSE 0 END)/nullif(count(*),0),2) AS AOB_PCT,
               sum("CHANGE COUNT") AS TOTAL_CHANGED
        FROM x {week_condition} GROUP BY 1,2,3 ORDER BY 3,1
    '''


def page_detailed():
    st.title("Parameter-wise Changes Dashboard")
    con = connect(); require_data(con)
    filters = filter_panel(con, "dt")
    base, params = base_sql(con, filters)
    wk_base, wk_target = comparison_controls(con, "dt")
    grouping = st.selectbox("Display rows by", ["MODULE NEW","PG NEW","SG NEW","CLIENT"], format_func=lambda x:x.replace(" NEW",""))
    if grouping == "CLIENT" and not table_exists(con, MAPPING_TABLE):
        st.warning("Client mapping is not loaded."); st.stop()
    group_col = "CLIENT" if grouping == "CLIENT" else grouping

    detail = qdf(con, detailed_query(base, group_col), params)
    st.subheader("All available weeks")
    st.dataframe(detail, use_container_width=True, hide_index=True)
    st.download_button("Download all detailed results CSV", csv_bytes(detail), "parameter_changes_all_weeks.csv", "text/csv")

    measures = ["TOTAL_ITEMS","SG","PG","MOD","SUBCAT","SEGMENT","BRAND","BRAND_OWNER","BRAND_NO","AOB_NO","TOTAL_CHANGED"]
    selected_measure = st.selectbox("Trend measure", measures, index=10)
    fig = px.line(detail, x="WEEK START", y=selected_measure, color="DIMENSION", markers=True, title=f"{selected_measure} trend")
    st.plotly_chart(fig, use_container_width=True)

    if wk_base != "ALL" and wk_target != "ALL":
        both = qdf(con, detailed_query(base, group_col, 'WHERE WEEK IN (?,?)'), params+[wk_base,wk_target])
        pivot = both.pivot_table(index="DIMENSION", columns="WEEK", values=measures, aggfunc="sum", fill_value=0)
        rows=[]
        for dim in pivot.index:
            row={"DIMENSION":dim}
            for m in measures:
                bv = pivot.loc[dim].get((m,wk_base),0)
                tv = pivot.loc[dim].get((m,wk_target),0)
                row[f"{m} BASE"] = bv; row[f"{m} TARGET"] = tv; row[f"{m} NET"] = tv-bv
            rows.append(row)
        comp=pd.DataFrame(rows)
        st.subheader(f"{wk_base} vs {wk_target} parameter comparison")
        st.dataframe(comp, use_container_width=True, hide_index=True)
        st.download_button("Download detailed comparison CSV", csv_bytes(comp), "parameter_week_comparison.csv", "text/csv")
    con.close()


def page_mapping():
    st.title("Module and Client Mapping")
    con=connect()
    if not table_exists(con, MAPPING_TABLE):
        st.warning("No mapping is loaded. Open Data Management and upload the Module and Client mapping sheet."); st.stop()
    modules = st.multiselect("Module", all_values(con,"MODULE",True))
    clients = st.multiselect("Client", all_values(con,"CLIENT",True))
    clauses=["TRUE"]; params=[]
    if modules: clauses.append(f'MODULE IN ({",".join(["?"]*len(modules))})'); params += modules
    if clients: clauses.append(f'CLIENT IN ({",".join(["?"]*len(clients))})'); params += clients
    mp=qdf(con, f'SELECT MODULE, CLIENT FROM "{MAPPING_TABLE}" WHERE {" AND ".join(clauses)} ORDER BY MODULE, CLIENT', params)
    c1,c2,c3=st.columns(3); c1.metric("Mappings",f"{len(mp):,}"); c2.metric("Modules",f"{mp.MODULE.nunique():,}"); c3.metric("Clients",f"{mp.CLIENT.nunique():,}")
    st.dataframe(mp,use_container_width=True,hide_index=True)
    st.download_button("Download filtered mapping CSV",csv_bytes(mp),"module_client_mapping.csv","text/csv")
    con.close()


PAGES = {
    "Data Management": page_data_management,
    "High Level Insights": page_high_level,
    "Parameter Changes": page_detailed,
    "Module-Client Mapping": page_mapping,
}
st.sidebar.title("MP S&S Report")
page = st.sidebar.radio("Page", list(PAGES), index=1)
PAGES[page]()
