from __future__ import annotations
import io,re
from pathlib import Path
import duckdb,pandas as pd,polars as pl,streamlit as st

DB=Path('data/mp_ss_changes.duckdb'); SRC='changes'; MAP='module_client_mapping'
H=['SG','PG','MODULE','SUBCAT','SEGMENT','BRAND','BRAND OWNER']
REQ=['NANKEY','DATE']+[f'{x} OLD' for x in H]+[f'{x} NEW' for x in H]
AOB={'','0','N/A','NA','NONE','NULL','NOT APPLICABLE','UNCLASSIFIED','BRAND DETAIL UNKNOWN','BRAND OWNER DETAIL UNKNOWN','BRAND NOT AVAILABLE','NO BRAND LISTED','NO BRAND LISTED (NO COMPANY LISTED)','NO COMPANY LISTED','COMPANY NOT AVAILABLE'}
st.set_page_config(page_title='MP S&S Weekly Changes Report',page_icon='📊',layout='wide')
st.markdown('''<style>
:root{--navy:#0b1f3a;--panel:#102746;--blue:#2f86eb;--line:#6c89a9;--soft:#edf4fb}
.stApp{background:#f3f6fa}.block-container{padding-top:1rem;max-width:1850px}
[data-testid="stSidebar"]{background:var(--navy);border-right:1px solid #38587b}
[data-testid="stSidebar"] *{color:#f5f8ff!important}
[data-testid="stSidebar"] [data-baseweb="select"]>div,[data-testid="stSidebar"] input{background:#102746!important;border-color:#607fa2!important}
h1,h2,h3{color:var(--navy)}
[data-testid="stMetric"]{background:var(--panel);border:1px solid #6f8fb3;padding:13px 15px;border-radius:8px;box-shadow:0 2px 7px #10274622}
[data-testid="stMetricLabel"] p,[data-testid="stMetricValue"]{color:white!important}
div[data-testid="stDataFrame"]{border:1px solid #9db2c9;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px #10274618}
.stButton button,.stDownloadButton button{background:var(--blue);color:white;border:0;border-radius:6px;font-weight:650}
</style>''',unsafe_allow_html=True)

def canon(x):return re.sub(r'[\s_]+',' ',str(x or '').replace('\u00a0',' ').strip().upper())
def clean(c):return pl.col(c).cast(pl.Utf8,strict=False).fill_null('').str.replace_all('\u00a0',' ').str.replace_all(r'[\x00-\x1F\x7F]',' ').str.replace_all(r'\s+',' ').str.strip_chars()
def read_file(f,sheet=None):
 e=Path(f.name).suffix.lower(); b=io.BytesIO(f.getvalue())
 if e=='.csv':return pl.read_csv(b,infer_schema_length=10000,ignore_errors=True)
 if e=='.parquet':return pl.read_parquet(b)
 return pl.read_excel(b,sheet_name=sheet or 0,engine='calamine')
def norm(df):
 r={c:canon(c) for c in df.columns}
 if len(set(r.values()))!=len(r):raise ValueError('Duplicate columns after header normalization.')
 return df.rename(r)
def derive(df):
 df=norm(df); miss=[c for c in REQ if c not in df.columns]
 if miss:raise ValueError('Missing columns: '+', '.join(miss))
 df=df.with_columns([clean(c).alias(c) for c in df.columns if c!='DATE'])
 df=df.with_columns(pl.col('DATE').cast(pl.Utf8,strict=False).str.to_datetime(strict=False).alias('_D'))
 if df.select(pl.col('_D').is_null().sum()).item()==df.height:df=df.with_columns((pl.datetime(1899,12,30)+pl.duration(days=pl.col('DATE').cast(pl.Float64,strict=False))).alias('_D'))
 if df.select(pl.col('_D').is_null().sum()).item():raise ValueError('Some DATE values are invalid.')
 flags=[];parts=[]
 for x in H:
  ch=clean(f'{x} OLD').str.to_uppercase()!=clean(f'{x} NEW').str.to_uppercase()
  flags.append(pl.when(ch).then(pl.lit('CHANGED')).otherwise(pl.lit('NO CHANGE')).alias(f'{x} FLAG'))
  parts.append(pl.when(ch).then(pl.lit(x)).otherwise(pl.lit(None,dtype=pl.Utf8)))
 df=df.with_columns(flags).with_columns(pl.concat_list(parts).list.drop_nulls().list.join(', ').alias('RESULT'),pl.sum_horizontal([(pl.col(f'{x} FLAG')=='CHANGED').cast(pl.Int8) for x in H]).alias('CHANGE COUNT'),pl.when(clean('BRAND OLD').str.to_uppercase().is_in(list(AOB))).then(pl.lit('AOB')).otherwise(pl.lit('BRAND')).alias('BRAND STATUS OLD'),pl.when(clean('BRAND NEW').str.to_uppercase().is_in(list(AOB))).then(pl.lit('AOB')).otherwise(pl.lit('BRAND')).alias('BRAND STATUS NEW'))
 mon=pl.col('_D').dt.date()-pl.duration(days=pl.col('_D').dt.weekday()-1)
 return df.with_columns(pl.when(pl.col('CHANGE COUNT')>0).then(pl.lit('CHANGES')).otherwise(pl.lit('NO CHANGES')).alias('CHANGE LEVEL STATUS'),pl.when(pl.col('RESULT')=='').then(pl.lit('NO CHANGES')).otherwise(pl.col('RESULT')).alias('RESULT'),(pl.col('BRAND STATUS OLD')+pl.lit(' TO ')+pl.col('BRAND STATUS NEW')).alias('BRAND TRANSITION'),pl.col('_D').dt.year().alias('YEAR'),pl.col('_D').dt.strftime('%b %Y').alias('MONTH'),pl.col('_D').dt.strftime('%Y%m').cast(pl.Int32).alias('MONTH SORT'),mon.alias('WEEK START'),(pl.lit('Wk_')+pl.col('_D').dt.week().cast(pl.Utf8)).alias('WEEK'),pl.col('_D').dt.date().alias('DATE')).drop('_D')
def con():DB.parent.mkdir(exist_ok=True);return duckdb.connect(str(DB))
def exists(c,t):return bool(c.execute('select count(*) from information_schema.tables where table_name=?',[t]).fetchone()[0])
def replace(c,t,df):c.register('inc',df.to_arrow());c.execute(f'create or replace table "{t}" as select * from inc');c.unregister('inc')
def append(c,df):
 c.register('inc',df.to_arrow())
 if not exists(c,SRC):c.execute(f'create table "{SRC}" as select * from inc');ins=df.height
 else:
  b=c.execute(f'select count(*) from "{SRC}"').fetchone()[0];c.execute(f'''insert into "{SRC}" by name select i.* from inc i where not exists(select 1 from "{SRC}" x where cast(x.NANKEY as varchar)=cast(i.NANKEY as varchar) and x.DATE=i.DATE)''');ins=c.execute(f'select count(*) from "{SRC}"').fetchone()[0]-b
 c.unregister('inc');return ins,df.height-ins
def q(c,s,p=None):return c.execute(s,p or []).fetchdf()
def vals(c,col,map_=False):
 t=MAP if map_ else SRC;return [str(x[0]) for x in c.execute(f'''select distinct "{col}" from "{t}" where "{col}" is not null and trim(cast("{col}" as varchar))<>'' order by 1''').fetchall()]
def csv(df):return df.to_csv(index=False).encode('utf-8-sig')
def style(df,neg=None,pct=None):
 neg=neg or [];pct=pct or []; fmt={x:'{:,.0f}' for x in df.select_dtypes(include='number').columns}
 for x in pct:
  if x in df:fmt[x]='{:,.2f}%'
 def red(v):
  try:return 'background-color:#fde8e7;color:#8b1e1e;font-weight:700' if float(v)<0 else ''
  except:return ''
 s=df.style.format(fmt,na_rep='-').set_properties(**{'color':'#10213a','border-color':'#d5dfeb'}).set_table_styles([{'selector':'thead th','props':[('background-color','#0b1f3a'),('color','white'),('font-weight','700'),('border','1px solid #54749b')]},{'selector':'tbody tr:nth-child(even)','props':[('background-color','#edf4fb')]},{'selector':'tbody tr:hover','props':[('background-color','#dbeafe')]},{'selector':'td','props':[('padding','7px 9px'),('border','1px solid #d5dfeb')]}])
 for x in neg:
  if x in df:s=s.map(red,subset=[x])
 return s
def show(df,neg=None,pct=None,height=620):
 d=df.copy(); keys=[]
 if 'MONTH' in d:d['_MS']=pd.to_datetime(d.MONTH,format='%b %Y',errors='coerce');keys+=['_MS']
 if 'WEEK START' in d:keys+=['WEEK START']
 if 'WEEK' in d:d['_WS']=pd.to_numeric(d.WEEK.astype(str).str.extract(r'(\d+)')[0],errors='coerce');keys+=['_WS']
 for x in ['DIMENSION','MODULE','CLIENT']:
  if x in d:keys+=[x]
 if keys:d=d.sort_values(keys,kind='stable')
 d=d.drop(columns=[x for x in ['_MS','_WS'] if x in d])
 st.dataframe(style(d,neg,pct),use_container_width=True,hide_index=True,height=height)
def filters(c,key):
 st.sidebar.markdown('### Filters'); f={}
 for col,label in [('WEEK','Week'),('SG NEW','SG'),('PG NEW','PG'),('MODULE NEW','Module'),('MONTH','Month')]:f[col]=st.sidebar.multiselect(label,vals(c,col),key=key+col)
 if exists(c,MAP):f['CLIENT']=st.sidebar.multiselect('Client',vals(c,'CLIENT',True),key=key+'CLIENT')
 return f
def base(c,f):
 clauses=['true'];p=[]
 for col,v in f.items():
  if v:clauses.append(('m.CLIENT' if col=='CLIENT' else f'c."{col}"')+f' in ({",".join(["?"]*len(v))})');p+=v
 join=f'left join "{MAP}" m on upper(trim(c."MODULE NEW"))=upper(trim(m.MODULE))' if exists(c,MAP) else ''
 client=',m.CLIENT' if exists(c,MAP) else ',null::varchar CLIENT'
 return f'select c.* {client} from "{SRC}" c {join} where '+ ' and '.join(clauses),p
def week_select(c,key):
 w=q(c,f'''select WEEK,min("WEEK START") WS from "{SRC}" group by WEEK order by WS''').WEEK.tolist();a,b=st.columns(2);return a.selectbox('Wk_Base',['ALL']+w,key=key+'b'),b.selectbox('Wk_Targeted',['ALL']+w,key=key+'t')
def require(c):
 if not exists(c,SRC):st.warning('Open Data Management and load the database.');st.stop()

def admin():
 st.title('Data Management');c=con()
 if exists(c,SRC):
  r=c.execute(f'''select count(*),count(distinct WEEK),min("WEEK START"),max("WEEK START") from "{SRC}"''').fetchone();latest=c.execute(f'''select WEEK from "{SRC}" order by "WEEK START" desc limit 1''').fetchone()[0]
 else:r=(0,0,None,None);latest='No data'
 for col,(n,v) in zip(st.columns(4),[('Rows',r[0]),('Weeks',r[1]),('Latest Week',latest),('Latest Date',r[3] or 'No data')]):col.metric(n,f'{v:,}' if isinstance(v,int) else str(v))
 if r[0]:st.success('Saved database is loaded automatically. Upload again only when updating the data.')
 with st.expander('Upload or update source data',expanded=not bool(r[0])):
  f=st.file_uploader('Consolidation data',type=['csv','parquet','xlsx','xlsm']);sh=st.text_input('Source sheet','Q1 Consolidation data');mode=st.radio('Load mode',['Append only new NANKEY + DATE','Replace complete database'],horizontal=True)
  if st.button('Process source',disabled=f is None,type='primary'):
   try:
    d=derive(read_file(f,sh if Path(f.name).suffix.lower() in {'.xlsx','.xlsm'} else None))
    if mode.startswith('Replace'):replace(c,SRC,d);ins,skip=d.height,0
    else:ins,skip=append(c,d)
    st.success(f'{ins:,} inserted; {skip:,} overlapping rows skipped.');st.rerun()
   except Exception as e:st.exception(e)
 with st.expander('Upload or refresh Module and Client mapping'):
  f=st.file_uploader('Mapping file',type=['csv','parquet','xlsx','xlsm'],key='map');sh=st.text_input('Mapping sheet','Module and Client mapping')
  if st.button('Replace mapping',disabled=f is None):
   d=norm(read_file(f,sh if Path(f.name).suffix.lower() in {'.xlsx','.xlsm'} else None));d=d.select(clean('MODULE').alias('MODULE'),clean('CLIENT').alias('CLIENT')).filter((pl.col('MODULE')!='')&(pl.col('CLIENT')!='')).unique();replace(c,MAP,d);st.success(f'{d.height:,} mappings saved.')
 c.close()
def high():
 st.title('High Level Module Insights');c=con();require(c);f=filters(c,'h');bs,p=base(c,f);wb,wt=week_select(c,'h')
 def met(w='ALL'):
  ext='' if w=='ALL' else ' where WEEK=?';pp=p+([] if w=='ALL' else [w]);return q(c,f'''with x as ({bs}) select count(*) TOTAL_ITEMS,sum(case when "CHANGE LEVEL STATUS"='CHANGES' then 1 else 0 end) CHANGES,sum(case when "CHANGE LEVEL STATUS"='NO CHANGES' then 1 else 0 end) NO_CHANGES,sum(case when "BRAND TRANSITION"='AOB TO BRAND' then 1 else 0 end) AOB_TO_BRAND,sum(case when "BRAND TRANSITION"='BRAND TO AOB' then 1 else 0 end) BRAND_TO_AOB from x {ext}''',pp).fillna(0).iloc[0]
 cur=met(wt);labs=['TOTAL_ITEMS','CHANGES','NO_CHANGES','AOB_TO_BRAND','BRAND_TO_AOB'];names=['Total Items','Changes','No Changes','AOB to Brand','Brand to AOB']
 for col,x,n in zip(st.columns(5),labs,names):col.metric(n,f'{int(cur[x]):,}')
 hist=q(c,f'''with x as ({bs}) select "WEEK START",WEEK,count(*) TOTAL_ITEMS,sum(case when "CHANGE LEVEL STATUS"='CHANGES' then 1 else 0 end) CHANGES,sum(case when "CHANGE LEVEL STATUS"='NO CHANGES' then 1 else 0 end) NO_CHANGES,sum(case when "BRAND TRANSITION"='AOB TO BRAND' then 1 else 0 end) AOB_TO_BRAND,sum(case when "BRAND TRANSITION"='BRAND TO AOB' then 1 else 0 end) BRAND_TO_AOB from x group by 1,2''',p);hist['NET_AOB_CHANGE']=hist.AOB_TO_BRAND-hist.BRAND_TO_AOB
 st.subheader('Weekly history');show(hist,['NET_AOB_CHANGE'],height=560);st.download_button('Download history CSV',csv(hist),'high_level_history.csv')
 if wb=='ALL' or wt=='ALL':st.info('Select both Wk_Base and Wk_Targeted to generate the week-to-week comparison table.')
 else:
  b,t=met(wb),met(wt);d=pd.DataFrame({'MEASURE':names,wb:[int(b[x]) for x in labs],wt:[int(t[x]) for x in labs]});d['NET_CHANGE']=d[wt]-d[wb];d['DIRECTION']=d.NET_CHANGE.map(lambda x:'NEGATIVE' if x<0 else('POSITIVE' if x>0 else 'NO CHANGE'));st.subheader(f'Week-to-week comparison: {wb} vs {wt}');show(d,['NET_CHANGE'],height=390);st.download_button('Download comparison CSV',csv(d),'week_comparison.csv')
 c.close()
def detail_sql(bs,dim,cond=''):
 return f'''with x as ({bs}) select "{dim}" DIMENSION,WEEK,"WEEK START",count(*) TOTAL_ITEMS,sum(case when "CHANGE LEVEL STATUS"='CHANGES' then 1 else 0 end) CHANGES,sum(case when "CHANGE LEVEL STATUS"='NO CHANGES' then 1 else 0 end) NO_CHANGES,sum(case when "SG FLAG"='CHANGED' then 1 else 0 end) SG,sum(case when "PG FLAG"='CHANGED' then 1 else 0 end) PG,sum(case when "MODULE FLAG"='CHANGED' then 1 else 0 end) MOD,sum(case when "SUBCAT FLAG"='CHANGED' then 1 else 0 end) SUBCAT,sum(case when "SEGMENT FLAG"='CHANGED' then 1 else 0 end) SEGMENT,sum(case when "BRAND FLAG"='CHANGED' then 1 else 0 end) BRAND,sum(case when "BRAND OWNER FLAG"='CHANGED' then 1 else 0 end) BRAND_OWNER,sum(case when "BRAND FLAG"='CHANGED' then 1 else 0 end) BRAND_NO,round(100.0*sum(case when "BRAND FLAG"='CHANGED' then 1 else 0 end)/nullif(count(*),0),2) BRAND_PCT,sum(case when "BRAND TRANSITION" in ('AOB TO BRAND','BRAND TO AOB') then 1 else 0 end) AOB_NO,round(100.0*sum(case when "BRAND TRANSITION" in ('AOB TO BRAND','BRAND TO AOB') then 1 else 0 end)/nullif(count(*),0),2) AOB_PCT,sum("CHANGE COUNT") TOTAL_CHANGED from x {cond} group by 1,2,3'''
def detailed():
 st.title('Parameter-wise Changes Dashboard');c=con();require(c);f=filters(c,'d');bs,p=base(c,f);wb,wt=week_select(c,'d');g=st.selectbox('Display rows by',['MODULE NEW','PG NEW','SG NEW','CLIENT'],format_func=lambda x:x.replace(' NEW',''));dim='CLIENT' if g=='CLIENT' else g
 d=q(c,detail_sql(bs,dim),p);st.subheader('All available weeks');show(d,pct=['BRAND_PCT','AOB_PCT'],height=660);st.download_button('Download detailed CSV',csv(d),'parameter_changes.csv')
 meas=['TOTAL_ITEMS','CHANGES','NO_CHANGES','SG','PG','MOD','SUBCAT','SEGMENT','BRAND','BRAND_OWNER','BRAND_NO','AOB_NO','TOTAL_CHANGED']
 if wb=='ALL' or wt=='ALL':st.info('Select both Wk_Base and Wk_Targeted to generate the parameter week-to-week comparison table.')
 else:
  x=q(c,detail_sql(bs,dim,'where WEEK in (?,?)'),p+[wb,wt]);pv=x.pivot_table(index='DIMENSION',columns='WEEK',values=meas,aggfunc='sum',fill_value=0);rows=[]
  for z in pv.index:
   r={'DIMENSION':z}
   for m in meas:r[f'{m}_BASE']=pv.loc[z].get((m,wb),0);r[f'{m}_TARGET']=pv.loc[z].get((m,wt),0);r[f'{m}_NET']=r[f'{m}_TARGET']-r[f'{m}_BASE']
   rows.append(r)
  out=pd.DataFrame(rows);nets=[x for x in out if x.endswith('_NET')];st.subheader(f'Week-to-week parameter comparison: {wb} vs {wt}');show(out,nets,height=700);st.download_button('Download parameter comparison CSV',csv(out),'parameter_week_comparison.csv')
 c.close()
def mapping():
 st.title('Module and Client Mapping');c=con()
 if not exists(c,MAP):st.warning('Load mapping in Data Management.');st.stop()
 m=st.multiselect('Module',vals(c,'MODULE',True));cl=st.multiselect('Client',vals(c,'CLIENT',True));w=['true'];p=[]
 if m:w.append(f'MODULE in ({",".join(["?"]*len(m))})');p+=m
 if cl:w.append(f'CLIENT in ({",".join(["?"]*len(cl))})');p+=cl
 d=q(c,f'''select MODULE,CLIENT from "{MAP}" where {' and '.join(w)} order by MODULE,CLIENT''',p);show(d,height=680);st.download_button('Download mapping CSV',csv(d),'module_client_mapping.csv');c.close()
P={'Data Management':admin,'High Level Insights':high,'Parameter Changes':detailed,'Module-Client Mapping':mapping}
st.sidebar.title('MP S&S REPORT');page=st.sidebar.radio('Page',list(P),index=1);P[page]()
