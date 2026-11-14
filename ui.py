"""Small reusable presentation layer for the Streamlit dashboard."""
import html
import plotly.graph_objects as go
import streamlit as st

CYAN = '#38bdf8'
GREEN = '#34d399'
RED = '#fb7185'


def theme():
    st.markdown('''<style>
    .stApp {background:#080f1d;color:#e2e8f0}
    .block-container {max-width:1440px;padding:2rem 2.5rem 3rem}
    [data-testid="stSidebar"] {background:#0d1728;border-right:1px solid #223047}
    [data-testid="stSidebar"] .block-container {padding-top:2rem}
    h1 {font-size:2.25rem!important;font-weight:650!important;letter-spacing:-.065rem;line-height:1.18!important}
    h2 {font-size:1.6rem!important;letter-spacing:-.035rem}
    h3 {font-size:1.1rem!important;letter-spacing:-.01rem}
    [data-testid="stCaptionContainer"] {color:#9bacc4}
    [data-testid="stMetric"] {background:linear-gradient(125deg,#122137,#0e192b);border:1px solid #24354d;border-radius:14px;padding:18px 20px;min-height:118px}
    [data-testid="stMetricLabel"] {color:#aabbd1;font-size:.85rem}
    [data-testid="stMetricValue"] {font-size:2rem;font-weight:600;letter-spacing:-.05rem}
    [data-testid="stVerticalBlockBorderWrapper"]>div {border-color:#22334b!important;border-radius:14px!important}
    [data-testid="stSidebar"] [data-testid="stRadio"] label {padding:.25rem 0}
    [data-testid="stDataFrame"] {border-radius:10px;overflow:hidden}
    .eyebrow {font-family:ui-monospace,SFMono-Regular,monospace;color:#75caeb;font-size:.72rem;letter-spacing:.13rem;text-transform:uppercase;margin-bottom:14px}
    .brand {font-size:1.2rem;font-weight:650;letter-spacing:-.03rem;margin-bottom:3px}
    .brand-sub {color:#8298b7;font-size:.78rem;margin-bottom:26px}
    .hero {border-left:3px solid #38bdf8;background:linear-gradient(100deg,#102b40,#0c192a);padding:18px 22px;border-radius:0 12px 12px 0;margin:8px 0 23px}
    .hero strong {color:#def5ff;font-weight:550;font-size:1rem}
    .hero p {color:#9cb2cd;font-size:.88rem;margin:6px 0 0;line-height:1.55}
    .tag {display:inline-block;border:1px solid #26435a;border-radius:100px;padding:4px 10px;font-size:.73rem;color:#a2dff1;background:#0d2435;margin:0 6px 10px 0}
    .status {display:flex;align-items:center;gap:9px;color:#9ce8c9;font-size:.83rem;padding:12px 0}
    .status-dot {width:7px;height:7px;background:#34d399;border-radius:50%;display:inline-block}
    .section-note {color:#94a7c1;font-size:.86rem;margin-bottom:16px}
    @media(max-width:768px){.block-container{padding:1.2rem 1rem}h1{font-size:1.75rem!important}[data-testid="stMetricValue"]{font-size:1.65rem}}
    </style>''', unsafe_allow_html=True)


def hero(title, detail):
    st.markdown(f'<div class="hero"><strong>{html.escape(title)}</strong><p>{html.escape(detail)}</p></div>',unsafe_allow_html=True)


def chart(figure, height=330, key=None):
    figure.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', font=dict(family='Arial, sans-serif',color='#a9bcd5',size=12),
        margin=dict(l=12,r=12,t=35,b=16), height=height,
        legend=dict(orientation='h',y=1.14,x=0,title=None),
        colorway=[CYAN,GREEN,RED,'#fbbf24','#a78bfa'])
    figure.update_xaxes(gridcolor='#1e2d43',zerolinecolor='#34465f')
    figure.update_yaxes(gridcolor='#1e2d43',zerolinecolor='#34465f')
    st.plotly_chart(figure,width='stretch',config={'displaylogo':False},key=key)


def metrics_row(items):
    for col,(name,value,help_text) in zip(st.columns(len(items)),items):
        col.metric(name,value,help=help_text)


def section(title, detail=None):
    st.subheader(title)
    if detail:
        st.caption(detail)


def score_chart(frame):
    fig = go.Figure()
    for name,color in [('Normal',GREEN),('Anomalous',RED)]:
        values = frame.loc[frame.prediction == name,'anomaly_score']
        fig.add_trace(go.Histogram(x=values,name=name,marker_color=color,opacity=.78,nbinsx=55))
    fig.update_layout(barmode='overlay',xaxis_title='Anomaly score · distance above threshold',yaxis_title='Flows')
    fig.add_vline(x=0,line_color='#cbd5e1',line_dash='dash',annotation_text='Alert threshold',annotation_position='top right')
    return fig
