"""Network Intrusion Detection Using AI — local Junior Seminar research workspace."""
from datetime import datetime
import hashlib
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve, precision_recall_curve
from utils import (ROOT, MODEL_PATH, FEATURES, NUMERIC, COLORS, clean, labels, predict,
                   evaluate, investigate, summary, signatures, data_quality, metric_calculations,
                   evaluation_errors, FEATURE_DESCRIPTIONS)
from ui import theme, hero, chart, metrics_row, section, score_chart, CYAN, GREEN, RED

st.set_page_config(page_title='Network Intrusion Detection Using AI',page_icon='🛡',layout='wide')
theme()
PAGES = ['Dashboard Overview','Live Traffic Simulation','AI Intrusion Detection',
         'Explainable AI','Model Performance','Data & Methodology']
with st.sidebar:
    st.markdown('<div class="brand">FLOW / LAB</div><div class="brand-sub">Network anomaly research workspace</div>',unsafe_allow_html=True)
    page = st.radio('Workspace',PAGES,label_visibility='collapsed')
    st.divider()
    st.caption('FISK UNIVERSITY · JUNIOR SEMINAR')
    st.caption('A local academic prototype. An unusual flow is not a confirmed attack.')

st.markdown('<div class="eyebrow">Fisk University / Computer Science / Junior Seminar</div>',unsafe_allow_html=True)
st.title('Network Intrusion Detection Using AI')
st.caption('Learn normal behavior. Investigate the unusual. Evaluate the evidence.')

if not MODEL_PATH.exists():
    hero('Your research workspace is ready','Add the prepared UNSW-NB15 CSV files to data/, then train the local model.')
    st.code('./.venv/bin/python train_model.py',language='bash')
    st.stop()


@st.cache_resource
def load_model(stamp):
    return joblib.load(MODEL_PATH)  # Only this locally trained trusted artifact.


@st.cache_data(show_spinner='Analyzing the recorded test flows…')
def load_test(stamp,model_stamp):
    path = ROOT/'data/UNSW_NB15_testing-set.csv'
    frame = pd.read_csv(path)
    result = predict(frame,load_model(model_stamp))
    return result, hashlib.sha256(path.read_bytes()).hexdigest()


model_stamp = MODEL_PATH.stat().st_mtime_ns
try:
    bundle = load_model(model_stamp)
except Exception as error:
    st.error(f'Cannot load model: {error}. Retrain with ./.venv/bin/python train_model.py.')
    st.stop()
# Prevent predictions from a previous model surviving retraining in this browser.
if st.session_state.get('model_stamp') != model_stamp:
    for name in ['uploaded_result','uploaded_metrics','upload_quality','upload_name','replay_key','replay_log']:
        st.session_state.pop(name,None)
    st.session_state.model_stamp = model_stamp
if bundle.get('schema_version',1) < 2:
    st.warning('Dashboard updated. Retrain once to enable calibrated scores, data audits and improved evaluation.')
    st.code('./.venv/bin/python train_model.py')
with st.sidebar:
    st.markdown('<div class="status"><span class="status-dot"></span>Local model online</div>',unsafe_allow_html=True)
    st.caption(f"Isolation Forest · {bundle.get('selected_transform','original')} features")
    st.caption(f"{bundle['normal_training_rows']:,} normal fitting records · seed 42")
    st.caption('Safe replay only. No packet capture or traffic generation.')

baseline = None
test_path = ROOT/'data/UNSW_NB15_testing-set.csv'
if test_path.exists():
    try:
        baseline,test_digest = load_test(test_path.stat().st_mtime_ns,model_stamp)
        if test_digest != bundle.get('sha256',{}).get('test'):
            st.warning('The testing CSV differs from the saved evaluation source. Current traffic charts use the new file; saved performance refers to the original training run.')
    except (ValueError,pd.errors.ParserError,UnicodeDecodeError) as error:
        st.error(f'Testing CSV: {error}')


def current_source(key):
    choices = ['UNSW-NB15 testing set'] if baseline is not None else []
    if 'uploaded_result' in st.session_state:
        choices.append('Latest upload')
    if not choices:
        st.info('Add the testing CSV to data/ or analyze a CSV on AI Intrusion Detection.')
        st.stop()
    source = st.selectbox('Analysis source',choices,key=key)
    return (st.session_state.uploaded_result if source == 'Latest upload' else baseline), source


def reports(result, source, key):
    with st.expander('Export analysis',expanded=False):
        metrics = evaluate(result,result) if 'label' in result else None
        a,b = st.columns(2)
        a.download_button('Download flow report · CSV',result.to_csv(index=False),'traffic_report.csv','text/csv',key=key+'csv')
        b.download_button('Download research summary · TXT',summary(result,bundle,metrics,source),
                          'analysis_summary.txt','text/plain',key=key+'txt')
        st.caption('Includes predictions, signed scores, priorities, record identifiers and available original features.')


def table(result,key):
    a,b = st.columns([1,2])
    priority = a.selectbox('Prediction',['All flows','Anomalous','Normal'],key=key+'filter')
    search = b.text_input('Find record identifier',placeholder='Type an ID…',key=key+'search')
    selected = result if priority == 'All flows' else result[result.prediction == priority]
    if search:
        selected = selected[selected.record_identifier.astype(str).str.contains(search,regex=False)]
    columns = ['record_identifier','prediction','anomaly_score','alert_severity']
    columns += [c for c in ['proto','service','state','dur','sbytes','dbytes','label','attack_cat','detection_time'] if c in selected]
    st.dataframe(selected[columns].head(500),hide_index=True,width='stretch',
        column_config={'anomaly_score':st.column_config.NumberColumn('Anomaly score',format='%.4f'),
                       'record_identifier':'Record ID','alert_severity':'Priority'})
    st.caption(f'{len(selected):,} matching flows · showing up to 500 · full analysis available in exports')


def traffic_metrics(result):
    count = int((result.prediction == 'Anomalous').sum())
    metrics_row([('Analyzed flows',f'{len(result):,}','Rows scored by the saved local pipeline.'),
                 ('Normal',f'{len(result)-count:,}','Predictions at or below the anomaly threshold.'),
                 ('Anomalous',f'{count:,}','Unusual observations; not confirmed attacks.'),
                 ('Anomaly rate',f'{count/len(result):.1%}','Fraction flagged by the model.')])


if page == 'Dashboard Overview':
    result,source = current_source('overview_source')
    hero('Traffic intelligence, grounded in evidence',
         f'{source} · {len(result):,} recorded flows. Scores come from the trained model; labels are used only for evaluation.')
    traffic_metrics(result)
    st.write('')
    a,b = st.columns([.85,1.4])
    with a,st.container(border=True):
        section('Detection profile','Model predictions across the selected source')
        counts = result.prediction.value_counts().rename_axis('prediction').reset_index(name='flows')
        fig = px.pie(counts,names='prediction',values='flows',hole=.76,color='prediction',color_discrete_map=COLORS)
        fig.update_traces(textinfo='none',hovertemplate='%{label}: %{value:,} flows (%{percent})<extra></extra>')
        fig.add_annotation(text=f'<b>{(result.prediction == "Anomalous").mean():.1%}</b><br>flagged',x=.5,y=.5,showarrow=False,font_size=22,font_color='#e2e8f0')
        chart(fig)
    with b,st.container(border=True):
        section('Where normal ends','Positive scores cross the alert threshold. Scores are not probabilities.')
        chart(score_chart(result))
    a,b = st.columns([1.4,.85])
    with a,st.container(border=True):
        section('Protocol footprint','Actual flow counts, split by model prediction')
        grouped = result.groupby(['proto','prediction']).size().reset_index(name='flows')
        top = result.proto.value_counts().head(8).index
        chart(px.bar(grouped[grouped.proto.isin(top)],x='proto',y='flows',color='prediction',
                     color_discrete_map=COLORS,labels={'proto':'Protocol','flows':'Flows'}),height=280)
    with b,st.container(border=True):
        section('Alert priorities','Relative triage bands derived from normal calibration scores')
        severity = result.alert_severity.value_counts().reindex(['Low','Medium','High'],fill_value=0).rename_axis('priority').reset_index(name='flows')
        chart(px.bar(severity,x='flows',y='priority',color='priority',orientation='h',
                     color_discrete_map=COLORS,labels={'flows':'Flagged flows','priority':''}).update_layout(showlegend=False),height=280)
    section('Flow explorer','Inspect individual records and export the full analysis.')
    table(result,'overview')
    reports(result,source,'overview')

elif page == 'Live Traffic Simulation':
    hero('Replay the evidence, one flow at a time','Recorded UNSW-NB15 flows are replayed in a seeded, shuffled order. This is a simulation, not a live network monitor.')
    if baseline is None:
        st.info('Add the prepared testing CSV to data/ to enable replay.')
        st.stop()
    a,b,c = st.columns([1.5,1,1])
    selection = a.selectbox('Replay scenario',['Mixed traffic','Normal traffic','Suspicious traffic'])
    speed = b.slider('Flows per second',1,20,5)
    limit = c.selectbox('Replay length',[100,500,1000],index=1)
    source = baseline
    if selection != 'Mixed traffic':
        source = source.loc[labels(source) == (0 if selection == 'Normal traffic' else 1)]
    # Shuffling prevents sorted CSV labels from making a mixed replay look all-normal.
    source = source.sample(n=min(limit,len(source)),random_state=42).reset_index(drop=True)
    replay_key = (selection,limit,model_stamp)
    if st.session_state.get('replay_key') != replay_key:
        st.session_state.update(replay_key=replay_key,cursor=0,running=False,replay_log=[])
    a,b,c,_ = st.columns([1,1,1,3])
    if a.button('Start replay',type='primary',width='stretch'):
        st.session_state.running = st.session_state.cursor < len(source)
    if b.button('Pause',width='stretch'):
        st.session_state.running = False
    if c.button('Reset',width='stretch'):
        st.session_state.update(cursor=0,running=False,replay_log=[])
    st.caption('Scenario selection uses ground-truth labels. Detection is still performed by the saved model. Replay timestamps are processing times, not original capture times.')

    @st.fragment(run_every='1s')
    def replay():
        if st.session_state.running and st.session_state.cursor < len(source):
            start = st.session_state.cursor
            original = source.iloc[start:start+speed].drop(columns=['prediction','anomaly_score','raw_anomaly_score','alert_severity','record_identifier'],errors='ignore')
            batch = predict(original,bundle)
            batch['detection_time'] = datetime.now().astimezone().isoformat(timespec='seconds')
            st.session_state.replay_log.extend(batch.to_dict('records'))
            st.session_state.cursor += len(batch)
        if st.session_state.cursor >= len(source):
            st.session_state.running = False
        state = 'Running' if st.session_state.running else ('Complete' if st.session_state.cursor == len(source) else 'Paused')
        st.progress(st.session_state.cursor/max(len(source),1),text=f'{state} · {st.session_state.cursor} / {len(source)} flows')
        if not st.session_state.replay_log:
            st.info('Choose a scenario, then start replay.')
            return
        log = pd.DataFrame(st.session_state.replay_log)
        traffic_metrics(log)
        a,b = st.columns([1.5,1])
        with a,st.container(border=True):
            section('Detection timeline','Replay order · scores above zero trigger alerts')
            fig = px.line(log.reset_index(names='replay_order'),x='replay_order',y='anomaly_score',
                          labels={'replay_order':'Replay order','anomaly_score':'Anomaly score'})
            fig.add_hline(y=0,line_dash='dash',line_color=RED)
            chart(fig)
        with b,st.container(border=True):
            section('Latest alert')
            alerts = log[log.prediction == 'Anomalous']
            if alerts.empty:
                st.success('No anomalous flows detected so far.')
            else:
                row = alerts.iloc[-1]
                message = f'Suspicious Network Activity Detected · {row.alert_severity} priority'
                (st.error if row.alert_severity == 'High' else st.warning if row.alert_severity == 'Medium' else st.info)(message)
                st.write(f'**Record {row.record_identifier}** · Score **{row.anomaly_score:.4f}**')
                st.caption(row.detection_time)
                unusual = investigate(row,bundle)
                st.write('Most unusual numeric features: **'+', '.join(unusual.feature.head(3))+'**')
                st.caption('Priority indicates statistical unusualness, not confirmed attack impact.')
        st.dataframe(log[['record_identifier','prediction','anomaly_score','alert_severity','detection_time']].tail(12).iloc[::-1],hide_index=True,width='stretch')
        if not st.session_state.running:
            reports(log,'Safe dataset replay','replay')
        else:
            st.caption('Pause to export the replay report.')
    replay()

elif page == 'AI Intrusion Detection':
    hero('Bring your own network-flow records','Validate the data contract, score each flow with the saved pipeline, and export an inspectable result. Nothing leaves your laptop.')
    with st.expander('CSV schema and processing rules'):
        st.code(', '.join(FEATURES))
        with st.expander('Feature dictionary and example values'):
            dictionary = pd.DataFrame([{'feature': name, 'type': 'numeric' if name in NUMERIC else 'categorical',
                'meaning': FEATURE_DESCRIPTIONS[name]} for name in FEATURES])
            st.dataframe(dictionary, hide_index=True, width='stretch')
            st.caption('Flow summaries describe connections. The CSV is not a raw packet capture. Feature selection is fixed manually and has not been established as optimal.')
        st.write('All 26 features are required. Blank/infinite numeric cells use training medians. Nonnumeric and negative numeric values are rejected. Categories are trimmed and lowercased. Unknown categories map to the trained infrequent bucket where one exists. Extra columns are preserved but excluded from model input.')
        st.caption('Optional label must be 0 or 1 for evaluation. Upload precomputed flows, not raw packet captures.')
    upload = st.file_uploader('Choose a network-flow CSV',type='csv')
    if upload and st.button('Validate & analyze',type='primary'):
        try:
            frame = pd.read_csv(upload)
            quality = data_quality(frame)
            result = predict(frame,bundle)
            metrics = evaluate(frame,result) if 'label' in frame else None
            st.session_state.update(uploaded_result=result,uploaded_metrics=metrics,
                                    upload_quality=quality,upload_name=upload.name)
            st.success(f'Validation passed · {len(result):,} flows analyzed.')
            unknown = {c:int((~clean(frame)[c].isin(values)).sum()) for c,values in bundle.get('categories',{}).items()}
            if any(unknown.values()):
                st.warning(f'Previously unseen category values by column: {unknown}. Predictions remain available; category novelty may affect reliability.')
        except (ValueError,pd.errors.ParserError,UnicodeDecodeError) as error:
            st.error(str(error))
    if 'uploaded_result' in st.session_state:
        result = st.session_state.uploaded_result
        st.caption(f"Latest successful analysis: {st.session_state.upload_name}")
        traffic_metrics(result)
        quality = st.session_state.upload_quality
        st.caption(f"{quality['missing_numeric_cells']:,} missing numeric cells imputed · {quality['feature_equivalent_rows']:,} repeated model-input rows retained in report")
        table(result,'upload')
        reports(result,st.session_state.upload_name,'upload')

elif page == 'Explainable AI':
    hero('Investigate the anomaly','A statistical comparison with normal training behavior. This supports investigation; it is not an exact explanation of the forest’s decision.')
    result,source = current_source('investigation_source')
    suspicious = result[result.prediction == 'Anomalous'].sort_values('anomaly_score',ascending=False)
    if suspicious.empty:
        st.info('No anomalies detected in this source.')
        st.stop()
    chosen = st.selectbox('Flagged flow · ordered by anomaly score',suspicious.index,
        format_func=lambda i:f"Record {suspicious.loc[i,'record_identifier']} · {suspicious.loc[i,'alert_severity']} · score {suspicious.loc[i,'anomaly_score']:.4f}")
    row = suspicious.loc[chosen]
    a,b,c = st.columns(3)
    a.metric('Anomaly score',f'{row.anomaly_score:.4f}')
    b.metric('Alert priority',row.alert_severity)
    c.metric('Protocol / service',f'{row.proto} / {row.service}')
    comparison = investigate(row,bundle)
    a,b = st.columns([1.4,1])
    with a,st.container(border=True):
        section('What stands out','Signed standard deviations in log1p space reduce the dominance of skewed byte and packet values.')
        chart(px.bar(comparison.sort_values('deviation_std'),x='deviation_std',y='feature',orientation='h',
                     color='deviation_std',color_continuous_scale=['#34d399','#38bdf8','#fb7185'],
                     labels={'deviation_std':'Standard deviations from normal · log1p scale','feature':''}).update_layout(coloraxis_showscale=False))
    with b,st.container(border=True):
        section('Investigation note')
        if not comparison.empty:
            top = comparison.iloc[0]
            st.write(f"**{top.feature}** stands out most among the measured numeric features.")
            st.write(f"Observed: **{top.value:,.3g}** · normal median: **{top.normal_median:,.3g}**.")
            st.write(f"It is **{abs(top.deviation_std):.2f}** log-space standard deviations {'above' if top.deviation_std > 0 else 'below'} the normal reference mean.")
        st.caption('Large deviation does not prove malicious intent. Correlated features are not independent evidence. Missing and constant features are omitted.')
        if 'label' in row:
            st.write('Dataset ground truth: **'+('Attack' if int(row.label) == 1 else 'Normal')+'**')
    section('Reference comparison','Original units below; normal reference uses fitting records only.')
    st.dataframe(comparison,hide_index=True,width='stretch')
    feature = st.selectbox('Inspect normal reference interval',NUMERIC)
    reference = pd.DataFrame(bundle['reference'])
    if 'p05' in reference:
        ref = reference.loc[feature]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[ref.p05,ref.p95],y=['Normal 5–95% interval']*2,mode='lines+markers',line=dict(color=CYAN,width=8),name='Normal reference'))
        fig.add_trace(go.Scatter(x=[row[feature]],y=['Selected flow'],mode='markers',marker=dict(color=RED,size=13,symbol='diamond'),name='Observed value'))
        fig.update_layout(xaxis_title=f'{feature} · original units',showlegend=False)
        chart(fig,height=210)

elif page == 'Model Performance':
    hero('Measure detection, not just accuracy','Independent calibration, explicit false-alarm trade-offs, and separate views of the official split and the feature-overlap audit.')
    views = ['Official test split']
    if bundle.get('strict_metrics'):
        views.append('Unique test inputs absent from training')
    view = st.radio('Evaluation population',views,horizontal=True)
    metrics = bundle['metrics'] if view == views[0] else bundle['strict_metrics']
    st.caption(f"Saved training run · {metrics.get('records',bundle['test_rows']):,} evaluated records · anomalies are the positive class")
    items = [('Precision','precision','Of flagged flows, how many have attack labels?'),
             ('Recall','recall','Of labeled attacks, how many were flagged?'),
             ('F1-score','f1','Harmonic mean of precision and recall.'),
             ('False-positive rate','false_positive_rate','Fraction of normal flows incorrectly flagged.')]
    metrics_row([(name,'N/A' if metrics[key] is None else f'{metrics[key]:.1%}',help_text) for name,key,help_text in items])
    a,b = st.columns([1.1,1])
    with a,st.container(border=True):
        section('Confusion matrix','Actual class on rows · model prediction on columns')
        matrix = metrics['confusion_matrix']
        fig = px.imshow(matrix,text_auto=',d',x=['Normal','Anomalous'],y=['Normal','Attack'],
                        labels={'x':'Predicted class','y':'Actual class','color':'Flows'},color_continuous_scale=['#102338','#278bb1','#70dded'],aspect='auto')
        chart(fig,height=320)
    with b,st.container(border=True):
        section('Research readout')
        tn,fp = metrics['confusion_matrix'][0]; fn,tp = metrics['confusion_matrix'][1]
        st.write(f'**{tp:,} attacks detected** · **{fn:,} attacks missed**')
        st.write(f'**{fp:,} normal flows flagged** · **{tn:,} correctly cleared**')
        st.write(f"Accuracy **{metrics['accuracy']:.1%}** · balanced accuracy **{metrics.get('balanced_accuracy',metrics['accuracy']):.1%}**")
        if metrics.get('roc_auc') is not None:
            st.write(f"ROC-AUC **{metrics['roc_auc']:.3f}** · average precision **{metrics['average_precision']:.3f}**")
            st.caption(f"No-skill average-precision reference = attack prevalence ({metrics['attack_prevalence']:.1%}).")
        st.warning('Benchmark performance is not evidence of readiness for live network deployment.')
        if bundle.get('false_alarm_budget') is not None:
            calibration_fpr = bundle['calibration_metrics']['false_positive_rate']
            st.caption(f"Normal-calibration false-positive rate: {calibration_fpr:.1%}; nominal budget: {bundle['false_alarm_budget']:.0%}. The budget does not guarantee the test or real-world false-positive rate.")
    with st.expander('How the metrics are calculated'):
        st.caption('Computed with sklearn.metrics from label (0 normal, 1 attack) and the frozen predictions. TP/FP/FN/TN refer to attacks as the positive class.')
        calculations = metric_calculations(metrics)
        st.dataframe(calculations, hide_index=True, width='stretch',
            column_config={'value': st.column_config.NumberColumn('Value (fraction)', format='%.6f')})
        st.caption('ROC-AUC and average precision use continuous raw scores across thresholds; the table above uses one fixed operating threshold.')
    supervised = bundle.get('supervised_baseline')
    if supervised:
        section('Supervised reference comparison', 'The reference uses labeled normal AND attack fitting rows. Isolation Forest fits normal rows only. This is additional supervision, not an isolated algorithm comparison.')
        reference_metrics = supervised['metrics'] if view == views[0] else supervised.get('strict_metrics')
        if reference_metrics:
            rows = []
            for name, measured in [('Isolation Forest (main detector)', metrics), ('Logistic Regression (supervised reference)', reference_metrics)]:
                rows.append({'model': name, **{k: measured[k] for k in ['accuracy', 'precision', 'recall', 'f1', 'false_positive_rate', 'roc_auc', 'average_precision']}})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
            st.caption('Same selected features and feature-group partitions. Logistic Regression has separate imputation, log1p, standardization and one-hot encoding fitted on its fitting pool only. Fixed C=1. Both models use the same nominal normal-calibration false-alarm budget; neither threshold is tuned on the test set.')
    if baseline is not None and test_digest == bundle.get('sha256',{}).get('test') and view == views[0]:
        with st.expander('Inspect false positives and missed attacks'):
            errors = evaluation_errors(baseline)
            kind = st.selectbox('Error type', ['False positive', 'False negative'])
            selected_errors = errors[errors.error_type == kind]
            st.caption(f'{len(selected_errors):,} actual {kind.lower()} records; showing up to 50. These are error examples, not model explanations.')
            cols = ['record_identifier', 'error_type', 'prediction', 'label', 'anomaly_score', 'proto', 'service', 'dur', 'sbytes', 'dbytes']
            st.dataframe(selected_errors[cols].head(50), hide_index=True, width='stretch')
            st.download_button('Download all evaluation errors · CSV', errors.to_csv(index=False), 'evaluation_errors.csv', 'text/csv')
    if baseline is not None and test_digest == bundle.get('sha256',{}).get('test') and view == views[0]:
        a,b = st.columns(2)
        actual = labels(baseline)
        fpr,tpr,_ = roc_curve(actual,baseline.raw_anomaly_score)
        precision,recall,_ = precision_recall_curve(actual,baseline.raw_anomaly_score)
        with a,st.container(border=True):
            section('ROC curve','Ranking performance across thresholds · current operating point marked')
            fig = go.Figure(go.Scatter(x=fpr,y=tpr,name='Isolation Forest',line_color=CYAN))
            fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',name='Chance',line=dict(color='#52657f',dash='dash')))
            fig.add_trace(go.Scatter(x=[metrics['false_positive_rate']],y=[metrics['recall']],mode='markers',name='Saved threshold',marker=dict(color=RED,size=10)))
            fig.update_layout(xaxis_title='False-positive rate',yaxis_title='True-positive rate')
            chart(fig)
        with b,st.container(border=True):
            section('Precision–recall curve','The threshold is frozen; this chart does not tune the test set.')
            fig = go.Figure(go.Scatter(x=recall,y=precision,name='Isolation Forest',line_color=CYAN))
            fig.add_hline(y=float(actual.mean()),line_dash='dash',line_color='#52657f')
            fig.add_trace(go.Scatter(x=[metrics['recall']],y=[metrics['precision']],mode='markers',name='Saved threshold',marker=dict(color=RED,size=10)))
            fig.update_layout(xaxis_title='Recall',yaxis_title='Precision')
            chart(fig)
    if view == views[0] and bundle.get('attack_breakdown'):
        with st.container(border=True):
            section('Detection by attack family','Official test split · these labels evaluate the model; they are never inputs.')
            family = pd.DataFrame(bundle['attack_breakdown']).sort_values('recall')
            fig = px.bar(family,x='recall',y='attack_family',orientation='h',text='detected',
                         custom_data=['total'],labels={'recall':'Recall','attack_family':''})
            fig.update_traces(hovertemplate='%{y}: %{x:.1%}<br>Detected %{text:,} / %{customdata[0]:,}<extra></extra>')
            fig.update_xaxes(tickformat='.0%',range=[0,1])
            chart(fig,height=360)
            st.dataframe(family,hide_index=True,width='stretch',column_config={'recall':st.column_config.NumberColumn(format='%.3f')})
    original = bundle.get('original_baseline_metrics')
    if original and view == views[0]:
        section('Experiment comparison','Same official test CSV. The original run was already inspected before this upgrade; this is a development comparison, not a new blind benchmark.')
        comparison = pd.DataFrame([{'experiment':'Original baseline',**{k:original[k] for k in ['accuracy','precision','recall','f1','false_positive_rate']}},
                                   {'experiment':'Audited pipeline',**{k:metrics[k] for k in ['accuracy','precision','recall','f1','false_positive_rate']}}])
        if bundle.get('majority_baseline_metrics'):
            majority = bundle['majority_baseline_metrics']
            comparison = pd.concat([comparison, pd.DataFrame([{'experiment':'Majority-class baseline (always '+('attack' if bundle['majority_baseline_class'] else 'normal')+')',
                **{k:majority[k] for k in ['accuracy','precision','recall','f1','false_positive_rate']}}])],ignore_index=True)
        st.dataframe(comparison,hide_index=True,width='stretch')
    with st.expander('Selection & calibration details'):
        st.write('Raw and log1p transforms are compared using only the training-file selection partition. Choose the highest F1 within a 10% selection false-positive ceiling, or lowest FPR if neither qualifies. Then freeze a new threshold using the separate normal calibration partition (95th percentile by default).')
        st.json({k:bundle[k] for k in ['selected_transform','threshold','severity_thresholds','candidate_selection','calibration_metrics'] if k in bundle})
    st.download_button('Download evaluation record · JSON',(ROOT/'models/evaluation.json').read_text(), 'evaluation.json','application/json')

elif page == 'Data & Methodology':
    hero('A reproducible experiment, not a black box','Follow the records from schema validation to normal-only fitting, independent calibration, and held-out evaluation.')
    if bundle.get('schema_version',1) < 2:
        st.info('Retrain the model to generate the data audit.')
        st.stop()
    metrics_row([('Training-file rows',f"{bundle['training_rows']:,}",'Original prepared training CSV.'),
                 ('Testing-file rows',f"{bundle['test_rows']:,}",'Original prepared testing CSV.'),
                 ('Model features',str(len(FEATURES)),'23 numeric and 3 categorical input columns.'),
                 ('Missing numeric cells',str(bundle['training_quality']['missing_numeric_cells']),'Observed before training-median imputation.')])
    pipeline_tab,quality_tab,repro_tab = st.tabs(['Processing pipeline','Quality & overlap','Reproducibility'])
    with pipeline_tab:
        st.markdown('''
1. **Validate.** Require all selected columns; reject malformed and negative numeric values. Convert infinities and blank numeric cells to missing; normalize categorical whitespace and case.
2. **Group equivalent inputs.** Deduplicate selected feature vectors within training; exclude feature groups with conflicting labels from development. Identical selected inputs stay in one partition.
3. **Split training only.** Stratify unique inputs into 60% fitting, 20% selection and 20% calibration. Train the forest and preprocessing on normal fitting rows only.
4. **Preprocess.** Fit median imputation and category encoding on those same normal rows. Compare raw numeric values with log1p compression for skewed features. Group rare categories (fewer than 10 fitting rows).
5. **Select & calibrate.** Use selection labels to choose the transform, then use independent normal calibration scores to set the alert threshold. Do not refit after calibration.
6. **Evaluate.** Freeze the pipeline before test evaluation. Report the full official test split and a separate subset of unique test inputs absent from the entire training file.
''')
        st.info('The forest learns from normal records only. Transform selection uses labeled validation attacks, so the full experiment is semi-supervised, not fully unsupervised.')
        st.caption('ID, label and attack_cat never enter preprocessing, model fitting or scoring. This is an input-feature grouping strategy, not a patient/session/time split; the prepared CSVs lack sufficient session provenance for that stronger claim.')
        st.code(', '.join(FEATURES))
        with st.expander('Feature dictionary and example values'):
            dictionary = pd.DataFrame([{'feature': name, 'type': 'numeric' if name in NUMERIC else 'categorical',
                'meaning': FEATURE_DESCRIPTIONS[name]} for name in FEATURES])
            st.dataframe(dictionary, hide_index=True, width='stretch')
            st.caption('Flow summaries describe connections. The CSV is not a raw packet capture. Feature selection is fixed manually and has not been established as optimal.')
    with quality_tab:
        audit = bundle['overlap_audit']
        st.warning(f"{audit['test_rows_matching_training_features']:,} test rows match training-file values across the 26 selected inputs. This does not prove they are identical original network connections. Use both evaluation populations when presenting results.")
        st.json(bundle['splits'])
        st.json(audit)
        which = st.radio('Quality table',['Training CSV','Testing CSV'],horizontal=True)
        quality = bundle['training_quality' if which == 'Training CSV' else 'test_quality']
        st.caption(f"{quality['feature_equivalent_rows']:,} repeated selected-feature rows · constant features: {', '.join(quality['constant_features']) or 'none'}")
        st.dataframe(pd.DataFrame(quality['detail']),hide_index=True,width='stretch')
    with repro_tab:
        st.write(f"**Random seed:** {bundle['seed']} · **Training time:** {bundle['training_seconds']:.1f}s · **scikit-learn:** {bundle['sklearn_version']}")
        st.write(f"**Saved at:** {bundle['trained_at']} · **Selected transform:** {bundle['selected_transform']}")
        st.write('**Model:** 250 Isolation Forest trees, 512 sampled records per tree. CPU only. Threshold: positive raw-score excess over the calibrated cutoff. Scores are not probabilities.')
        st.write('**Priority bands:** Low above cutoff; Medium at the median excess of normal calibration alerts; High at their 90th percentile. These are demonstration triage priorities, not attack impact ratings.')
        st.json(bundle['sha256'])
        st.caption('SHA-256 fingerprints identify the exact source CSV bytes. The original model is preserved locally; raw data and model binaries stay out of Git.')
        st.markdown('[Official UNSW-NB15 dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset) · [Isolation Forest reference](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)')
        st.write('**Limitations:** older laboratory benchmark; fixed split/seed; known prior test inspection; selected-feature collisions; no session/time grouping; no live network validation; rare categorical patterns may be missed; statistical explanations are not causal.')
