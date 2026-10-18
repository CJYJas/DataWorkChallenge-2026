"""Read-only scoring/coverage summary; does not choose methods or rewrite forecasts."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline import FLUIDS, metric_rows, write_json, save_csv, digest

def summarize(root):
    root=Path(root);out=root/'08_Review'
    if out.exists():raise FileExistsError('Review already exists; preserve it.')
    out.mkdir();meta=json.load(open(root/'03_Proxy/manifest.json'));scales=np.array(meta['scales'])
    hist=pd.read_csv(root/'06_Field/matched_history.csv',parse_dates=['Date'])
    stage=[]
    for cid,g in hist.groupby('Candidate_ID'):
        g=g.sort_values('Date');n=len(g)
        sections={'All':np.arange(n),'Early':np.array_split(np.arange(n),3)[0],
                  'Middle':np.array_split(np.arange(n),3)[1],'Late':np.array_split(np.arange(n),3)[2],
                  'Last_3_years':np.flatnonzero(g.Date>=g.Date.max()-pd.DateOffset(years=3))}
        for name,idx in sections.items():
            z=g.iloc[idx]
            stage+=metric_rows(z[['Observed_'+f for f in FLUIDS]].to_numpy(),z[FLUIDS].to_numpy(),z.Date.to_numpy(),scales,Candidate_ID=cid,Stage=name)
    save_csv(out,'field_stage_errors.csv',stage)
    summary=[]
    for group in ['04_Validation','05_Test']:
        r=pd.read_csv(root/group/'parameter_recovery.csv')
        for j,p in enumerate(meta['parameters']):
            e=abs(r[p]-r['Actual_'+p]);scale=meta['bounds'][j][1]-meta['bounds'][j][0]
            for obj in r.Objective.unique():
                v=e[r.Objective==obj]
                summary.append({'Split':group,'Objective':obj,'Parameter':p,'Median_absolute_error':v.median(),
                    'Median_range_normalized_error':(v/scale).median(),'Max_range_normalized_error':(v/scale).max()})
    save_csv(out,'parameter_recovery_summary.csv',summary)
    val=pd.read_csv(root/'04_Validation/method_summary.csv');test=pd.read_csv(root/'05_Test/method_summary.csv')
    field=pd.read_csv(root/'06_Field/field_development_backtest.csv');sel=json.load(open(root/'selection_lock.json'))
    checks=[]
    for shift in [0,3]:
        status=json.load(open(root/f'07_Export_Q{shift}/export_status.json'))
        checks.append({'Check':f'Q{shift}: ten case IDs and complete quarterly curves','Pass':status['case_count']==10 and status['rows']==810})
        checks.append({'Check':f'Q{shift}: distinct forecast trajectories','Pass':status['distinct_forecast_trajectories_at_6dp']==10})
    checks.extend([{'Check':'15 validation cases × 2 origins','Pass':len(pd.read_csv(root/'04_Validation/support_coverage.csv'))==30},
        {'Check':'15 test cases × 2 origins','Pass':len(pd.read_csv(root/'05_Test/support_coverage.csv'))==30},
        {'Check':'Selection code fingerprint unchanged','Pass':sel['code_sha256']==digest(Path(__file__).parent/'pipeline.py')}])
    save_csv(out,'technical_checks.csv',checks)
    if not all(c['Pass'] for c in checks):raise ValueError('Technical checks failed; read table.')
    write_json(out/'readiness.json',{'computational_pipeline_executed':True,'technical_checks_passed':True,
       'prediction_quality_accepted':False,'twenty_year_validated':False,'official_submission_ready':False,
       'reasons':['Field transfer still imperfect, including a poor 2005 origin.',
         'Date convention, operations and raw-data audit unresolved.',
         'PPTX, PBIX/video/repository submission and official schema remain separate tasks.']})
    text=['# Run review\n',f"Selected: {sel['objective']} / {sel['method']}\n",
          f"Validation mean score: {val.iloc[0]['Mean']:.6f}; TEST mean score: {test.iloc[0]['Mean']:.6f}.\n",
          'These are team-defined NRMSE scores, not accuracy percentages.\n',
          'Validation comparison:\n',val.to_csv(index=False), '\nField development results:\n',field.to_csv(index=False),
          '\nImportant: field forecast errors vary by origin. No acceptance threshold or independent 20-year truth is available.\n',
          'The close validation tie with direct WOR means added value from HM is not clearly demonstrated.\n',
          'The +3-month hypothesis is not selected solely by its inspected field backtest.\n']
    (out/'RUN_REVIEW.md').write_text('\n'.join(text),encoding='utf-8')
    print('Technical checks passed; forecast quality and competition submission remain separate reviews.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);summarize(p.parse_args().run)
