"""DataWorks revision: auditable staged development, not a validated 20-year simulator.
Only load joblib files you created/trust. All model selection uses cases 71--85.
The existing cleaned CSV is reused; this does not verify raw grid-cell cleaning.
"""
from __future__ import annotations
import argparse, hashlib, json, platform, sys, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import scipy, sklearn, joblib
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.optimize import least_squares
from scipy.stats import qmc
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PARAMS = ['Fault Transmissibility', 'Porosity Multiplier', 'Permeability Multiplier', 'Aquifer Pore Volume']
FLUIDS = ['Oil_Rate', 'Gas_Rate', 'Water_Rate']
FEATURES = PARAMS + ['Years_Since_Start']
UNITS = ['STB/day', 'MSCF/day', 'STB/day']
METHODS = ['exp_linear_water', 'arps_linear_water', 'wor_constant_liquid', 'wor_declining_liquid', 'hm_wor_constant_liquid', 'hm_wor_declining_liquid']
OBJECTIVES = {
    'equal': {'fluid_weights':[1/3]*3, 'recent_multiplier':1.0, 'slope_lambda':0.0},
    'recent_balanced': {'fluid_weights':[0.45,0.10,0.45], 'recent_multiplier':2.0, 'slope_lambda':0.15},
}
DEFAULTS = {
    'seed':42, 'n_estimators':200, 'pool_power':11, 'refine_per_round':128,
    'cutoffs':['2005-01-01','2006-01-01'], 'horizon_quarters':8,
    'trend_window':9, 'query_shifts':[0,3], 'rate_convention':'instantaneous_assumed',
    'operations_confirmed':False, 'date_rule_confirmed':False,
    'test_previously_inspected':True, 'candidate_relative_allowance':0.25,
    'candidate_absolute_allowance':0.002, 'methods':METHODS, 'objectives':OBJECTIVES,
    'limitations':['Raw Phase 1 cleaning and original Phase 3 training notebook unavailable.',
                   'Full-history TRAIN simulator proxy retained; pseudo-field target future is hidden.',
                   'Only short-horizon tests available, not 20-year validation.',
                   'Bounds are TRAIN extrema, not official physical uncertainty bounds.',
                   'Case numbering split follows supplied bundle; verify official ID assignments.',
                   'Timing and future operations remain hypotheses. No calibrated probabilities.'],
}

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(p, obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
def read_json(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def save_csv(root, name, data):
    p=Path(root)/name;p.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(data).to_csv(p,index=False)
def version_info():
    return {'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
            'scipy':scipy.__version__,'sklearn':sklearn.__version__,'joblib':joblib.__version__}
def stage_dir(root,name):
    p=Path(root)/name
    if p.exists(): raise FileExistsError(f'{p} already exists. Use a NEW run directory or inspect the existing stage; no silent checkpoint reuse.')
    p.mkdir(parents=True);return p

def setup(source,root):
    root=Path(root)
    if root.exists(): raise FileExistsError('Choose a new output directory.')
    root.mkdir(parents=True);(root/'Inputs').mkdir()
    with zipfile.ZipFile(source) as z:
        for suffix in ['Inputs/linked_parameters_production.csv','Inputs/observed_production.csv']:
            hits=[n for n in z.namelist() if n==suffix or n.endswith('/'+suffix)]
            if len(hits)!=1: raise ValueError(f'Expected exactly one {suffix}, found {hits}')
            (root/suffix).write_bytes(z.read(hits[0]))
    write_json(root/'config.json',DEFAULTS)
    write_json(root/'provenance.json',{'source_zip_sha256':digest(source),'versions':version_info(),
        'code_sha256':digest(__file__), 'inputs':{p.name:digest(p) for p in (root/'Inputs').glob('*.csv')}})
    print('Setup complete:',root,flush=True)

def data(root):
    root=Path(root);cfg=read_json(root/'config.json');prov=read_json(root/'provenance.json')
    for n,h in prov['inputs'].items():
        if digest(root/'Inputs'/n)!=h:raise ValueError('Input changed after setup; use a new run.')
    sim=pd.read_csv(root/'Inputs/linked_parameters_production.csv',parse_dates=['Date'])
    field=pd.read_csv(root/'Inputs/observed_production.csv',parse_dates=['Date']).rename(columns={
        'Oil production rate (STB/d)':'Oil_Rate','Gas production rate (MSCF/d)':'Gas_Rate',
        'Water production rate (STB/d)':'Water_Rate','Water injection rate (STB/d)':'Water_Injection_Rate'})
    sim['Case_Number']=sim['Case'].str.extract(r'^Case_(\d+)$')[0].astype(int)
    sim=sim.sort_values(['Case_Number','Date']).reset_index(drop=True)
    field=field.sort_values('Date').reset_index(drop=True)
    return sim,field,cfg

def validate_data(sim,field):
    if set(sim.Case_Number.unique())!=set(range(1,101)):raise ValueError('Expected Case_1 through Case_100.')
    for frame,keys in [(sim,['Case','Date']),(field,['Date'])]:
        if frame.duplicated(keys).any(): raise ValueError('Duplicate case/date records.')
        if frame.Date.isna().any(): raise ValueError('Missing dates.')
        if not np.isfinite(frame[FLUIDS].to_numpy()).all() or (frame[FLUIDS]<0).any().any():raise ValueError('Invalid rate.')
    if not np.isfinite(sim[PARAMS+['Years_Since_Start']]).all().all():raise ValueError('Nonfinite parameter/time.')
    expected=np.asarray((sim.Date-sim.Date.min()).dt.total_seconds()/86400/365.25)
    if not np.allclose(expected,sim.Years_Since_Start,atol=1e-8):raise ValueError('Time feature inconsistent with dates.')
    dates=sim[sim.Case_Number==1].Date.tolist()
    for _,g in sim.groupby('Case_Number'):
        if g.Date.tolist()!=dates:raise ValueError('Simulation date grids differ.')
        if (g[PARAMS].nunique()!=1).any():raise ValueError('Rock parameters must remain static within each case.')
    if len(field)<12:raise ValueError('Insufficient field history.')

def derive(g):
    g=g.sort_values('Date').copy();o=g.Oil_Rate.to_numpy();w=g.Water_Rate.to_numpy()
    liquid=o+w
    g['Total_Liquid_Rate']=liquid
    g['Water_Cut']=np.divide(w,liquid,out=np.full(len(g),np.nan),where=liquid>0)
    g['WOR']=np.divide(w,o,out=np.full(len(g),np.nan),where=o>0)
    g['GOR_MSCF_per_STB']=np.divide(g.Gas_Rate,o,out=np.full(len(g),np.nan),where=o>0)
    days=(g.Date-g.Date.iloc[0]).dt.total_seconds().to_numpy()/86400
    g['Np_STB']=cumulative_trapezoid(o,days,initial=0)
    g['Np_MMSTB']=g.Np_STB/1e6
    return g

def plot_curves(path,dates,y,curves,title,cutoff=None):
    fig,axs=plt.subplots(3,1,figsize=(11,9),sharex=True)
    for j,ax in enumerate(axs):
        if y is not None:ax.plot(dates,y[:,j],'k.-',label='Observed / simulator target',lw=1.6,ms=3)
        for name,(dd,qq) in curves.items():ax.plot(dd,qq[:,j],label=name,lw=1.1,alpha=.85)
        if cutoff is not None:ax.axvline(pd.Timestamp(cutoff),color='gray',ls='--')
        ax.set_ylabel(f'{FLUIDS[j]}\n{UNITS[j]}');ax.grid(alpha=.2)
    axs[0].legend(fontsize=7,ncol=2);axs[-1].set_xlabel('Original target date')
    fig.suptitle(title,fontsize=11);fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)

def audit(root):
    sim,field,cfg=data(root);validate_data(sim,field)
    if cfg['rate_convention']!='instantaneous_assumed':
        raise ValueError('This version implements instantaneous-sample trapezoids only. Interval-average data needs explicit interval mapping before use.')
    out=stage_dir(root,'01_Audit');train=sim[sim.Case_Number<=70]
    rows=[]
    for label,g in [('Observed field',field)]+[(f'Case_{i}',g) for i,g in train.groupby('Case_Number')]:
        dg=derive(g)
        for fluid in FLUIDS:
            pos=g[g[fluid]>0]
            rows.append({'History':label,'Fluid':fluid,'First_positive_sample':str(pos.Date.min()),'Peak_date':str(g.loc[g[fluid].idxmax(),'Date'])})
        if label=='Observed field':save_csv(out,'field_derived.csv',dg)
    save_csv(out,'onset_audit.csv',rows)
    d=derive(field);fig,axs=plt.subplots(2,3,figsize=(13,7))
    diagnostics=[('Date','GOR_MSCF_per_STB'),('Date','Total_Liquid_Rate'),('Date','Water_Cut'),('Np_MMSTB','Oil_Rate'),('Np_MMSTB','WOR'),('Np_MMSTB','log1p_WOR')]
    d['log1p_WOR']=np.log1p(d.WOR)
    for ax,(x,y) in zip(axs.flat,diagnostics):
        ax.plot(d[x],d[y],'.-');ax.set(xlabel=x,ylabel=y);ax.grid(alpha=.2)
        if y=='WOR':ax.set_yscale('log')
        if x=='Date':ax.tick_params(axis='x',rotation=25)
    fig.suptitle('Observed-field development diagnostics; not independent validation');fig.tight_layout();fig.savefig(out/'diagnostics.png',dpi=140);plt.close(fig)
    scales=train[FLUIDS].max().to_numpy();baseline=[]
    for case,g in train.groupby('Case_Number'):
        common=field[['Date']+FLUIDS].merge(g[['Date']+FLUIDS],on='Date',suffixes=('_obs','_sim'))
        y=common[[f+'_obs' for f in FLUIDS]].to_numpy();q=common[[f+'_sim' for f in FLUIDS]].to_numpy()
        nr=np.sqrt(np.mean((q-y)**2,axis=0))/scales
        baseline.append({'Case':case,'Score':nr.mean(),**dict(zip(FLUIDS,nr))})
    ranks=pd.DataFrame(baseline).sort_values('Score');save_csv(out,'training_baseline_ranking.csv',ranks)
    best=sim[sim.Case_Number==int(ranks.iloc[0].Case)]
    plot_curves(out/'baseline_match.png',field.Date,field[FLUIDS].to_numpy(),{'Best existing training case':(best.Date,best[FLUIDS].to_numpy())},'Phase 2 baseline; unshifted dates')
    checks=[{'Item':'100 cases, static parameters, complete date grids, nonnegative rates','Status':'PASS'},
        {'Item':'Original raw Phase 1 cleaning / grid mapping','Status':'NOT VERIFIED'},
        {'Item':'Instantaneous rate interpretation and trapezoids','Status':'ASSUMPTION'},
        {'Item':'Operating schedule comparability','Status':'UNCONFIRMED'},
        {'Item':'Date correction','Status':'UNCONFIRMED'},
        {'Item':'70/15/15 case ID mapping','Status':'FROM EXISTING MANIFEST; verify official assignment'}]
    save_csv(out,'audit_status.csv',checks)
    write_json(out/'audit.json',{'rows':len(sim),'observations':len(field),'train_scales':dict(zip(FLUIDS,scales)),
        'interpolated_counts':{c:int(sim[c].sum()) for c in sim if c.endswith('_Interpolated')},
        'config_sha256':digest(Path(root)/'config.json')})
    print('Audit complete. Read audit_status.csv before interpreting forecasts.',flush=True)

def metric_rows(y,q,dates,scales,**tags):
    days=(pd.DatetimeIndex(dates)-pd.Timestamp(dates[0])).total_seconds().to_numpy()/86400
    rows=[]
    for j,f in enumerate(FLUIDS):
        e=q[:,j]-y[:,j];den=np.sum((y[:,j]-y[:,j].mean())**2)
        rows.append({**tags,'Fluid':f,'N':len(y),'RMSE':np.sqrt(np.mean(e**2)),
                     'NRMSE':np.sqrt(np.mean(e**2))/scales[j],'MAE':np.mean(abs(e)),
                     'Bias':e.mean(),'R2':1-np.sum(e**2)/den if den>0 else np.nan})
    return rows

def train_models(root):
    sim,_,cfg=data(root);validate_data(sim,_)
    out=stage_dir(root,'03_Proxy');(out/'models').mkdir()
    tr=sim[sim.Case_Number<=70];va=sim[sim.Case_Number.between(71,85)]
    scales=tr[FLUIDS].max().to_numpy();rows=[];chosen={};preds=[]
    for j,f in enumerate(FLUIDS):
        options={
            'ExtraTrees':ExtraTreesRegressor(n_estimators=cfg['n_estimators'],random_state=cfg['seed'],n_jobs=2),
            'RandomForest':RandomForestRegressor(n_estimators=cfg['n_estimators'],random_state=cfg['seed'],n_jobs=2),
            'Ridge':make_pipeline(StandardScaler(),Ridge(alpha=1.0)),
            'TimeOnly':ExtraTreesRegressor(n_estimators=cfg['n_estimators'],random_state=cfg['seed'],n_jobs=2)}
        scores={};fitted={}
        for name,model in options.items():
            features=['Years_Since_Start'] if name=='TimeOnly' else FEATURES
            model.fit(tr[features],tr[f]);q=np.maximum(0,model.predict(va[features]));fitted[name]=model
            for case,g in va.assign(Prediction=q).groupby('Case_Number'):
                err=g.Prediction.to_numpy()-g[f].to_numpy();rmse=np.sqrt(np.mean(err**2))
                rows.append({'Fluid':f,'Model':name,'Case':case,'RMSE':rmse,'NRMSE':rmse/scales[j]})
            scores[name]=np.mean([r['NRMSE'] for r in rows if r['Fluid']==f and r['Model']==name])
            print(f,'validation',name,round(scores[name],6),flush=True)
        name=min([n for n in options if n!='TimeOnly'],key=lambda n:(scores[n],n))
        chosen[f]=name;joblib.dump(fitted[name],out/'models'/f'{f}.joblib',compress=3)
    save_csv(out,'validation_case_scores.csv',rows)
    table=pd.DataFrame(rows).groupby(['Fluid','Model'],as_index=False).NRMSE.mean();save_csv(out,'validation_summary.csv',table)
    meta={'selected':chosen,'features':FEATURES,'parameters':PARAMS,'scales':scales.tolist(),
          'bounds':[[float(tr[p].min()),float(tr[p].max())] for p in PARAMS],
          'origin':str(sim.Date.min().date()),'time_max':float(tr.Years_Since_Start.max()),
          'versions':version_info(),'config_sha256':digest(Path(root)/'config.json'),
          'selection':'lowest mean validation-case NRMSE per fluid; TimeOnly diagnostic only',
          'test_used_in_selection':False,'test_previously_inspected':cfg['test_previously_inspected'],
          'training_time_scope':'All historical times of TRAIN cases only'}
    write_json(out/'manifest.json',meta)
    proxy=Proxy(root)
    for case,g in va.groupby('Case_Number'):
        actual=g[PARAMS].iloc[0].to_numpy()
        inside=bool(((actual>=proxy.bounds[:,0])&(actual<=proxy.bounds[:,1])).all())
        # Forward evaluation of known held-out cases includes out-of-box cases;
        # inverse search still strictly obeys TRAIN bounds.
        q=proxy.predict(actual,g.Date,check_bounds=False)[0]
        plot_curves(out/f'validation_Case_{case}.png',g.Date,g[FLUIDS].to_numpy(),{'Selected proxy':(g.Date,q)},f'Simulation validation Case {case}; inside TRAIN box: {inside}')
    write_json(out/'status.json',{'complete':True,'test_scored':False})
    print('Proxy selection complete; no TEST cases scored yet.',flush=True)

class Proxy:
    def __init__(self,root):
        root=Path(root);self.meta=read_json(root/'03_Proxy/manifest.json')
        if self.meta['versions']['sklearn']!=sklearn.__version__:raise ValueError('Install the recorded scikit-learn version or retrain in a new run.')
        if self.meta['config_sha256']!=digest(root/'config.json'):raise ValueError('Config changed after training. Start a new run.')
        self.models=[joblib.load(root/'03_Proxy/models'/f'{f}.joblib') for f in FLUIDS]
        for m in self.models:
            if hasattr(m,'n_jobs'):m.n_jobs=1
        self.bounds=np.array(self.meta['bounds']);self.scales=np.array(self.meta['scales'])
    def predict(self,pars,dates,check_bounds=True):
        pars=np.atleast_2d(np.asarray(pars,float));dates=pd.DatetimeIndex(dates)
        years=(dates-pd.Timestamp(self.meta['origin'])).total_seconds().to_numpy()/86400/365.25
        if len(years)==0 or not np.isfinite(years).all() or min(years)<-1e-8 or max(years)>self.meta['time_max']+1e-8:raise ValueError('Outside proxy time domain.')
        if pars.shape[1]!=4 or not np.isfinite(pars).all():raise ValueError('Invalid parameters.')
        if check_bounds and ((pars<self.bounds[:,0]-1e-12).any() or (pars>self.bounds[:,1]+1e-12).any()):raise ValueError('Outside TRAIN parameter bounds.')
        result=[]
        for start in range(0,len(pars),256):
            a=pars[start:start+256]
            x=pd.DataFrame(np.column_stack([np.repeat(a,len(years),axis=0),np.tile(years,len(a))]),columns=FEATURES)
            q=np.column_stack([np.maximum(0,m.predict(x)) for m in self.models])
            result.append(q.reshape(len(a),len(years),3))
        return np.concatenate(result)

def loss(q,y,dates,scales,settings):
    dates=pd.DatetimeIndex(dates);recent=dates>=dates[-1]-pd.DateOffset(years=3)
    wt=np.where(recent,settings['recent_multiplier'],1.)
    e=(q-y[None,:,:])/scales[None,None,:]
    rates=np.sqrt(np.average(e**2,axis=1,weights=wt))@np.asarray(settings['fluid_weights'])
    if settings['slope_lambda']:
        t=(dates[recent]-dates[recent][0]).total_seconds().to_numpy()/86400/365.25;t=t-t.mean()
        if len(t)>=3 and np.sum(t*t)>0:
            slope=np.sum(e[:,recent,:]*t[None,:,None],axis=1)/np.sum(t*t)
            rates+=settings['slope_lambda']*np.mean(abs(slope[:,[0,2]]),axis=1)
    return rates

def initial_pool(sim,proxy,cfg):
    u=qmc.Sobol(4,scramble=True,seed=cfg['seed']).random_base2(cfg['pool_power'])
    lo,hi=proxy.bounds.T
    return np.vstack([lo+u*(hi-lo),sim[sim.Case_Number<=70].groupby('Case_Number')[PARAMS].first().to_numpy()])

def match(proxy,pool,base_predictions,history,query_dates,settings,cfg,seed):
    y=history[FLUIDS].to_numpy();pars=pool.copy();q=base_predictions.copy()
    scores=loss(q,y,history.Date,proxy.scales,settings);rng=np.random.default_rng(seed)
    lo,hi=proxy.bounds.T
    for sigma in [.05,.015,.004]:
        top=np.argsort(scores)[:8]
        centers=pars[rng.choice(top,cfg['refine_per_round'])]
        new=np.clip(centers+rng.normal(size=centers.shape)*sigma*(hi-lo),lo,hi)
        newq=proxy.predict(new,query_dates)
        pars=np.vstack([pars,new]);q=np.concatenate([q,newq]);scores=np.r_[scores,loss(newq,y,history.Date,proxy.scales,settings)]
    order=np.argsort(scores)
    return pars[order],q[order],scores[order]

def fitted_trends(hist,cfg):
    d=derive(hist);recent=d[d.Oil_Rate>0].tail(cfg['trend_window'])
    if len(recent)<5:raise ValueError('At least five positive-oil fitting samples required.')
    if not np.isfinite(recent.WOR).all():raise ValueError('Invalid WOR.')
    t=(recent.Date-recent.Date.iloc[-1]).dt.total_seconds().to_numpy()/86400/365.25
    x=recent.Np_MMSTB.to_numpy()-recent.Np_MMSTB.iloc[-1]
    oil_slope=float(np.polyfit(t,np.log(recent.Oil_Rate),1)[0])
    liquid_slope=float(np.polyfit(t,np.log(recent.Total_Liquid_Rate),1)[0])
    beta=float(np.polyfit(x,np.log1p(recent.WOR),1)[0])
    water_slope=float(np.polyfit(t,recent.Water_Rate,1)[0])
    gor=float(np.median(recent.Gas_Rate/recent.Oil_Rate))
    return {'oil_decline':float(np.clip(-oil_slope,0,2)), 'liquid_decline':float(np.clip(-liquid_slope,0,2)),
            'beta':float(np.clip(beta,0,20)), 'water_slope':water_slope,'gor':gor,
            'raw_beta':beta,'raw_liquid_decline':-liquid_slope}

def trend_model(sim,cutoff,cfg):
    rows=[]
    for case,g in sim[sim.Case_Number<=70].groupby('Case_Number'):
        g=g[g.Date<=cutoff];tr=fitted_trends(g,cfg)
        rows.append({'Case':case,**g[PARAMS].iloc[0].to_dict(),**tr})
    tab=pd.DataFrame(rows)
    model=make_pipeline(StandardScaler(),Ridge(alpha=1.0))
    model.fit(tab[PARAMS],tab[['beta','liquid_decline']])
    return model,tab

def forecast(hist,dates,method,cfg,pars=None,coefficient_model=None):
    hist=hist.sort_values('Date');dates=pd.DatetimeIndex(dates);origin=hist.Date.iloc[-1]
    days=(dates-origin).total_seconds().to_numpy()/86400
    if (days<0).any() or np.any(np.diff(days)<=0):raise ValueError('Forecast dates must be increasing and at/after origin.')
    last=hist[FLUIDS].iloc[-1].to_numpy();o0,g0,w0=last;l0=o0+w0
    if o0<=0:raise ValueError('Positive oil at forecast origin required.')
    tr=fitted_trends(hist,cfg);t=days/365.25
    beta=tr['beta'];dl=tr['liquid_decline']
    if method.startswith('hm_'):
        if pars is None or coefficient_model is None:raise ValueError('HM forecast requires parameters and a pre-cutoff coefficient model.')
        beta,dl=coefficient_model.predict(pd.DataFrame([pars],columns=PARAMS))[0]
        beta=float(np.clip(beta,0,20));dl=float(np.clip(dl,0,2))
    if method in ['exp_linear_water','arps_linear_water']:
        decline=tr['oil_decline'];oil=o0*np.exp(-decline*t)
        if method=='arps_linear_water':
            recent=hist[hist.Oil_Rate>0].tail(cfg['trend_window'])
            tx=(recent.Date-recent.Date.iloc[0]).dt.total_seconds().to_numpy()/86400/365.25
            yy=recent.Oil_Rate.to_numpy()/recent.Oil_Rate.iloc[0]
            def arps(tt,D,b):return np.exp(-np.log1p(b*D*tt)/b)
            fit=least_squares(lambda v:arps(tx,*v)-yy,x0=[max(.01,decline),.5],bounds=([0,.01],[2,1]))
            di,b=fit.x;dc=di/(1+b*di*tx[-1]);oil=o0*arps(t,dc,b)
        water=np.maximum(0,w0+tr['water_slope']*t)
    else:
        if method not in METHODS:raise ValueError('Unknown method.')
        if method.endswith('constant_liquid'):dl=0.
        z0=float(np.log1p(w0/o0))
        def rhs(day,n):
            liquid=l0*np.exp(-dl*day/365.25)
            return [liquid*np.exp(-z0-beta*n[0]/1e6)]
        if days[-1]==0:np_future=np.zeros(len(days))
        else:
            sol=solve_ivp(rhs,(0,float(days[-1])),[0.],t_eval=days,rtol=1e-8,atol=1e-3,max_step=31.)
            if not sol.success:raise ValueError(sol.message)
            np_future=sol.y[0]
        liquid=l0*np.exp(-dl*t);oil=liquid*np.exp(-z0-beta*np_future/1e6);water=liquid-oil
    gas=tr['gor']*oil
    q=np.column_stack([oil,gas,water]);q[days==0]=last
    if not np.isfinite(q).all() or (q<0).any():raise ValueError('Invalid forecast.')
    return q,{'beta_per_MMSTB':beta,'liquid_decline_per_year':dl,'fitted_GOR_MSCF_per_STB':tr['gor'],
              'form':method,'origin':str(origin.date()),'raw_beta':tr['raw_beta']}

def evaluate_cases(root,which):
    sim,field,cfg=data(root);proxy=Proxy(root)
    if which=='test':
        lock=read_json(Path(root)/'selection_lock.json')
        if lock['config_sha256']!=digest(Path(root)/'config.json'):raise ValueError('Config differs from locked selection.')
        if lock['code_sha256']!=digest(__file__):raise ValueError('Code differs from locked selection.')
        configs=[lock['objective']];methods=[lock['method']];caseids=range(86,101);folder='05_Test'
    else:configs=list(cfg['objectives']);methods=cfg['methods'];caseids=range(71,86);folder='04_Validation'
    out=stage_dir(root,folder);pool=initial_pool(sim,proxy,cfg);all_dates=sim[sim.Case_Number==1].Date
    poolq=proxy.predict(pool,all_dates);metrics=[];volumes=[];params_rows=[];pred_rows=[];support=[]
    for cutoff_s in cfg['cutoffs']:
        cutoff=pd.Timestamp(cutoff_s);coef,coef_table=trend_model(sim,cutoff,cfg)
        save_csv(out,f'coefficient_training_{cutoff_s}.csv',coef_table)
        fit_mask=(all_dates<=cutoff).to_numpy();forecast_dates=pd.date_range(cutoff,periods=cfg['horizon_quarters']+1,freq='QS')
        if forecast_dates[-1]>all_dates.max():raise ValueError('Backtest target beyond available simulation history.')
        for case in caseids:
            g=sim[sim.Case_Number==case];hist=g[g.Date<=cutoff].copy();truth=g.set_index('Date').loc[forecast_dates,FLUIDS].to_numpy()
            actual=g[PARAMS].iloc[0].to_numpy();inside=bool(((actual>=proxy.bounds[:,0])&(actual<=proxy.bounds[:,1])).all())
            support.append({'Case':case,'Cutoff':cutoff_s,'Actual_parameters_inside_TRAIN_box':inside,'Included_in_backtest':True})
            curves={}
            for objective in configs:
                pp,qq,ss=match(proxy,pool,poolq[:,fit_mask,:],hist,hist.Date,cfg['objectives'][objective],cfg,cfg['seed']+case)
                pars=pp[0];params_rows.append({'Case':case,'Cutoff':cutoff_s,'Objective':objective,'Match_score':ss[0],
                    **{p:pars[k] for k,p in enumerate(PARAMS)},**{'Actual_'+p:actual[k] for k,p in enumerate(PARAMS)}})
                for method in methods:
                    q,assume=forecast(hist,forecast_dates,method,cfg,pars,coef)
                    tags={'Case':case,'Cutoff':cutoff_s,'Objective':objective,'Method':method}
                    metrics+=metric_rows(truth[1:],q[1:],forecast_dates[1:],proxy.scales,**tags)
                    dd=(forecast_dates-forecast_dates[0]).total_seconds().to_numpy()/86400
                    for j,f in enumerate(FLUIDS):
                        tv=float(np.trapezoid(truth[:,j],dd));pv=float(np.trapezoid(q[:,j],dd))
                        volumes.append({**tags,'Fluid':f,'True_volume':tv,'Predicted_volume':pv,'Relative_volume_error':(pv-tv)/tv if tv else np.nan})
                    for k,date in enumerate(forecast_dates):
                        pred_rows.append({**tags,'Date':date,**{f:q[k,j] for j,f in enumerate(FLUIDS)},**{'Observed_'+f:truth[k,j] for j,f in enumerate(FLUIDS)}})
                    if objective==configs[-1]:curves[method]=(forecast_dates,q)
                if case==71 and cutoff_s==cfg['cutoffs'][0]:
                    plot_curves(out/f'history_matching_{objective}.png',hist.Date,hist[FLUIDS].to_numpy(),{'Matched proxy':(hist.Date,qq[0])},f'Pseudo-field Case {case}, {objective}')
            if cutoff_s==cfg['cutoffs'][-1]:
                plot_curves(out/f'backtest_Case_{case}.png',g.Date,g[FLUIDS].to_numpy(),curves,f'{which} Case {case}; forecast origin {cutoff_s}',cutoff)
            print(which,cutoff_s,'Case',case,'complete',flush=True)
    scores=pd.DataFrame(metrics);save_csv(out,'metrics.csv',scores);save_csv(out,'volume_errors.csv',volumes)
    save_csv(out,'parameter_recovery.csv',params_rows);save_csv(out,'predictions.csv',pred_rows);save_csv(out,'support_coverage.csv',support)
    combo=scores.groupby(['Case','Cutoff','Objective','Method'],as_index=False).NRMSE.mean()
    summary=combo.groupby(['Objective','Method']).NRMSE.agg(Mean='mean',Median='median',Worst='max',P90=lambda x:x.quantile(.9)).reset_index().sort_values(['Mean','Objective','Method'])
    save_csv(out,'method_summary.csv',summary)
    fig,ax=plt.subplots(figsize=(10,5));labels=summary.Objective+' / '+summary.Method
    ax.barh(labels,summary.Mean);ax.invert_yaxis();ax.set_xlabel('Mean case/origin score: equal mean of three fluid NRMSE');fig.tight_layout();fig.savefig(out/'method_comparison.png',dpi=140);plt.close(fig)
    if which=='validation':
        winner=summary.iloc[0]
        write_json(out/'recommended_selection.json',{'objective':winner.Objective,'method':winner.Method,'score':float(winner.Mean),
            'rule':'Minimum mean case/origin equal-fluid NRMSE on validation only; ties alphabetical.',
            'configuration_count':len(summary),'case_count':15,'cutoff_count':len(cfg['cutoffs']),
            'known_limit':'Same validation cases used for proxy selection; development comparison, not untouched evidence.'})
    else:
        testrows=[]
        for case,g in sim[sim.Case_Number.between(86,100)].groupby('Case_Number'):
            q=proxy.predict(g[PARAMS].iloc[0].to_numpy(),g.Date,check_bounds=False)[0]
            testrows+=metric_rows(g[FLUIDS].to_numpy(),q,g.Date.to_numpy(),proxy.scales,Case=case)
        save_csv(out,'proxy_test_metrics.csv',testrows)
    write_json(out/'status.json',{'complete':True,'case_count':15,'origins':cfg['cutoffs'],
        'test_previously_inspected':cfg['test_previously_inspected'],'versions':version_info()})
    print(summary.to_string(index=False),flush=True)

def lock_selection(root):
    root=Path(root);path=root/'selection_lock.json'
    if path.exists():raise FileExistsError('Selection already locked; do not tune using test outcomes.')
    rec=read_json(root/'04_Validation/recommended_selection.json')
    rec.update({'config_sha256':digest(root/'config.json'),'code_sha256':digest(__file__),
                'validation_metrics_sha256':digest(root/'04_Validation/metrics.csv')})
    write_json(path,rec);print('Locked:',rec['objective'],rec['method'],flush=True)

def diverse_indices(pars,q,scores,bounds,cfg,count=10):
    ok=np.flatnonzero(scores<=scores[0]*(1+cfg['candidate_relative_allowance'])+cfg['candidate_absolute_allowance'])
    if len(ok)<count:raise ValueError(f'Only {len(ok)} acceptable candidates; do not fabricate 10. Increase search budget in a NEW run.')
    z=(pars-bounds[:,0])/(bounds[:,1]-bounds[:,0]);chosen=[int(ok[0])]
    # Diversity here is parameter based; forecast spread is reported separately and never called probability.
    while len(chosen)<count:
        dist=np.min(np.linalg.norm(z[ok,None,:]-z[np.array(chosen)][None,:,:],axis=2),axis=1)
        dist[np.isin(ok,chosen)]=-1
        pick=int(ok[np.argmax(dist)])
        if np.max(dist)<1e-6:raise ValueError('Fewer than 10 distinct acceptable parameter vectors.')
        chosen.append(pick)
    return chosen

def field_work(root):
    root=Path(root);sim,field,cfg=data(root);proxy=Proxy(root);lock=read_json(root/'selection_lock.json')
    if lock['code_sha256']!=digest(__file__):raise ValueError('Code changed after lock.')
    out=stage_dir(root,'06_Field');pool=initial_pool(sim,proxy,cfg);settings=cfg['objectives'][lock['objective']]
    compare=[];candidate_rows=[];hist_rows=[];forecast_rows=[];milestones=[];backmetrics=[];assumptions=[];timing_curves={}
    # Shared observation support for both query hypotheses. Original field dates never change.
    valid=np.ones(len(field),bool)
    for shift in cfg['query_shifts']:
        ds=field.Date+pd.DateOffset(months=shift)
        valid&=(ds>=pd.Timestamp(proxy.meta['origin']))&(ds<=sim.Date.max())
    common=field[valid].copy();save_csv(out,'excluded_common_dates.csv',field[~valid])
    horizon=pd.date_range(field.Date.max(),periods=81,freq='QS');coef,_=trend_model(sim,field.Date.max(),cfg)
    for shift in cfg['query_shifts']:
        qdates=common.Date+pd.DateOffset(months=shift);base=proxy.predict(pool,qdates)
        pp,qq,ss=match(proxy,pool,base,common,qdates,settings,cfg,cfg['seed']+shift)
        idx=diverse_indices(pp,qq,ss,proxy.bounds,cfg);timing_curves[f'query +{shift} months']=(common.Date,qq[0])
        compare+=metric_rows(common[FLUIDS].to_numpy(),qq[0],common.Date.to_numpy(),proxy.scales,Query_shift_months=shift)
        for rank,k in enumerate(idx,1):
            cid=f'Q{shift:+d}_C{rank:02d}';pars=pp[k];q,assume=forecast(field,horizon,lock['method'],cfg,pars,coef)
            candidate_rows.append({'Candidate_ID':cid,'Query_shift_months':shift,'Rank_in_hypothesis':rank,'Objective_score':ss[k],
                **dict(zip(PARAMS,pars)),'Forecast_method':lock['method'],'Forecast_parameter_conditioned':lock['method'].startswith('hm_')})
            assumptions.append({'Candidate_ID':cid,**assume,'Date_rule_confirmed':cfg['date_rule_confirmed'],
                 'Future_operations_confirmed':cfg['operations_confirmed'],'Scenario_not_probability':True})
            for j,date in enumerate(common.Date):hist_rows.append({'Candidate_ID':cid,'Date':date,'Model_Query_Date':qdates.iloc[j],
                **dict(zip(FLUIDS,qq[k,j])),**{'Observed_'+f:common[f].iloc[j] for f in FLUIDS}})
            days=(horizon-horizon[0]).total_seconds().to_numpy()/86400
            cumulative=cumulative_trapezoid(q,days,axis=0,initial=0)
            for j,date in enumerate(horizon):
                forecast_rows.append({'Candidate_ID':cid,'Date':date,'Query_shift_months':shift,**dict(zip(FLUIDS,q[j])),
                    'Total_Liquid_Rate':q[j,0]+q[j,2],'Water_Cut':q[j,2]/(q[j,0]+q[j,2]),
                    'Forecast_Cumulative_Oil_MMSTB':cumulative[j,0]/1e6,'Forecast_Cumulative_Gas_Bscf':cumulative[j,1]/1e6,
                    'Forecast_Cumulative_Water_MMSTB':cumulative[j,2]/1e6})
            milestones.append(forecast_rows[-1])
        # Field development backtests refit parameters using only pre-cutoff observations for EACH timing hypothesis.
        for cs in cfg['cutoffs']:
            cutoff=pd.Timestamp(cs);fit=field[field.Date<=cutoff].copy()
            fq=fit.Date+pd.DateOffset(months=shift);supported=(fq>=sim.Date.min())&(fq<=sim.Date.max())
            fit=fit[supported].copy();fq=fq[supported]
            bp=proxy.predict(pool,fq);ps,_,_=match(proxy,pool,bp,fit,fq,settings,cfg,cfg['seed']+shift)
            model,_=trend_model(sim,cutoff,cfg)
            ds=pd.date_range(cutoff,periods=cfg['horizon_quarters']+1,freq='QS')
            hist=field[field.Date<=cutoff];pred,_=forecast(hist,ds,lock['method'],cfg,ps[0],model)
            truth=field.set_index('Date').loc[ds,FLUIDS].to_numpy()
            backmetrics+=metric_rows(truth[1:],pred[1:],ds[1:],proxy.scales,Query_shift_months=shift,Cutoff=cs,Method=lock['method'])
            plot_curves(out/f'field_backtest_Q{shift}_{cs}.png',field.Date,field[FLUIDS].to_numpy(),{'Locked forecast':(ds,pred)},f'FIELD DEVELOPMENT ONLY; query +{shift}m; {cs}',cutoff)
        print('Field query hypothesis',shift,'complete',flush=True)
    save_csv(out,'matching_metrics_common_dates.csv',compare);save_csv(out,'candidate_parameters.csv',candidate_rows)
    save_csv(out,'matched_history.csv',hist_rows);save_csv(out,'forecast_scenarios.csv',forecast_rows)
    save_csv(out,'endpoint_summary.csv',milestones);save_csv(out,'scenario_assumptions.csv',assumptions)
    save_csv(out,'field_development_backtest.csv',backmetrics)
    plot_curves(out/'time_hypotheses.png',field.Date,field[FLUIDS].to_numpy(),timing_curves,'Common-date field history matching; timing NOT confirmed')
    forecasts=pd.DataFrame(forecast_rows);curves={}
    for cid,g in forecasts.groupby('Candidate_ID'):curves[cid]=(pd.to_datetime(g.Date),g[FLUIDS].to_numpy())
    plot_curves(out/'twenty_year_scenarios.png',field.Date,field[FLUIDS].to_numpy(),curves,'Conditional 20-year scenarios, not calibrated probabilities',field.Date.max())
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for cid,g in forecasts.groupby('Candidate_ID'):
        axs[0].plot(g.Date,g.Forecast_Cumulative_Oil_MMSTB,lw=.8)
        axs[1].plot(g.Date,g.Water_Cut*100,lw=.8)
    axs[0].set_ylabel('Forecast-only cumulative oil, MMSTB');axs[1].set_ylabel('Water cut, %')
    for ax in axs:ax.grid(alpha=.2);ax.tick_params(axis='x',rotation=30)
    fig.suptitle('Conditional scenarios; overlapping curves do not imply low uncertainty');fig.tight_layout();fig.savefig(out/'cumulative_watercut.png',dpi=140);plt.close(fig)
    write_json(out/'status.json',{'complete':True,'number_of_scenarios':len(candidate_rows),'selected_method':lock['method'],
        'parameter_conditioned_forecast':lock['method'].startswith('hm_'),
        'structural_uncertainty_complete':False,'probabilities_calibrated':False,'twenty_year_validated':False,
        'ten_cases_exported':False,'case_selection_required':'Choose one date hypothesis after review; separate 10-case draft available for each.'})

def export_ten(root,shift):
    root=Path(root);out=stage_dir(root,f'07_Export_Q{shift}')
    p=pd.read_csv(root/'06_Field/candidate_parameters.csv');f=pd.read_csv(root/'06_Field/forecast_scenarios.csv',parse_dates=['Date'])
    selected=p[p.Query_shift_months==shift].sort_values('Rank_in_hypothesis').head(10)
    if len(selected)!=10:raise ValueError('Expected ten distinct case IDs in this hypothesis.')
    curves=f[f.Candidate_ID.isin(selected.Candidate_ID)].copy()
    if curves.groupby('Candidate_ID').size().ne(81).any():raise ValueError('Incomplete 20-year quarterly series.')
    if (curves[FLUIDS]<0).any().any() or not np.isfinite(curves[FLUIDS]).all().all():raise ValueError('Invalid rates.')
    if not curves.Water_Cut.between(0,1).all():raise ValueError('Invalid water cut.')
    if not np.allclose(curves.Total_Liquid_Rate,curves.Oil_Rate+curves.Water_Rate):raise ValueError('Liquid identity failed.')
    for _,g in curves.groupby('Candidate_ID'):
        if (np.diff(g.Forecast_Cumulative_Oil_MMSTB)<0).any():raise ValueError('Cumulative oil decreases.')
    save_csv(out,'selected_10_parameters.csv',selected);save_csv(out,'selected_10_forecast.csv',curves)
    h=pd.read_csv(root/'06_Field/matched_history.csv');h=h[h.Candidate_ID.isin(selected.Candidate_ID)]
    save_csv(out,'selected_10_history_match.csv',h)
    # Power BI long fact table: one row per candidate/date/fluid, segment explicit.
    facts=curves.melt(id_vars=['Candidate_ID','Date'],value_vars=FLUIDS,var_name='Fluid',value_name='Rate');facts['Segment']='Forecast'
    hm=h.melt(id_vars=['Candidate_ID','Date'],value_vars=FLUIDS,var_name='Fluid',value_name='Rate');hm['Segment']='History match'
    save_csv(out,'PowerBI_Rates.csv',pd.concat([hm,facts],ignore_index=True))
    _,obs,_=data(root);save_csv(out,'PowerBI_Observed.csv',obs[['Date']+FLUIDS].melt(id_vars='Date',var_name='Fluid',value_name='Rate'))
    unique=np.unique(np.round(np.stack([g[FLUIDS].to_numpy() for _,g in curves.groupby('Candidate_ID')]),6),axis=0).shape[0]
    write_json(out/'export_status.json',{'case_count':10,'rows':len(curves),'distinct_forecast_trajectories_at_6dp':unique,
        'query_shift_months':shift,'date_rule_confirmed':False,'schema':'Team schema, not an organizer-verified upload template',
        'official_submission_ready':False,'warning':'Identical forecasts under a direct trend method remain identical despite different HM parameters.' if unique<10 else 'Scenario spread is not probability.',
        'missing_deliverables':['Final documentation PPTX','Power BI PBIX or required service link','Video','Repository submission URL'],
        'required_reviews':['Date convention','Operating assumptions','Final 10 selection','Official upload schema','Raw Phase 1 audit']})
    print('Exported ten draft cases. Read export_status.json; not automatically submission-ready.',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['setup','audit','train','validate','lock','test','field','export']);ap.add_argument('--run',required=True);ap.add_argument('--source');ap.add_argument('--shift',type=int,default=0);a=ap.parse_args()
    if a.stage=='setup':setup(a.source,a.run)
    else:
        funcs={'audit':audit,'train':train_models,'validate':lambda r:evaluate_cases(r,'validation'),'lock':lock_selection,
               'test':lambda r:evaluate_cases(r,'test'),'field':field_work,'export':lambda r:export_ten(r,a.shift)}
        funcs[a.stage](a.run)
if __name__=='__main__':main()
