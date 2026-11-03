"""fig3_and_supp.py — Figure 3 and supplementary Figures S2, S6 and S8.
Self-contained: the type and colour conventions live in style_main.py.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]   # repository root
BASE = ROOT
OLD = ROOT/'results'
RES, FIG = ROOT/'results', ROOT/'figures'
QA = RES/'figure_qa'
QA.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
SOURCES = ['CHARLS','ELSA','HRS']
LAB = {'CHARLS':'China (CHARLS)', 'ELSA':'England (ELSA)', 'HRS':'USA (HRS)', 'POOLED':'Pooled'}
# Cohort colours must match Figures 1-3 house style (main12_style.COH).
COL = {'CHARLS':'#3C6E9E','ELSA':'#C0803F','HRS':'#4E8F7C'}
MARK = {'CHARLS':'o','ELSA':'s','HRS':'^'}
reports = []


def apply_figure_style():
    # Fonts are pinned explicitly: importing another style module in the same
    # session must not silently change the typeface of these figures.
    plt.rcParams.update({'font.family':'sans-serif',
        'font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
        'font.size':8,'axes.titlesize':8,
        'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
        'figure.dpi':120,'savefig.dpi':300,'pdf.fonttype':42,'ps.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'xtick.direction':'out',
        'ytick.direction':'out','axes.linewidth':.65,'lines.linewidth':1.1,
        'legend.frameon':False,'axes.unicode_minus':True})


def letter(ax, text):
    ax.text(-.07, 1.17, text, transform=ax.transAxes, fontsize=10, weight='bold', va='bottom')


def save(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [(t,t.get_window_extent(renderer)) for t in fig.findobj(Text)
             if t.get_visible() and t.get_text().strip()]
    # Figure bounds and true text overlap; axis-title/tick siblings are included.
    outside = [t.get_text() for t,b in texts if not fig.bbox.contains(b.x0,b.y0) or not fig.bbox.contains(b.x1,b.y1)]
    overlaps = [(a.get_text(),b.get_text()) for i,(a,ba) in enumerate(texts) for b,bb in texts[i+1:]
                if ba.overlaps(bb)]
    own_ticks = {ax:set(ax.get_xticklabels(which='both')+ax.get_yticklabels(which='both')) for ax in fig.axes}
    spines = [(sp,sp.get_window_extent(renderer)) for ax in fig.axes for sp in ax.spines.values() if sp.get_visible()]
    spine_hits = [(t.get_text(),sp.spine_type) for t,bt in texts for sp,bs in spines
                  if bt.overlaps(bs) and t not in own_ticks[sp.axes]]
    reports.append({'figure':name,'outside_text':outside,'text_overlaps':overlaps,'spine_overlaps':spine_hits})
    if outside or overlaps or spine_hits:
        raise RuntimeError(reports[-1])
    fig.savefig(FIG/f'{name}.png', dpi=300, facecolor='white')
    fig.savefig(FIG/f'{name}.pdf', facecolor='white')
    # QA crops are explicitly non-deliverable derived inspection files in results.
    im = Image.open(FIG/f'{name}.png')
    for i,ax in enumerate(fig.axes):
        b = ax.get_position()
        pad = .055
        box = (max(0,int((b.x0-pad)*im.width)),max(0,int((1-b.y1-pad)*im.height)),
               min(im.width,int((b.x1+pad)*im.width)),min(im.height,int((1-b.y0+pad)*im.height)))
        im.crop(box).save(QA/f'{name}_panel{i+1}.png')
    plt.close(fig)


def figure3():
    d = pd.read_csv(RES/'transport_deciles_q.csv')
    u = pd.read_csv(RES/'local_updates.csv').query("strategy == 'Q'")
    p = (pd.read_csv(RES/'local_update_paired.csv')
         .groupby(['test', 'update'], as_index=False)['d_brier'].mean()
         .rename(columns={'d_brier': 'mean'})
         .assign(metric='d_brier'))
    # 170 mm is the journal's full text width; keep all three main figures inside it.
    fig = plt.figure(figsize=(170/25.4, 150/25.4))
    gs = fig.add_gridspec(2,6, left=.105,right=.975,bottom=.18,top=.89,hspace=.73,wspace=.80)
    for i,src in enumerate(SOURCES):
        ax = fig.add_subplot(gs[0, i*2:i*2+2])
        for method,mk,ls,fill in [('none','o','-',COL[src]),('intercept','s','--','white')]:
            r = d[(d.test==src)&(d['update']==method)].sort_values('decile')
            ax.plot(r.pred,r.obs,marker=mk,linestyle=ls,color=COL[src],mfc=fill,ms=3.7,mew=1)
        ax.plot([0,.5],[0,.5],':',color='.6',lw=.8,zorder=0)
        ax.set(xlim=(-.015,.515),ylim=(-.015,.515),xticks=[0,.2,.4],yticks=[0,.2,.4],xlabel='Predicted ADL risk')
        if i==0: ax.set_ylabel('Observed ADL risk')
        ax.set_title(LAB[src],loc='left',pad=6)
        letter(ax,'a'+str(i+1))
    methods = ['none','intercept','logistic_recal','coefficient_revision']
    names = ['No\nupdate','Intercept','Logistic\nrecal.','Coefficient\nrefit']
    axb, axc = fig.add_subplot(gs[1,:3]), fig.add_subplot(gs[1,3:])
    for i,src in enumerate(SOURCES):
        r = (u[u.test==src].groupby('update', as_index=False)['auc'].mean()
             .set_index('update').reindex(methods))
        pr = p[p.test==src].set_index('update')
        xx = np.arange(4)+(i-1)*.09
        diff = [0]+[pr.loc[m,'mean'] for m in methods[1:]]
        axb.plot(xx,diff,marker=MARK[src],color=COL[src],ms=4,label=LAB[src],lw=.9)
        axc.plot(xx,r.auc,marker=MARK[src],color=COL[src],ms=4,lw=.9)
    axb.axhline(0,color='.6',ls=':',lw=.8)
    axb.set_ylabel('Paired Δ three-state Brier')
    axc.set_ylabel('Absolute held-out AUC')
    axb.set_title('Lower = better',loc='left',pad=8)
    axc.set_title('Higher = better',loc='left',pad=8)
    for ax in [axb,axc]:
        ax.set_xticks(range(4),names)
        ax.margins(x=.1,y=.18)
    letter(axb,'b'); letter(axc,'c')
    handles=[Line2D([0],[0],marker='o',color='.25',label='No update',ms=4),
             Line2D([0],[0],marker='s',mfc='white',color='.25',ls='--',label='Intercept update',ms=4)]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,.995),ncol=2)
    fig.legend(handles=axb.get_legend_handles_labels()[0],loc='lower center',bbox_to_anchor=(.54,.015),ncol=3)
    fig.text(.54,.077,'Q model; 20 paired local holdouts per cohort',ha='center',fontsize=7)
    save(fig,'fig3_calibration_update')
    d.assign(panel='a').to_csv(FIG/'fig3_calibration_update_values.csv',index=False)
    u.to_csv(FIG/'fig3_update_absolute_values.csv',index=False)
    p.to_csv(FIG/'fig3_update_paired_values.csv',index=False)


def s2():
    d = pd.read_csv(OLD/'subgroup.csv')
    d = d[d.src.isin(SOURCES)&d.strategy.isin(['R','Q','E3'])].copy()
    groups = [('age','60-69','60–69 years'),('age','70-79','70–79 years'),('age','80+','80+ years'),
              ('male','female','Female'),('male','male','Male'),('educ','low','Education: low'),
              ('educ','mid','Education: mid'),('educ','high','Education: high')]
    fig,axs = plt.subplots(1,3,figsize=(180/25.4,115/25.4),sharex=True,sharey=True,
                          gridspec_kw={'left':.20,'right':.98,'top':.82,'bottom':.19,'wspace':.30})
    # Strategies, not cohorts: use the house style's lightness ladder so these
    # hues cannot be confused with the cohort palette used in Figures 1-3.
    color={'R':'#7A7A7A','Q':'#4F5D75','E3':'#1A1A1A'}
    marker={'R':'o','Q':'s','E3':'^'}
    missing=[]
    for i,(ax,src) in enumerate(zip(axs,SOURCES)):
        for j,(var,group,label) in enumerate(groups):
            for k,st in enumerate(['R','Q','E3']):
                r=d[(d.src==src)&(d.strategy==st)&(d['var']==var)&(d.group==group)]
                yy=j+(k-1)*.19
                if len(r): ax.plot(r.auc.iloc[0],yy,marker[st],color=color[st],ms=4)
                else:
                    missing.append(dict(src=src,strategy=st,var=var,group=group,reason='not reported by source n/event threshold'))
                    ax.plot([1.025], [yy], transform=ax.get_yaxis_transform(),
                            marker='x', ls='', color=color[st], ms=4.5, mew=0.9,
                            clip_on=False, zorder=5)
        ax.set_title(LAB[src],loc='left',pad=12)
        ax.set_xlim(.49,.86); ax.set_xticks([.5,.6,.7,.8]); ax.set_xlabel('Apparent ADL AUC')
        ax.set_yticks(range(8),[g[2] for g in groups]); ax.set_ylim(7.55,-.55)
        ax.grid(axis='x',color='.9',lw=.5)
        letter(ax,chr(97+i))
    fig.legend(handles=[Line2D([0],[0],marker=marker[s],color=color[s],ls='',label=s,ms=4) for s in ['R','Q','E3']],
               loc='upper center',bbox_to_anchor=(.59,.995),ncol=3)
    fig.text(.59,.06,'Descriptive; model-specific complete cases. x (right margin) = subgroup result not reported by the source analysis',ha='center',fontsize=7)
    save(fig,'figS2_subgroups')
    d.to_csv(FIG/'figS2_subgroups_values.csv',index=False)
    pd.DataFrame(missing).to_csv(RES/'figS2_not_reported.csv',index=False)


def s6():
    d=pd.read_csv(OLD/'selection_diagnostic.csv')
    d=d[(d.strategy=='Q')&d['subset'].isin(['grip not measured','grip measured'])].copy()
    fig,ax=plt.subplots(figsize=(150/25.4,90/25.4))
    fig.subplots_adjust(left=.15,right=.97,top=.80,bottom=.26)
    for i,src in enumerate(SOURCES):
        r=d[d.src==src].set_index('subset').loc[['grip not measured','grip measured']]
        ax.plot([i-.15,i+.15],r.auc,color=COL[src],lw=.9)
        for x,(_,row),mk,fill in zip([i-.15,i+.15],r.iterrows(),['o','s'],['white',COL[src]]):
            ax.plot(x,row.auc,marker=mk,color=COL[src],mfc=fill,ms=5)
            ax.annotate(f'{row.auc:.3f}\nn={int(row.n)}',(x,row.auc),xytext=(14 if src=='ELSA' and row.name=='grip measured' else 0,8),textcoords='offset points',ha='center',fontsize=7, bbox={'facecolor':'white','edgecolor':'none','pad':.8})
    ax.set_xticks(range(3),[LAB[s] for s in SOURCES]); ax.set_ylabel('Q-model apparent ADL AUC')
    ax.set_ylim(.67,.85); ax.set_xlim(-.5,2.5)
    ax.set_title('Descriptive measurement-availability comparison',loc='left',pad=15)
    fig.legend(handles=[Line2D([0],[0],marker='o',mfc='white',color='.3',ls='',label='Grip not measured'),
                        Line2D([0],[0],marker='s',color='.3',ls='',label='Grip measured')],loc='lower center',bbox_to_anchor=(.55,.035),ncol=2)
    save(fig,'figS6_selection')
    d.to_csv(FIG/'figS6_selection_values.csv',index=False)


def s8():
    cv=pd.read_csv(OLD/'cv_performance.csv')
    pooled=json.loads((OLD/'cv_pooled.json').read_text())
    cv=pd.concat([cv,pd.DataFrame([dict(src='POOLED',strategy=st,n=pooled['n'],events=pooled['events'],auc_cv=pooled['auc_'+st]) for st in ['R','Q']])])
    mp=pd.read_csv(OLD/'main_performance.csv').query("set == 'R_vs_Q'")
    d=mp.merge(cv,left_on=['block','strategy'],right_on=['src','strategy'])
    fig,axs=plt.subplots(1,2,figsize=(170/25.4,95/25.4),sharey=True,gridspec_kw={'left':.21,'right':.98,'top':.81,'bottom':.25,'wspace':.18})
    for ax,st in zip(axs,['R','Q']):
        for i,src in enumerate(SOURCES+['POOLED']):
            row=d[(d.src==src)&(d.strategy==st)].iloc[0]
            ax.plot([row.auc,row.auc_cv],[i,i],color='.65',lw=1)
            ax.plot(row.auc,i,'o',color='#7A7A7A',mfc='white',ms=5)
            ax.plot(row.auc_cv,i,'s',color='#4F5D75',ms=4)
        ax.set_title('Background (R)' if st=='R' else '+ Questionnaire (Q)',loc='left',pad=14)
        ax.set_xlabel('ADL AUC; higher = better'); ax.set_xlim(.61,.79); ax.set_xticks([.62,.67,.72,.77])
        ax.set_yticks(range(4),[LAB[s] for s in SOURCES+['POOLED']]); ax.set_ylim(3.5,-.5)
        letter(ax,'a' if st=='R' else 'b')
    fig.legend(handles=[Line2D([0],[0],marker='o',mfc='white',color='#7A7A7A',ls='',label='Apparent'),
                        Line2D([0],[0],marker='s',color='#4F5D75',ls='',label='5-fold out-of-fold')],loc='lower center',bbox_to_anchor=(.59,.025),ncol=2)
    save(fig,'figS8_cv')
    d.to_csv(FIG/'figS8_cv_values.csv',index=False)


if __name__=='__main__':
    apply_figure_style()
    figure3(); s2(); s6(); s8()
    (RES/'figure_collision_checks.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    print('Rendered and collision-checked Figure3/S2/S6/S8; QA panel crops in results/figure_qa',flush=True)
