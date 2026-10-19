"""Render six manuscript figure groups from existing, unmodified analysis caches.

Run in the project's modularity_env: python example/manuscript_figures.py --output DIR
No model training or numerical recomputation is performed.
"""
from pathlib import Path
import argparse, csv, json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'example'), str(ROOT)]
import figure2 as f2, figure4 as f4, figure5 as f5, figure6 as f6
import figure8 as f8, figure10 as f10, modularity_time as mt
RESULTS = ROOT/'example/results'
LAYERS=['#56B4E9','#D55E00','#0072B2']
RULES=['#0072B2','#D55E00','#E69F00','#CC79A7','#009E73']
for module in [f2,f4,f5,f6,f8,mt]:
    if hasattr(module,'LAYER_COLORS'): module.LAYER_COLORS=LAYERS
    if hasattr(module,'RULE_COLORS'): module.RULE_COLORS=RULES

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.labelsize':7,
      'axes.titlesize':7,'xtick.labelsize':6,'ytick.labelsize':6,'legend.fontsize':6,
      'axes.linewidth':.5,'lines.linewidth':.7,'pdf.fonttype':42,'ps.fonttype':42,
      'svg.fonttype':'none','mathtext.fontset':'custom','mathtext.rm':'DejaVu Sans',
      'mathtext.it':'DejaVu Sans:italic','mathtext.bf':'DejaVu Sans:bold',
      'mathtext.cal':'DejaVu Sans','mathtext.sf':'DejaVu Sans','mathtext.tt':'DejaVu Sans','savefig.dpi':450})
style()

def read(path):
    with open(path,newline='') as h: return list(csv.DictReader(h))

def capture(function,*args,**kwargs):
    """Use existing scientific plotters without writing their intermediate figures."""
    figs=[]; old_save=plt.Figure.savefig; old_close=plt.close
    def save(fig,*a,**kw):
        if fig not in figs:figs.append(fig)
    plt.Figure.savefig=save; plt.close=lambda *a,**kw:None
    try:function(*args,**kwargs)
    finally:plt.Figure.savefig=old_save; plt.close=old_close
    style()
    return figs[-1]

def clean(fig):
    for ax in fig.axes:
        for loc in ['left','center','right']: ax.set_title('',loc=loc)
        for t in list(ax.texts):
            if t.get_text() in list('abcdefgh'):t.remove()
        for get,setter in [(ax.get_xlabel,ax.set_xlabel),(ax.get_ylabel,ax.set_ylabel)]:
            s=get()
            if s and s[0].isalpha():s=s[0].upper()+s[1:]
            setter(s,fontsize=7,labelpad=2)
        ax.tick_params(axis='both',which='both',labelsize=6,width=.5,length=2,pad=2)
        ax.grid(False)
        for spine in ax.spines.values():spine.set_linewidth(.5)
    for text in fig.findobj(Text):
        text.set_fontfamily('DejaVu Sans')
        if text.get_fontsize()>7:text.set_fontsize(7)
    for leg in fig.legends:
        for t in leg.get_texts():t.set_fontsize(6)

def panel(fig,ax,letter,x=-.055,y=1.055):
    pos=ax.get_position()
    fig.text(pos.x0+x*pos.width,pos.y0+y*pos.height,letter,fontfamily='DejaVu Sans',
             fontweight='bold',fontsize=8,ha='left',va='bottom',zorder=100)

def condition(ax,text):
    ax.text(.98,.96,text,ha='right',va='top',transform=ax.transAxes,fontsize=7)

def legend(fig,labels,colors,y,ncol=None,x=.54):
    return fig.legend([Patch(facecolor=c,alpha=.8) for c in colors],labels,
       loc='upper center',bbox_to_anchor=(x,y),ncol=ncol or len(labels),
       frameon=False,fontsize=6,handlelength=1,handleheight=.6,columnspacing=1)

def export(fig,stem,out):
    fig.canvas.draw()
    for ext in ['pdf','svg','png']:fig.savefig(out/(stem+'.'+ext),dpi=450)
    # A lower-resolution local review copy; does not replace publication artwork.
    fig.savefig(out/(stem+'_review.png'),dpi=180)
    print('Rendered',stem,'axes',len(fig.axes),'mm',np.round(fig.get_size_inches()*25.4,1).tolist(),flush=True)
    plt.close(fig)

def box(ax,x,y,w,h,text,fc='#E8F1F5'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.009',
                 linewidth=.6,edgecolor='#465766',facecolor=fc))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=7)

def arrow(ax,a,b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=7,lw=.7,color='#465766'))

def figure1(out):
    paths=f2.analysis_mat_paths(f2.DEFAULT_ANALYSIS_DIR)
    counts,acc=f2.load_figure2_arrays(paths)
    labels=f2.combo_labels(5,counts.shape[2])
    fig=capture(f2.plot_combined_fig2,counts,acc,labels,out,['pdf'])
    fig.set_size_inches(180/25.4,170/25.4)
    axes=list(fig.axes)
    for i,ax in enumerate(axes):
        ax.set_position([.09,[.535,.31,.085][i],.89,.135])
        if ax.get_legend():ax.get_legend().remove()
    clean(fig)
    for ax,l in zip(axes,'bcd'):panel(fig,ax,l)
    axes[0].set_ylabel('Log(1 + weight-token count)',fontsize=7)
    axes[0].set_xlabel('')
    axes[1].set_xlabel('Retained rule set')
    axes[2].set_xlabel('Retained rule set')
    legend(fig,['Layer 1','Layer 2','Layer 3'],LAYERS,.725)
    legend(fig,[f'Rule {i}' for i in range(1,6)],RULES,.487)
    ax=fig.add_axes([.055,.80,.93,.15]);ax.set_axis_off()
    box(ax,.025,.25,.17,.42,'Shared input\n$(a,b)$')
    box(ax,.255,.25,.22,.42,'Mother network\nJoint five-rule loss')
    box(ax,.555,.25,.19,.42,'Rule-wise\nimportance masks')
    box(ax,.825,.25,.15,.42,'Functional\ntokens')
    for a,b in [((.195,.46),(.255,.46)),((.475,.46),(.555,.46)),((.745,.46),(.825,.46))]:arrow(ax,a,b)
    ax.text(.365,.05,'MLP: three 300-unit hidden layers; or TinyTransformer',ha='center',fontsize=6)
    panel(fig,ax,'a',x=0,y=1.0)
    export(fig,'main_fig1',out)

def figure2(out):
    retained,runs=f4.build_task_retained_data(f4.TASK_DIRS)
    taylor=f4.load_mat_stack(f4.TAYLOR_MLP_DIR,'weight_combo_retained_accuracy')
    trans=f4.load_mat_stack(f4.TRANSFORMER_DIR,'weight_combo_retained_accuracy')
    # The original sensitivity-profile JSON is absent. Show the actually available
    # mean retained accuracy, explicitly labelled as such, with no proxy substitution.
    matrix=np.nanmean(taylor[:,:,:5],axis=0).T
    fig=plt.figure(figsize=(180/25.4,165/25.4))
    positions=[[x,y,.365,.205] for y in [.735,.415,.095] for x in [.10,.615]]
    axes=[fig.add_axes(p) for p in positions]
    data=list(retained.values())
    for i in range(3):f4.plot_retained_demo_axes(axes[i],data[i],'',compact=True,tick_size=6,rotate=50)
    f4.plot_taylor_heatmap(axes[3],matrix,'',with_colorbar=False)
    axes[3].set_aspect('auto')
    cax=fig.add_axes([.985,.415,.007,.205]);fig.colorbar(axes[3].images[0],cax=cax)
    cax.set_ylabel('Mean relative accuracy',fontsize=7)
    f4.plot_retained_demo_axes(axes[4],taylor,'',compact=True,tick_size=6,rotate=50)
    f4.plot_retained_demo_axes(axes[5],trans,'',compact=True,tick_size=6,rotate=50)
    clean(fig)
    cax.yaxis.set_ticks_position('right');cax.yaxis.set_label_position('right')
    # Heatmap colour scale shares the panel boundary; leave it outside the data grid.
    axes[3].set_position([.615,.415,.28,.205]);cax.set_position([.913,.415,.01,.205])
    for ax in axes:
        if ax.get_legend():ax.get_legend().remove()
    for t in axes[3].texts:t.set_color('#222222')
    for ax,l in zip(axes,'abcdef'):panel(fig,ax,l,x=-.17)
    for ax,label in zip([axes[0],axes[1],axes[2],axes[4],axes[5]],
       ['Binary arithmetic','Pattern logic','MNIST arithmetic','Taylor / MLP','Taylor / Transformer']):
        ax.text(.01,1.10,label,transform=ax.transAxes,fontsize=6.5)
    legend(fig,[f'Rule {i}' for i in range(1,6)],RULES,1.0)
    export(fig,'main_fig2',out)

def figure3(out):
    single=f5.read_csv(RESULTS/'figure5/figure5_weight_single_rule_metrics_score_quantile_frac0p75.csv')
    threshold=f5.read_csv(RESULTS/'figure5/figure5_weight_threshold_metrics_score_quantile_frac0p75.csv')
    fig=capture(f5.plot_combined_fig5,single,threshold,[0,1,5],[0,1,5,10],[.5,1,2],5,out,['pdf'],'weight','ablation','fixed')
    fig.set_size_inches(180/25.4,170/25.4)
    axes=fig.axes
    pos=[]
    for y in [.785,.425]:
        pos += [[x,y,.245,.125] for x in [.10,.4175,.735]]
        pos += [[.10,y-.135,.88,.060]]
    pos += [[.10,.065,.365,.125],[.615,.065,.365,.125],[.1,.24,.88,.02]]
    for ax,p in zip(axes,pos):ax.set_position(p)
    for leg in list(fig.legends):leg.remove()
    clean(fig)
    for i,lab in [(0,'a'),(4,'b'),(8,'c'),(9,'d')]:panel(fig,axes[i],lab,x=-.17)
    for ax,l in zip(axes[:3],[0,1,5]):condition(ax,f'$l={l}$')
    for ax,g in zip(axes[4:7],[.5,1,2]):condition(ax,f'$\gamma={g:g}$')
    axes[3].set_ylabel('Retained-weight\nfraction')
    axes[7].set_ylabel('Retained-weight\nfraction')
    for ax in [axes[7],axes[9]]:ax.set_xlabel(r'Regularizer exponent $\gamma$')
    legend(fig,[f'Rule {i}' for i in range(1,6)],RULES,.985)
    legend(fig,['Layer 1','Layer 2','Layer 3'],LAYERS,.240)
    export(fig,'main_fig3',out)

def figure4(out):
    summary=f6.load_summary(RESULTS/'ruleweight/ruleweight_summary.json')
    matrices=f6.figure_matrices(summary)
    fig=plt.figure(figsize=(180/25.4,138/25.4))
    axes=[fig.add_axes([x,.60,.205,.275]) for x in [.09,.42,.75]]
    for i,(ax,mat) in enumerate(zip(axes,matrices)):
        f6.draw_heatmap(fig,ax,mat*(1e4 if i==1 else 1),'','',
            ['Accuracy change',r'Weight-fraction change ($\times10^{-4}$)','Specialization margin'][i],
            '{:+.2f}' if i<2 else '{:.2f}','RdBu_r' if i<2 else 'viridis',i<2)
        cb=fig.axes[-1]
        ax.set_position([[.09,.42,.75][i],.65,.205,.265]);ax.set_aspect('auto')
        cb.set_position([[.09,.42,.75][i],.55,.205,.013])
        # Replace vertical colourbar with a horizontal one in a reserved row.
        cb.remove();cax=fig.add_axes([[.09,.42,.75][i],.55,.205,.013])
        fig.colorbar(ax.images[0],cax=cax,orientation='horizontal',label=
            ['Accuracy change',r'Weight-fraction change ($\times10^{-4}$)','Specialization margin'][i])
    a=fig.add_axes([.09,.12,.365,.255]);b=fig.add_axes([.61,.12,.365,.255]);axes +=[a,b]
    # Match panel e's axis direction: retained weights on x, accuracy on y.
    xv,yv,xe,ye=f6.matched_target_statistics(summary)
    for i in range(5):a.errorbar(yv[i]*1e4,xv[i],xerr=ye[i]*1e4,yerr=xe[i],fmt='o',
        color=RULES[i],markersize=4,capsize=2,elinewidth=.7)
    fit=np.polyfit(yv*1e4,xv,1);xx=np.linspace(min(yv)*1e4,max(yv)*1e4,50)
    a.plot(xx,np.polyval(fit,xx),'--',color='#555555',lw=.7)
    a.axhline(0,color='#999999',lw=.5);a.axvline(0,color='#999999',lw=.5)
    a.text(.03,.95,f'Pearson $r$ = {np.corrcoef(xv,yv)[0,1]:.2f}',transform=a.transAxes,va='top',fontsize=7)
    rows=read(RESULTS/'ruleweight/figures/figure6_paired_trials.csv')
    for row in rows:
        for k in ['accuracy_delta','weight_ratio_delta']:row[k]=float(row[k])
        for k in ['rule','repeat']:
            if k in row:row[k]=int(row[k])
    f6.draw_trial_scatter(b,rows)
    clean(fig)
    for ax,l in zip(axes,'abcde'):panel(fig,ax,l,x=-.20 if ax in axes[:3] else -.14)
    for ax in axes[:3]:ax.set_xlabel('Evaluated rule');ax.set_ylabel('Upweighted rule')
    for ax in [a,b]:ax.set_xlabel(r'Retained-weight change ($\times10^{-4}$)');ax.set_ylabel('Accuracy change')
    for ax in [a,b]:
        if ax.get_legend():ax.get_legend().remove()
    legend(fig,[f'Rule {i}' for i in range(1,6)],RULES,.455)
    export(fig,'main_fig4',out)

def figure5(out):
    rows=mt.read_csv(RESULTS/'modularity_time/modularity_time_metrics.csv')
    metrics=f8.read_csv(RESULTS/'figure8/figure8_representation_metrics_relu.csv')
    hist=f8.read_csv(RESULTS/'figure8/figure8_correlation_hist.csv')
    fig=capture(f8.plot_fig8,metrics,hist,[0,1,2,3,5,10,15],[0,1,5,10],out,['pdf'])
    fig.set_size_inches(180/25.4,170/25.4)
    ga=list(fig.axes)
    for i,ax in enumerate(ga):
        if i<4:ax.set_position([[.085,.325,.565,.805][i],.44,.177,.165])
        else:ax.set_position([[.085,.4175,.75][i-4],.10,.23,.185])
    dyn=[fig.add_axes([x,.76,.177,.155]) for x in [.085,.325,.565,.805]]
    for ax,l in zip(dyn,[0,1,5,10]):
        data=[];positions=[];ids=[]
        for j,ep in enumerate([1000,2000,3000,4000,5000]):
            for k,layer in enumerate([1,2,3]):
                v=np.array([r['modularity'] for r in rows if r['lasso']==l and r['epoch']==ep and r['layer_index']==layer])
                if len(v):data.append(mt._violin_values(v));positions.append(j*1.55+[-.26,0,.26][k]);ids.append(k)
        vp=ax.violinplot(data,positions=positions,widths=.20,showmeans=True,showmedians=True,showextrema=False)
        for body,k in zip(vp['bodies'],ids):body.set_facecolor(LAYERS[k]);body.set_edgecolor('none');body.set_alpha(.8)
        vp['cmeans'].set_color('black');vp['cmedians'].set_color('#C62828')
        vp['cmeans'].set_linewidth(.65);vp['cmedians'].set_linewidth(.65)
        ax.set_xticks(np.arange(5)*1.55,[1,2,3,4,5]);ax.set_xlabel('Epoch (×10³)');ax.set_ylabel('$M$');ax.set_ylim(-.03,1.03)
        condition(ax,f'$l={l}$')
    clean(fig)
    for ax,l in zip(dyn,'abcd'):panel(fig,ax,l,x=-.25)
    for ax,l in zip(ga[4:],'fgh'):panel(fig,ax,l,x=-.20)
    panel(fig,ga[0],'e',x=-.25)
    for ax,l in zip(ga[:4],[0,1,5,10]):condition(ax,f'$l={l}$');ax.set_xlabel('Absolute correlation')
    for ax in ga[:4]:
        if ax.get_legend():ax.get_legend().remove()
    for ax in ga[1:4]:ax.set_ylabel('')
    for ax in ga[4:]:
        if ax.get_legend():ax.get_legend().remove()
    legend(fig,['Layer 1','Layer 2','Layer 3','Output'],LAYERS+[f8.OUTPUT_COLOR],.35,ncol=2,x=.20)
    legend(fig,['1–2','2–3','1–3'],f8.PAIR_COLORS,.35,ncol=3,x=.5325)
    legend(fig,['1–output','2–output','3–output'],f8.OUTPUT_PAIR_COLORS,.35,ncol=2,x=.865)
    ga[5].set_ylabel('Linear CKA');ga[6].set_ylabel('Linear CKA')
    legend(fig,['Layer 1','Layer 2','Layer 3'],LAYERS,.995)
    export(fig,'main_fig5',out)

def figure6(out):
    rows=read(RESULTS/'figure10/fig10_recomposition_metrics.csv')
    fig=plt.figure(figsize=(180/25.4,140/25.4))
    ax=fig.add_axes([.045,.62,.93,.30]);ax.set_axis_off()
    # Three alternative computational graphs; all input symbols match the manuscript.
    for x,label in [(0,'Control'),(.34,'Type 1'),(.68,'Type 2')]:
        ax.text(x+.145,.98,label,ha='center',fontsize=7)
        ax.text(x+.145,.80,'$(a,b)$',ha='center',fontsize=7)
    box(ax,.045,.36,.20,.23,'Rule 5 token');arrow(ax,(.145,.75),(.145,.60));arrow(ax,(.145,.35),(.145,.14))
    ax.text(.145,-.06,'$a(a+b)$',ha='center',fontsize=7)
    box(ax,.36,.43,.10,.19,'Add');box(ax,.505,.15,.12,.19,'Multiply')
    arrow(ax,(.485,.75),(.41,.63));arrow(ax,(.46,.52),(.565,.35));arrow(ax,(.515,.75),(.595,.35))
    ax.text(.495,.48,'$a+b$',fontsize=6);ax.text(.605,.59,'$a$',fontsize=6)
    arrow(ax,(.565,.14),(.565,.06));ax.text(.565,-.06,'$a(a+b)$',ha='center',fontsize=7)
    box(ax,.69,.44,.12,.19,'Square');box(ax,.855,.44,.12,.19,'Multiply')
    box(ax,.775,.12,.12,.19,'Add')
    arrow(ax,(.825,.75),(.75,.64));arrow(ax,(.825,.75),(.915,.64))
    arrow(ax,(.75,.43),(.805,.32));arrow(ax,(.915,.43),(.875,.32))
    ax.text(.735,.35,'$a^2$',fontsize=6);ax.text(.923,.35,'$ab$',fontsize=6)
    arrow(ax,(.835,.11),(.835,.04));ax.text(.835,-.06,'$a^2+ab$',ha='center',fontsize=7)
    a=fig.add_axes([.10,.115,.365,.345]);b=fig.add_axes([.61,.115,.365,.345])
    f10.violin_axis(a,rows,'relative_accuracy','Relative accuracy','',True)
    f10.violin_axis(b,rows,'reconstruction_efficiency',r'Normalized reconstruction score $E_w$','',False)
    clean(fig)
    panel(fig,ax,'a',x=0,y=1.06);panel(fig,a,'b',x=-.14);panel(fig,b,'c',x=-.14)
    for aa in [a,b]:
        if aa.get_legend():aa.get_legend().remove()
    handles=f10.legend_handles()
    for handle in handles:
        if handle.get_label()=='control':handle.set_label('Control')
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.55),ncol=3,frameon=False,fontsize=6)
    export(fig,'main_fig6',out)
    cost=f10.load_cost_summary(f10.DEFAULT_COST_SUMMARY)
    fig=plt.figure(figsize=(90/25.4,72/25.4));a=fig.add_axes([.19,.28,.76,.67])
    f10.plot_cost_axis(a,cost,'');clean(fig);a.set_ylabel('Relative heuristic cost')
    export(fig,'supp_training_cost',out)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--only',type=int,nargs='*',default=[1,2,3,4,5,6]);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    for n in args.only:globals()['figure'+str(n)](args.output)
