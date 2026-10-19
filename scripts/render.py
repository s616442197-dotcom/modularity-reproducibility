"""Vector plots recomputed from individual-run observations, in current panel order."""
from pathlib import Path
import io, json, csv, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pypdf import PdfReader, PdfWriter, Transformation
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'example')]
import figure2 as f2, figure4 as f4, figure5 as f5, figure6 as f6, figure8 as f8, figure10 as f10, ruleweight as rw
from example.composite import multitask_vs_separate_cost as cost
RULE=['#0072BD','#D95319','#EDB120','#7E2F8E','#77AC30']; LAYER=['#4DBEEE','#A2142F','#0072BD']
def read(p):
 rows=list(csv.DictReader(p.open()))
 for r in rows:
  for k,v in r.items():
   try:r[k]=float(v)
   except (ValueError,TypeError):pass
 return rows
def style():
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'legend.fontsize':6,'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.5,'savefig.bbox':None})
def save(fig,out,n):
 fig.savefig(out/f'main_fig{n}.pdf');fig.savefig(out/f'main_fig{n}.png',dpi=140);plt.close(fig)
def label(ax,s):
 pos=ax.get_position();x=max(-.14,(.02-pos.x0)/pos.width)
 ax.text(x,1.04,s,transform=ax.transAxes,fontweight='bold',fontsize=12)
def legend(ax,colors,names,**kw):ax.legend([Patch(facecolor=c) for c in colors],names,frameon=False,**kw)
def violin(ax,a,colors,labels,aligned=False,log=False):
 # a: repeats x series x groups; same x coordinates across layers for guide lines.
 a=np.asarray(a,float);a=np.log1p(a) if log else a
 for k,color in enumerate(colors):
  x=np.arange(a.shape[2])+(0 if aligned else (k-(len(colors)-1)/2)*.14)
  vs=[a[:,k,j][np.isfinite(a[:,k,j])] for j in range(a.shape[2])]
  vs=[v if np.ptp(v)>1e-12 else np.r_[v,v[0]+1e-7] for v in vs]
  p=ax.violinplot(vs,positions=x,widths=.58 if aligned else .14,showextrema=False,showmedians=True)
  for b in p['bodies']:b.set_facecolor(color);b.set_edgecolor(color);b.set_alpha(.32 if aligned else .65)
  p['cmedians'].set_color(color);p['cmedians'].set_linewidth(.8)
  if aligned:ax.plot(x,np.nanmedian(a[:,k,:],axis=0),'-o',color=color,lw=.8,ms=2)
 ax.set_xticks(range(len(labels)),labels,rotation=45 if len(labels)>6 else 0)
 ax.spines[['top','right']].set_visible(False)
def capture(fn,*args,**kw):
 figs=[];oldsave=plt.Figure.savefig;oldclose=plt.close
 plt.Figure.savefig=lambda fig,*a,**k:figs.append(fig) if fig not in figs else None
 plt.close=lambda *a,**k:None
 try:fn(*args,**kw)
 finally:plt.Figure.savefig=oldsave;plt.close=oldclose
 return figs[-1]
def inset_pdf(file,source,box):
 # box is x,y,width,height in bottom-left PDF points; preserves vectors.
 page=PdfReader(file).pages[0];p=PdfReader(source).pages[0];x,y,w,h=box
 if x<0 or y<0 or x+w>float(page.mediabox.width) or y+h>float(page.mediabox.height):
  raise ValueError(f'Schematic placement exceeds the PDF canvas: {file}, {box}')
 s=min(w/float(p.mediabox.width),h/float(p.mediabox.height))
 page.merge_transformed_page(p,Transformation().scale(s).translate(x,y));writer=PdfWriter();writer.add_page(page);writer.write(file)

def overlay_labels(file,labels):
 page=PdfReader(file).pages[0];fig=plt.figure(figsize=(float(page.mediabox.width)/72,float(page.mediabox.height)/72))
 for x,y,letter in labels:fig.text(x,y,letter,fontsize=12,fontweight='bold')
 buf=io.BytesIO();fig.savefig(buf,format='pdf',transparent=True);plt.close(fig);buf.seek(0);page.merge_page(PdfReader(buf).pages[0]);writer=PdfWriter();writer.add_page(page);writer.write(file)
def recompute_ruleweight(metrics):
 for m in metrics:
  for k,v in list(m.items()):
   if isinstance(v,list):m[k]=np.asarray(v,float)
  n=m['num_runs'];ret=m['retained_runs'];diag=np.diagonal(ret,axis1=1,axis2=2)
  margins=np.stack([ret[:,i,i]-np.nanmean(np.delete(ret[:,:,i],i,axis=1),axis=1) for i in range(5)],axis=1)
  for prefix,a in [('retained',ret),('diag_retained',diag),('weight_ratio',m['weight_ratio_runs']),('baseline_acc',m['baseline_acc_runs']),('specialization_margin',margins)]:
   m[prefix+'_mean']=np.nanmean(a,axis=0);m[prefix+'_sem']=np.nanstd(a,axis=0)/np.sqrt(n)
 return metrics
def render(data,out):
 out.mkdir(parents=True,exist_ok=True);style()
 z=np.load(data/'robustness.npz');sel=json.loads((data/'selection.json').read_text());assert len(sel['transformer'])==23
 get=lambda n,k:z[n+'__'+k]
 import scripts.draw_schematics as sch
 sch.OUT=out;(out/'_review').mkdir(exist_ok=True);sch.build();style()
 counts=get('binary','weight_combo_exclusive_counts');acc=get('binary','weight_combo_retained_accuracy');labs=f2.combo_labels(5,31)
 fig=plt.figure(figsize=(7.5,8.7));c=fig.add_axes([.09,.49,.88,.17]);d=fig.add_axes([.09,.27,.32,.14]);e=fig.add_axes([.60,.27,.37,.14]);f=fig.add_axes([.09,.055,.88,.14])
 violin(c,counts[:,:3],LAYER,labs,log=True);c.set_ylabel('Log(1 + weight-token count)');label(c,'c');legend(c,LAYER,['Layer 1','Layer 2','Layer 3'],loc='upper left',ncol=3)
 for ax,ids,l in [(d,np.arange(5),'d'),(e,np.arange(5,15),'e'),(f,np.arange(15,31),'f')]:
  violin(ax,acc[:,:,ids],RULE,[labs[i] for i in ids]);ax.axhline(1,c='.5',ls='--',lw=.5);ax.set_ylabel('Relative accuracy');label(ax,l)
 legend(d,RULE,[f'Rule {i}' for i in range(1,6)],loc='center left',bbox_to_anchor=(1.02,.5))
 save(fig,out,1)
 inset_pdf(out/'main_fig1.pdf',out/'fig1a_schematic.pdf',(8,455,195,170));inset_pdf(out/'main_fig1.pdf',out/'fig1b_schematic.pdf',(208,455,324,170))
 overlay_labels(out/'main_fig1.pdf',[(.03,.97,'a'),(.39,.97,'b')])
 # Current Fig.2: Pattern, MNIST; row-normalized Taylor matrix, Taylor MLP; schematic, 23-run Transformer.
 fig=plt.figure(figsize=(7.5,8));positions=[[x,y,.36,.19] for y in [.75,.43,.09] for x in [.10,.61]];axs=[fig.add_axes(p) for p in positions]
 for ax,n in [(axs[0],'pattern'),(axs[1],'mnist'),(axs[3],'taylor_mlp'),(axs[5],'transformer')]:
  f4.plot_retained_demo_axes(ax,get(n,'weight_combo_retained_accuracy'),'',compact=True,tick_size=6,rotate=50)
  if ax.get_legend():ax.get_legend().remove()
 matrix,_=f4.taylor_sensitivity_matrix(data/'no_optional_profile.json',get('taylor_mlp','weight_combo_retained_accuracy'))
 f4.plot_taylor_heatmap(axs[2],matrix,'',with_colorbar=False);axs[2].set_aspect('auto');axs[4].axis('off')
 for ax,l in zip(axs,'abcdef'):
  if l!='e':label(ax,l)
 legend(axs[0],RULE,[f'Rule {i}' for i in range(1,6)],loc='center left',bbox_to_anchor=(1.01,.5));save(fig,out,2)
 inset_pdf(out/'main_fig2.pdf',out/'fig2e_schematic.pdf',(10,20,260,165))
 overlay_labels(out/'main_fig2.pdf',[(.05,.32,'e')])
 single=read(data/'single_rule.csv');nip=json.loads((data/'nip.json').read_text())
 fig=plt.figure(figsize=(7.5,8.3)); stats={}
 for row,conds,letter in [(0,[(1,0),(1,1),(1,5)],'a'),(1,[(.5,5),(1,5),(2,5)],'b')]:
  for j,(g,l) in enumerate(conds):
   a=np.full((30,5,5),np.nan);rat=np.full((30,5),np.nan)
   for r in single:
    if r['gamma']==g and r['lasso']==l:
     run=int(r['run'].split('_')[-1]);i=int(r['retained_rule_index'])-1;k=int(r['evaluated_rule_index'])-1;a[run,k,i]=r['relative_accuracy'];rat[run,i]=r['retained_weight_ratio']
   assert np.isfinite(a).all();ax=fig.add_axes([.10+j*.30,.77-row*.31,.235,.15]);ra=fig.add_axes([.10+j*.30,.68-row*.31,.235,.055])
   for k,color in enumerate(RULE):ax.bar(np.arange(5)+(k-2)*.15,np.mean(a[:,k],0),.14,yerr=np.std(a[:,k],0,ddof=1),color=color,error_kw={'elinewidth':.4,'capsize':1})
   ax.set_xticks(range(5),range(1,6));ax.set_ylabel('Relative accuracy' if j==0 else '');ax.text(.97,.96,f'$l={l},\\gamma={g}$',ha='right',va='top',transform=ax.transAxes,fontsize=6)
   ra.errorbar(range(5),rat.mean(0),yerr=rat.std(0,ddof=1),fmt='o',ms=2,lw=.5);ra.set_xticks(range(5),range(1,6));ra.set_ylabel('Weight\nfraction' if j==0 else '',fontsize=6)
   if j==0:label(ax,letter)
   stats[f'g{g}_l{l}']={'mean':a.mean(0).tolist(),'sd':a.std(0,ddof=1).tolist(),'ratio':rat.mean(0).tolist()}
 for ax,conds,l in [(fig.add_axes([.10,.09,.34,.16]),[(1,x) for x in [0,1,5,10]],'c'),(fig.add_axes([.64,.09,.32,.16]),[(x,5) for x in [.5,1,2]],'d')]:
  a=np.stack([np.stack([f5.modularity_samples(nip,g,v,k) for k in [1,2,3]],axis=1) for g,v in conds],axis=2)
  assert a.shape[0]==30;violin(ax,a,LAYER,[str(v if l=='c' else g) for g,v in conds],aligned=True);ax.set_ylim(-.03,1.03);ax.set_ylabel('$M$ (NIP)');ax.set_xlabel('$l$' if l=='c' else '$\\gamma$');label(ax,l)
  if l=='c':legend(ax,LAYER,['Layer 1','Layer 2','Layer 3'],loc='center left',bbox_to_anchor=(1.01,.5))
 fig.legend([Patch(fc=c) for c in RULE],[f'Rule {i}' for i in range(1,6)],ncol=5,loc='upper center',frameon=False);save(fig,out,3)
 # Geometry panels use the original scientific plotter, with current a/b/c/d/e grouping.
 rows=read(data/'time.csv');metrics=read(data/'geometry.csv');hist=read(data/'correlation_hist.csv')
 fig=capture(f8.plot_fig8,metrics,hist,[0,1,2,3,5,10,15],[0,1,5,10],out,['pdf']);fig.set_size_inches(7.5,8)
 ga=list(fig.axes)
 for i,ax in enumerate(ga):
  ax.set_title('');ax.set_position(([.10+i*.225,.43,.17,.16] if i<4 else [.10+(i-4)*.30,.10,.235,.18]))
  if ax.get_legend():ax.get_legend().remove()
  for t in list(ax.texts):
   if t.get_text() in list('abcdefg'):t.remove()
  if i<4:ax.text(.97,.95,f'$l={[0,1,5,10][i]}$',ha='right',va='top',transform=ax.transAxes)
  else:
   for line in ax.lines:
    if len(line.get_xdata())==7:line.set_linestyle('-')
 for j,l in enumerate([0,1,5,10]):
  ax=fig.add_axes([.10+j*.225,.75,.17,.16]);a=np.stack([np.stack([np.array([r['modularity'] for r in rows if r['lasso']==l and r['epoch']==ep and r['layer_index']==k]) for k in [1,2,3]],axis=1) for ep in [1000,2000,3000,4000,5000]],axis=2)
  assert a.shape==(30,3,5);violin(ax,a,LAYER,['1','2','3','4','5'],aligned=True);ax.set_ylim(-.03,1.03);ax.set_xlabel('Epoch (×10³)');ax.text(.98,.96,f'$l={l}$',ha='right',va='top',transform=ax.transAxes)
  if j==0:label(ax,'a');ax.set_ylabel('$M$')
 label(ga[0],'b')
 for ax,s in zip(ga[4:],'cde'):label(ax,s)
 fig.legend([Patch(fc=c) for c in LAYER],['Layer 1','Layer 2','Layer 3'],ncol=3,loc='upper center',frameon=False)
 # Rebuild lower panels on identical x positions for each series and its guide line.
 specifications=[('pca_entropy',[1,2,3,4],LAYER+[f8.OUTPUT_COLOR],['Layer 1','Layer 2','Layer 3','Output'],'PCA entropy'),('cka_hidden_pair',['1-2','2-3','1-3'],f8.PAIR_COLORS,['1–2','2–3','1–3'],'CKA'),('cka_output',['1-out','2-out','3-out'],f8.OUTPUT_PAIR_COLORS,['1–out','2–out','3–out'],'CKA')]
 for ax,(metric,series,colors,names,yl),letter in zip(ga[4:],specifications,'cde'):
  ax.clear();a=np.stack([np.stack([np.array([r['value'] for r in metrics if r['metric']==metric and r['lasso']==l and r['layer_index' if metric=='pca_entropy' else 'pair']==s]) for s in series],axis=1) for l in [0,1,2,3,5,10,15]],axis=2)
  assert a.shape[0]==30;violin(ax,a,colors,['0','1','2','3','5','10','15'],aligned=True);ax.set_xlabel('$l$');ax.set_ylabel(yl);label(ax,letter)
  legend(ax,colors,names,loc='upper center',bbox_to_anchor=(.5,1.32),fontsize=5,ncol=2)
 save(fig,out,4)
 metrics_rw=recompute_ruleweight(json.loads((data/'ruleweight_runs.json').read_text()));rw.OUTPUT_ROOT=out;summary=rw.write_summary_tables(metrics_rw)
 matrices=f6.figure_matrices(summary);fig=plt.figure(figsize=(7.5,5.8));axs=[]
 for i,m in enumerate(matrices):
  ax=fig.add_axes([.09+i*.31,.64,.20,.27]);axs.append(ax)
  f6.draw_heatmap(fig,ax,m*(1e4 if i==1 else 1),'','',['Accuracy change','Weight-fraction change (×10⁻⁴)','Specialization margin'][i],'{:+.2f}' if i<2 else '{:.2f}','RdBu_r' if i<2 else 'viridis',i<2)
  fig.axes[-1].remove();ax.set_aspect('auto');ax.set_position([.09+i*.31,.64,.20,.27]);cb=fig.add_axes([.09+i*.31,.52,.20,.015]);fig.colorbar(ax.images[0],cax=cb,orientation='horizontal');cb.set_xlabel(['Accuracy change','Weight-fraction change (×10⁻⁴)','Specialization margin'][i],fontsize=6);ax.set_xlabel('Evaluated rule');ax.set_ylabel('Upweighted rule');label(ax,'abc'[i])
 a=fig.add_axes([.10,.12,.34,.25]);b=fig.add_axes([.63,.12,.33,.25]);x,y,xe,ye=f6.matched_target_statistics(summary)
 for i in range(5):a.errorbar(y[i]*1e4,x[i],xerr=ye[i]*1e4,yerr=xe[i],fmt='o',color=RULE[i],ms=4,capsize=2)
 xx=np.linspace(min(y)*1e4,max(y)*1e4,50);a.plot(xx,np.polyval(np.polyfit(y*1e4,x,1),xx),'--',c='.4');a.text(.02,.95,f'Pearson r = {np.corrcoef(x,y)[0,1]:.2f}',va='top',transform=a.transAxes)
 trials=f6.paired_trial_rows(metrics_rw);f6.draw_trial_scatter(b,trials)
 b.set_title('',loc='left')
 for t in list(b.texts):
  if t.get_text()=='e':t.remove()
 for ax,l in [(a,'d'),(b,'e')]:label(ax,l);ax.set_xlabel('Retained-weight change (×10⁻⁴)');ax.set_ylabel('Accuracy change')
 save(fig,out,5)
 # Recomposition graph is preserved as an editable plotting function; quantitative panels are recomputed.
 import manuscript_figures as mf
 oldread=mf.read;oldload=f10.load_cost_summary;oldexport=mf.export
 cs=cost.aggregate_costs(*[json.loads((data/'cost'/n).read_text()) for n in ['multitask_runs.json','extraction_runs.json','separate_runs.json']])
 mf.read=lambda p:read(data/'recomposition.csv');f10.load_cost_summary=lambda p:cs
 def ex(fig,stem,dest):
  if stem=='main_fig6':
   fig.set_size_inches(7.5,7);axes=list(fig.axes);axes[0].set_position([.045,.71,.93,.24]);axes[1].set_position([.10,.38,.34,.23]);axes[2].set_position([.63,.38,.33,.23]);ca=fig.add_axes([.26,.07,.55,.18]);f10.plot_cost_axis(ca,cs,'');label(ca,'d')
   for t in list(fig.texts):
    if t.get_text() in 'abc':t.remove()
   axes[0].text(0,1.04,'a',transform=axes[0].transAxes,fontsize=12,fontweight='bold')
   for ax,letter in zip(axes[1:],'bc'):label(ax,letter)
   for leg in fig.legends:leg.set_bbox_to_anchor((.54,.68))
  for t in fig.findobj(matplotlib.text.Text):t.set_fontfamily('DejaVu Sans')
  fig.savefig(out/(stem+'.pdf'));fig.savefig(out/(stem+'.png'),dpi=140);plt.close(fig)
 mf.export=ex
 try:mf.figure6(out)
 finally:mf.read=oldread;f10.load_cost_summary=oldload;mf.export=oldexport
 result={'selected_counts':{k:len(v) for k,v in sel.items()},'fig2c_matrix':matrix.tolist(),'fig3_bar_statistics':stats,'fig5_matrices':[v.tolist() for v in matrices],'fig5_pair_count':len(trials),'fig5_mean_r':float(np.corrcoef(x,y)[0,1]),'fig5_trial_r':float(np.corrcoef([r['accuracy_delta'] for r in trials],[r['weight_ratio_delta'] for r in trials])[0,1]),'cost':cs}
 (out/'statistics.json').write_text(json.dumps(result,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)))
 return result
