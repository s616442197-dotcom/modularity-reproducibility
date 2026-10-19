"""Compact, editable scientific schematics; geometry in final PDF points.

Design reference: user-supplied simplified schematic PPTX. Architecture checked
against manuscript Methods and model.py (TorchTinyTransformer._encode_tensor).
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
import numpy as np

OUT=Path(__file__).resolve().parent
INK='#34424D'; TEXT='#25313B'; MUTED='#65737D'; TEAL='#34AEBB'
PALE='#DDF3F5'; LIGHT='#F5F8FA'; GRID='#BAC7CF'
RULE=['#0072BD','#D95319','#EDB120','#7E2F8E','#77AC30']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':6.5,'pdf.fonttype':42,'svg.fonttype':'none'})

class Panel:
 def __init__(self,w,h):
  self.w=w;self.h=h;self.fig=plt.figure(figsize=(w/72,h/72))
  self.ax=self.fig.add_axes([0,0,1,1]);self.ax.set(xlim=(0,w),ylim=(h,0));self.ax.axis('off')
 def text(self,x,y,s,size=6.5,bold=False,color=TEXT,ha='center'):
  self.ax.text(x,y,s,fontsize=size,fontweight='bold' if bold else 'normal',color=color,ha=ha,va='center',linespacing=1.2)
 def box(self,x,y,w,h,fill=LIGHT,edge=INK,r=2,lw=.6):
  self.ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',fc=fill,ec=edge,lw=lw))
 def line(self,pts,color=INK,lw=.65,ls='-'):
  a=np.array(pts);self.ax.plot(a[:,0],a[:,1],color=color,lw=lw,ls=ls,solid_capstyle='round')
 def arrow(self,x,y,xx,yy,color=INK,lw=.7,scale=5):
  self.ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=scale,color=color,lw=lw,shrinkA=0,shrinkB=0))
 def dot(self,x,y,color=INK,r=1.25):self.ax.add_patch(Circle((x,y),r,fc=color,ec='none'))
 def matrix(self,x,y,rows=5,cols=5,cell=3,col=TEAL,selected=()):
  for r in range(rows):
   for c in range(cols):self.ax.add_patch(Rectangle((x+c*cell,y+r*cell),cell,cell,fc=col if (r,c) in selected else 'white',ec=GRID,lw=.3))
 def save(self,name):
  for ext in ['pdf','svg']:self.fig.savefig(OUT/f'{name}.{ext}',transparent=False)
  self.fig.savefig(OUT/'_review'/f'{name}.png',dpi=220)
  plt.close(self.fig)

def mlp():
 p=Panel(196,175)
 p.text(98,10,'Shared multitask MLP',8,True)
 p.text(17,39,'Inputs',6.5,True)
 for k,seq in enumerate(['1010011','0111010']):
  y=56+k*53;p.text(17 if k==0 else 30,y-9,'x' if k==0 else 'y',5.9)
  for j,bit in enumerate(seq):
   x=2+j*4;p.box(x,y,4,8,fill='white',r=0,lw=.35);p.text(x+2,y+4,bit,4.7)
 p.box(2,78,30,15,PALE,TEAL,2)
 p.text(17,85.5,'[x ; y]',6.2,True)
 p.arrow(17,65,17,77);p.arrow(17,108,17,94)
 p.text(18,132,'Concatenate',5.5,color=MUTED)
 p.arrow(33,85,39,85,scale=4)
 for k in range(3):
  x=40+k*28;p.box(x,48,22,74,PALE,INK,3)
  p.text(x+11,58,f'FC{k+1}',6.5,True)
  for row in range(4):
   for col in range(3):p.dot(x+5+6*col,72+8*row,TEAL if row==2 and col==1 else INK,1.1)
  p.text(x+11,113,'300\nReLU',5.3,color=MUTED)
  if k<2:p.arrow(x+23,84,x+27,84,scale=4)
 p.arrow(119,84,125,84,scale=4)
 p.box(126,66,20,37,LIGHT,INK,2)
 p.text(136,84.5,'Linear\noutput',5.5,True)
 p.arrow(147,84,153,84,scale=4)
 p.text(174,39,'Outputs',6.5,True)
 p.box(154,48,40,74,'white',INK,3)
 for j,col in enumerate(RULE):
  y=53+j*13;p.box(158,y,32,10,LIGHT,'none',1)
  p.line([(159,y+2),(159,y+8)],col,2)
  p.text(167,y+5,f'R{j+1}',5.8,True,col)
  p.text(183,y+5,'P-dim',5.1,color=MUTED)
 p.text(80,132,'3 shared hidden layers',5.6,color=MUTED)
 p.text(164,132,'P = 7',5.6,color=MUTED)
 p.box(22,147,154,19,PALE,TEAL,3)
 p.text(99,156.5,'Joint loss + sparsity penalty',6.6,True)
 p.save('fig1a_schematic')

def profiling():
 p=Panel(326,175)
 p.text(163,10,'Functional profiling and token extraction',8,True)
 titles=['Trained model','Profile weights','Rule scores','Extract masks','Retain + test']
 for k,title in enumerate(titles):
  x=1+k*66;p.text(x+29,31,title,6.2,True)
  p.box(x,43,59,92,'white' if k in [0,4] else LIGHT,INK,3)
  if k<4:p.arrow(x+60,87,x+65,87,scale=4)
 # Every stage uses the same illustrative scalar-coordinate grid.
 masks=[[(0,0),(1,2),(2,1),(3,3)],[(0,0),(0,3),(2,2)],
        [(1,0),(1,2),(3,1)],[(0,2),(2,1),(3,0)],[(0,1),(2,3),(3,3)]]
 p.text(30,54,'Weight block W',5.8,True)
 p.matrix(18,67,4,4,6,TEAL,[(r,c) for r in range(4) for c in range(4)])
 for row in range(4):
  y=70+6*row;p.dot(9,y,INK,1.2);p.dot(51,y,INK,1.2)
  p.line([(11,y),(17,y)],GRID,.4);p.line([(43,y),(49,y)],GRID,.4)
 p.text(30,104,'Scalar weights',5.8)
 p.text(30,116,'between layers',5.8)
 p.text(30,127,'Illustrative grid',5.2,color=MUTED)
 # A scalar coordinate is removed from a matrix. No fabricated quantitative bar chart.
 p.matrix(86,51,4,4,5,'#D65353',[(1,2)])
 p.text(96,79,'Set one $w_{ij}=0$',5.8,True,color='#C64848')
 p.text(96,91,'Δf: baseline − ablated',5.0)
 p.text(96,102,r'$s_r=\mathbb{E}\|\Delta f_r\|^2$',6.2)
 p.text(96,115,'Repeat for all weights',5.0,color=MUTED)
 p.text(96,127,'or Taylor scoring',5.3,color=MUTED)
 # Five rule scores for the same scalar coordinate.
 for j,col in enumerate(RULE):
  y=57+13*j;p.text(145,y,f'R{j+1}',6,True,col)
  p.box(157,y-4,24,8,'white','none',1)
  p.text(169,y,f'$s_{j+1}(w)$',6,color=MUTED)
 p.text(162,126,'One score / weight',5.3,color=MUTED)
 for j,col in enumerate(RULE):
  y=48+j*13;p.text(211,y+5,f'R{j+1}',5.6,True,col)
  p.matrix(224,y,4,4,2.5,col,masks[j])
 p.text(228,121,'Rulewise fixed',5.3,color=MUTED)
 p.text(228,129,'quantile (0.75)',5.3,color=MUTED)
 # Apply all profiled matrix masks simultaneously and evaluate the model.
 p.text(294,51,'R1 example',5.6,True,RULE[0])
 p.matrix(284,60,4,4,5,RULE[0],masks[0])
 p.text(294,89,r"$W'=W\odot M_{R1}$",6.2)
 p.text(294,102,'Keep original values',5.1)
 p.text(294,116,'Evaluate all rules',5.6,True)
 p.text(294,126,'Relative accuracy',5.5,color=MUTED)
 p.box(15,144,296,26,PALE,TEAL,3)
 p.text(163,152,'Apply rule masks across all profiled matrices, including output weights.',6.1,True)
 p.text(163,162,'Biases and other unmasked parameters remain unchanged.',6.1)
 p.save('fig1b_schematic')

def transformer():
 p=Panel(252,176)
 p.text(126,8,'TinyTransformer',8,True)
 # Left-to-right input and token projection, then compact encoder details.
 for j,seq in enumerate(['1010011','0111010']):
  y=46+j*43;p.text(24,y-10,f'Input {j+1}',6.3,True)
  for k,bit in enumerate(seq):
   x=2+k*6;p.box(x,y,6,9,'white',INK,0,.35);p.text(x+3,y+4.5,bit,5.2)
  p.arrow(45,y+4.5,53,y+4.5)
 p.box(55,33,51,82,PALE,INK,3)
 p.text(80.5,44,'Shared input',6.1,True);p.text(80.5,53,'projection',6.1,True)
 for j in range(2):
  p.box(62,64+j*16,37,11,'white',TEAL,2);p.text(80.5,69.5+j*16,f'Token {j+1}',6,color=TEAL)
 p.text(80.5,101,'32-d + learnable',5.3,color=MUTED)
 p.text(80.5,109,'positional bias',5.3,color=MUTED)
 p.box(118,29,82,92,'white',INK,3)
 p.line([(107,75),(113,75),(113,35),(159,35)])
 p.arrow(159,35,159,42,scale=4)
 p.text(159,23,'Encoder block × 2',6.7,True)
 p.box(128,43,62,14,PALE,TEAL,2);p.text(159,50,'Self-attention',6.2,True)
 p.arrow(159,58,159,63,scale=4)
 p.box(128,64,62,10,LIGHT,GRID,2);p.text(159,69,'Add & layer norm',5.7)
 p.line([(159,35),(123,35),(123,69)],MUTED,.55)
 p.arrow(123,69,127,69,MUTED,.55,4)
 p.arrow(159,75,159,82,scale=4)
 p.box(128,83,62,14,PALE,TEAL,2);p.text(159,90,'FFN (64, ReLU)',6.1,True)
 p.arrow(159,98,159,103,scale=4)
 p.box(128,104,62,10,LIGHT,GRID,2);p.text(159,109,'Add & layer norm',5.7)
 p.line([(159,78),(123,78),(123,109)],MUTED,.55)
 p.arrow(123,109,127,109,MUTED,.55,4)
 p.line([(159,115),(159,118),(205,118),(205,75)])
 p.arrow(205,75,209,75,scale=4)
 p.box(210,57,38,36,PALE,TEAL,5);p.text(229,75,'Mean\npool',6.5,True)
 # Output projection spans one row, keeping all five rule-specific outputs visible.
 p.arrow(229,94,229,126)
 p.box(22,127,214,11,LIGHT,GRID,2,.5)
 p.text(129,132.5,'Linear output projection',6.2,True)
 for j,col in enumerate(RULE):
  x=22+j*44;p.arrow(x+19,138,x+19,143,scale=4)
  p.box(x,143,38,19,LIGHT,GRID,2,.5)
  p.line([(x+4,147),(x+4,158)],col,2)
  p.text(x+13,152.5,f'R{j+1}',6.1,True,col);p.text(x+28,152.5,'P-dim',5.0,color=MUTED)
 p.text(126,172,'Two tokens · mean pooling · five P-dimensional outputs (P = 7)',5.7,color=MUTED)
 p.save('fig2e_schematic')

def build():mlp();profiling();transformer()
if __name__=='__main__':build()
