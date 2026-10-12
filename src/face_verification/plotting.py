"""Identity-free descriptive plots; paired scores are not independent trials."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, det_curve

def plot_scores(pairs,path,*,title='Verification scores'):
    scores = np.array([p.score for p in pairs])
    labels = np.array([p.genuine for p in pairs])
    if len(scores)==0 or len(set(labels)) < 2:
        raise ValueError('Both genuine and impostor scores are needed for descriptive curves.')
    fig, axes = plt.subplots(1,3,figsize=(13,3.8))
    for mask,label in [(labels,'Genuine'),(~labels,'Impostor')]:
        axes[0].hist(scores[mask],bins=np.linspace(-1,1,41),density=True,alpha=.65,label=label)
    axes[0].set(xlabel='Cosine similarity',ylabel='Density')
    axes[0].legend()
    fpr,tpr,_ = roc_curve(labels,scores)
    axes[1].plot(fpr,tpr)
    axes[1].set(xlabel='False match rate',ylabel='True match rate',xlim=(0,1),ylim=(0,1),title='ROC (descriptive)')
    fpr,fnr,_ = det_curve(labels,scores)
    # Rate-space DET to keep zero-error points visible; not probit-scaled.
    axes[2].plot(fpr,fnr)
    axes[2].set(xlabel='False match rate',ylabel='False non-match rate',xlim=(0,1),ylim=(0,1),title='DET (rate coordinates)')
    fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=150,bbox_inches='tight')
    return fig
