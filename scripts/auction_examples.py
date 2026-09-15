"""Synthetic auction calculations under the explicit assumptions in the paper."""
import json
import numpy as np
import matplotlib.pyplot as plt
from experiments import ROOT,savefig,COLORS


def main():
    prices=np.array([100.,100.01,100.02,100.03])
    demand=np.array([1200,1200,800,300]);supply=np.array([200,800,1100,1100])
    volume=np.minimum(demand,supply);imbalance=demand-supply
    candidates=np.flatnonzero(volume==volume.max())
    winner=candidates[np.argmin(np.abs(imbalance[candidates]))]
    assert winner==2 and volume[winner]==800
    y=np.linspace(0,1000,401)
    objective=lambda z:2e-5*(1000-z)**2+.012*(1000-z)+2e-5*z*z+.002*z
    optimum=625.0
    assert abs(objective(optimum)-16.375)<1e-12
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for a,label,color in [(demand,'Eligible buys',COLORS[0]),(supply,'Eligible sells',COLORS[1]),(volume,'Paired volume',COLORS[2])]:
        axes[0].plot(prices,a,'o-',label=label,color=color)
    axes[0].ticklabel_format(useOffset=False,style='plain',axis='x')
    axes[0].set_xticks(prices,[f'{p:.2f}' for p in prices]);axes[0].set(xlabel='Candidate price (USD/share)',ylabel='Shares')
    axes[0].legend(frameon=False,fontsize=8);axes[0].grid(alpha=.15)
    axes[1].plot(y,objective(y));axes[1].scatter([optimum],[objective(optimum)],color=COLORS[2],zorder=3)
    axes[1].annotate('625 shares; $16.375',(optimum,objective(optimum)),xytext=(330,23),arrowprops={'arrowstyle':'->','color':'#7b8290'},fontsize=9)
    axes[1].set(xlabel='Auction commitment (shares)',ylabel='Objective (USD)');axes[1].grid(alpha=.15)
    savefig(fig,'07_auction_math.png')
    r={'prices':prices.tolist(),'demand':demand.tolist(),'supply':supply.tolist(),
       'paired':volume.tolist(),'imbalance':imbalance.tolist(),'nasdaq_selected_price':float(prices[winner]),
       'nasdaq_matched_shares':800,'nyse_stipulated_price':100.01,
       'nyse_matched_without_dmm':800,'nyse_matched_with_200_dmm_sell_shares':1000,
       'at_price_capacities':[600,300,300],'parity_allocation':[200,200,200],
       'global_fifo_comparator':[600,0,0],
       'endpoint':{'auction_shares':625,'continuous_shares':375,'objective':16.375,
                   'continuous_only':32.0,'auction_only':22.0},
       'scope':'Artificial unrestricted book; NYSE DMM response is stipulated, not inferred.'}
    (ROOT/'results/auctions.json').write_text(json.dumps(r,indent=2)+'\n')
    print('Auction calculations and figure regenerated.')


if __name__=='__main__':main()
