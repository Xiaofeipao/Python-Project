"""Render report tables/plots from existing artifacts. Never train or call APIs."""
import hashlib
import json
from decimal import Decimal
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
LATEX=ROOT/'docs/report/latex'
RUNS=json.loads((ROOT/'runs/best.json').read_text(encoding='utf-8'))['reported_experiments']


def escape(value):
    return str(value).replace('_',r'\_').replace('%',r'\%')


def load_run(name):
    folder=ROOT/'runs'/name
    record={key:json.loads((folder/file).read_text()) for key,file in [('summary','summary.json'),('manifest','manifest.json'),('evaluation','evaluation.json')]}
    record['trials']=[json.loads(line) for line in (folder/'trials.jsonl').read_text().splitlines()]
    events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
    calls=[json.loads(line) for line in (folder/'model_calls.jsonl').read_text().splitlines()]
    record.update(run=name,calls=len(calls),accepted=sum(e['event']=='llm_plan_selected' for e in events),invalid=sum(e['event']=='llm_invalid_plan' for e in events),input_tokens=sum(c.get('input_tokens') or 0 for c in calls),output_tokens=sum(c.get('output_tokens') or 0 for c in calls),cost=str(sum((Decimal(c['cost_rmb']) for c in calls),Decimal(0))))
    assert record['manifest']['source_sha256']==record['evaluation']['source_sha256']
    return record


def main():
    inventory=json.loads((ROOT/'docs/accounting/ledger_inventory.json').read_text())
    for row in inventory['ledgers']:
        assert hashlib.sha256((ROOT/row['source']).read_bytes()).hexdigest()==row['sha256'], 'Ledger inventory is stale'
    records=[load_run(RUNS[task]['run']) for task in ('coupon','headlines')]
    data={'date':'2026-11-25','primary_runs':records,'accounting':inventory,'frontier':json.loads((ROOT/'docs/development/current_results.json').read_text())}
    (LATEX/'report_data.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    tex=[]
    tex.append(r'\newcommand{\PrimaryResults}{\begin{tabular}{lrrrr}\toprule Task&Val. (\%)&Test (\%)&Correct&Macro-F1\\\midrule')
    for task,r in zip(('Coupon','Headlines'),records):
        e=r['evaluation'];correct=round(e['score']*e['rows']);macro=e['classification_report']['macro avg']['f1-score']
        tex.append(f"{task}&{r['summary']['validation_score']*100:.4f}&{e['score']*100:.4f}&{correct}/{e['rows']}&{macro:.4f}\\\\")
    tex.append(r'\bottomrule\end{tabular}}')
    tex.append(r'\newcommand{\CallResults}{\begin{tabular}{lrrrrr}\toprule Task&Calls&Plans&Invalid&Input&Output\\\midrule')
    for task,r in zip(('Coupon','Headlines'),records):tex.append(f"{task}&{r['calls']}&{r['accepted']}&{r['invalid']}&{r['input_tokens']}&{r['output_tokens']}\\\\")
    tex.append(r'\bottomrule\end{tabular}}')
    tex.append(r'\newcommand{\LedgerResults}{\begin{tabular}{lrr}\toprule Evaluation ID (date omitted)&Calls&Cost (CNY)\\\midrule')
    for row in sorted(inventory['ledgers'],key=lambda v:v['evaluation_id']):
        name=row['evaluation_id'].rsplit('-',1)[0]
        tex.append(f"\\texttt{{{escape(name)}}}&{row['calls']}&{row['recorded_cost_rmb']}\\\\")
    tex.append(r'\midrule Total&'+str(inventory['call_count'])+'&'+inventory['recorded_cost_rmb']+r'\\\bottomrule\end{tabular}}')
    labels={'cross_boost_blend':'Boost blend','minilm_logistic':'MiniLM LR','logistic':'Logistic','text_blend':'Text blend','text_blend_diff':'Text blend (diff)','linear_svm':'Linear SVM','dummy':'Prior baseline','pairwise_overlap':'Overlap LR','pairwise_blend':'Pair blend'}
    for task,r in zip(('Coupon','Headlines'),records):
        tex.append('\\newcommand{\\'+task+r'Trials}{\begin{tabular}{rlrr}\toprule Trial&Family&Mean (\%)&SD (pt)\\\midrule')
        for t in r['trials']:
            name=labels.get(t['candidate']['family'],escape(t['candidate']['family']))
            if t['candidate'].get('pairwise_word_features'):name+=' + words'
            if t.get('status')=='success':tex.append(f"{t['trial']}&{name}&{t['score']*100:.4f}&{t.get('validation_std',0)*100:.4f}\\\\")
            else:tex.append(f"{t['trial']}&{name}&\\multicolumn{{2}}{{c}}{{{escape(t['status'])}}}\\\\")
        tex.append(r'\bottomrule\end{tabular}}')
    (LATEX/'generated.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(10,3.1))
    for ax,task,r in zip(axes,('Coupon','Headlines'),records):
        rows=[t for t in r['trials'] if t['status']=='success' and t['candidate']['family']!='dummy']
        x=[t['trial'] for t in rows];y=[100*t['score'] for t in rows];sd=[100*t.get('validation_std',0) for t in rows]
        ax.errorbar(x,y,yerr=sd,fmt='o-',color='#61758a',linewidth=1,markersize=4,label='Candidate mean and seed SD')
        retained=[];best=-1
        for value in y:best=max(best,value);retained.append(best)
        ax.plot(x,retained,color='#176d91',linewidth=2,label='Retained validation maximum')
        ax.set(title=task,xlabel='Trial',ylabel='Validation accuracy (%)');ax.grid(alpha=.2)
        ax.set_ylim(min(y)-1,max(y)+1);ax.legend(fontsize=9)
    fig.tight_layout();(LATEX/'figures').mkdir(exist_ok=True)
    fig.savefig(LATEX/'figures/current_validation.png',dpi=220);plt.close(fig)
    print('Generated report data, tables and plot from existing artifacts; per-run runtime identity verified')


if __name__=='__main__':main()
