"""Per-update behaviour counts from the saved rollouts (per episode): stalls (Chef calls that wait or talk while holding
raw *_slices, and the timesteps it holds them), repeated identical Collab actions, Collab calls, and the action
advantage these calls received. Writes eval/behavior_counts.json for plot_behavior.py.

    python runs/think_judge/count_behavior.py l2_none l2_sum_w0.5 ...
"""
import gzip,json,re,collections,sys,os
os.chdir(os.path.expanduser('~/Collab-Overcooked-private/runs/think_judge'))
RUNS=sys.argv[1:]
HOLD=re.compile(r'<Chef> holds ([^.]*)\.')
def act(r):
    m=re.search(r'Action\s*:\s*(.*)',r['response'] or '',re.S); return ' '.join(m.group(1).split()).strip('` ') if m else ''
out={}
for run in RUNS:
    n=len(open(f'rl/{run}/metrics.jsonl').readlines())
    for u in range(1,n+1):
        p=f'rl/{run}/rollouts/u{u:05d}.jsonl.gz'
        if not os.path.exists(p): continue
        rows=[json.loads(l) for l in gzip.open(p,'rt')]
        summ={str(r['episode_summary']):r['success'] for r in rows if 'episode_summary' in r}
        eps=collections.defaultdict(list)
        for r in rows:
            if 'call_index' in r: eps[str(r['episode'])].append(r)
        agg=collections.Counter()
        for ep,cs in eps.items():
            cs.sort(key=lambda r:(int(r['timestep']),int(r['call_index'])))
            raw_steps=set(); last={}
            for r in cs:
                a=act(r); ag=int(r['agent_index']); t=int(r['timestep'])
                collab=a.startswith('Collab'); wait=a.startswith('wait')
                agg['calls']+=1; agg['collab']+=collab; agg['wait']+=wait
                if collab and last.get(ag)==a: agg['repeat']+=1; agg['repeat_adv']+=float(r.get('adv_action') or 0)
                last[ag]=a
                if ag==0:
                    m=HOLD.search(r['observation'] or '')
                    h=m.group(1) if m else ''
                    if h.endswith('_slices') and not h.startswith(('baked','boiled')):
                        raw_steps.add(t)
                        if collab or wait: agg['stall_calls']+=1; agg['stall_adv']+=float(r.get('adv_action') or 0)
            agg['stall_steps']+=len(raw_steps)
            agg['steps']+=max(int(r['timestep']) for r in cs)+1
        E=len(eps)
        out.setdefault(run,[]).append(dict(update=u,episodes=E,success=sum(map(bool,summ.values()))/max(1,len(summ)),
            **{k:agg[k]/E for k in ('calls','collab','wait','repeat','stall_calls','stall_steps','steps','repeat_adv','stall_adv')}))
    print(run,len(out[run]),flush=True)
json.dump(out,open('eval/behavior_counts.json','w'))
