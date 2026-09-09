import json
from pathlib import Path
from process_runtime import Session
from batched_session import LocalController
P=Path(__file__).parent/'verification/current_matrix_20260909'
if __name__=='__main__':
    cases=json.loads((P/'manifest.json').read_text())
    for i in [65,59,78,90]:
        c=cases[i];s=Session(c['options'],capture=False,controller_factory=LocalController,log_root=P/f'detail_{i}')
        r=s.run();(P/f'case_{i}.json').write_text(json.dumps(r,indent=2),encoding='utf8')
        print('CASE',i,'config',{k:v for k,v in r['config'].items() if 'TIMEOUT' in k or 'MAX_DISTANCE' in k},flush=True)
        for e in r['events']:
            print(e['t'],e['state'],e['command'],e['lines'],flush=True)
