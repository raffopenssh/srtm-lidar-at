import gzip, json, time, sys, requests
from collections import Counter
d=json.load(gzip.open('data/shares/WILHELM.json.gz')); st=d['state']
geom=st['geometry'] if isinstance(st['geometry'],str) else json.dumps(st['geometry'])
base={'include_ortho':'true','include_copernicus':'true','include_cadastre':'true','include_hansen':'true','include_infra':'true','include_temporal':'true','mark_uncertain':'true','async':'true'}
m=sys.argv[1]
r=requests.post('http://localhost:8000/api/v1/segment', json=dict(base, model=m, geometry=geom), timeout=120); j=r.json(); tid=j.get('task_id')
print(m, r.status_code, {k:v for k,v in j.items() if k in ('task_id','status','error')}, 'cached' if 'features' in j else '')
t0=time.time()
res=j if 'features' in j else None
while res is None:
    p=requests.get(f'http://localhost:8000/api/v1/segment/progress?task_id={tid}', timeout=30).json()
    if p.get('done') or p.get('error') or p.get('step') in ('done','error'):
        if p.get('error'): print('ERROR', p); sys.exit(2)
        break
    time.sleep(3)
    if time.time()-t0>300: print('timeout', p); sys.exit(1)
if res is None:
    print('progress done', round(time.time()-t0), 's')
    res=requests.get(f'http://localhost:8000/api/v1/segment/result?task_id={tid}', timeout=120).json()
fs=res.get('features',[]); print(m, len(fs), res.get('meta',{}).get('model'), res.get('meta',{}).get('model_classifier','')[:24])
print(dict(Counter(f['properties']['type'] for f in fs).most_common(10))); print(dict(Counter(f['properties']['classifier_source'] for f in fs)))
json.dump(res, open(f'/tmp/wilhelm_{m}.json','w'))
