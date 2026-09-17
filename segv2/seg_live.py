import gzip, json, time, sys, requests
from collections import Counter
d=json.load(gzip.open('data/shares/WILHELM.json.gz')); st=d['state']
geom=st['geometry'] if isinstance(st['geometry'],str) else json.dumps(st['geometry'])
base={'include_ortho':'true','include_copernicus':'true','include_cadastre':'true','include_hansen':'true','include_infra':'true','include_temporal':'true','mark_uncertain':'true','async':'true'}
m=sys.argv[1]
r=requests.post('http://localhost:8000/api/v1/segment', json=dict(base, model=m, geometry=geom), timeout=60); j=r.json(); tid=j.get('task_id'); print(m, r.status_code, j)
t0=time.time()
while True:
    p=requests.get(f'http://localhost:8000/api/v1/segment/progress?task_id={tid}', timeout=30).json()
    if p.get('done') or p.get('error') or p.get('step') in ('done','error'):
        if p.get('error'): print('ERROR', p); sys.exit(2)
        break
    time.sleep(5)
    if time.time()-t0>1500: print('timeout', p); sys.exit(1)
print('progress', p.get('step'), round(time.time()-t0), 's')
res=requests.get(f'http://localhost:8000/api/v1/segment/result?task_id={tid}', timeout=120).json()
fs=res.get('features',[]); print(m, len(fs), res.get('meta',{}).get('model'), res.get('meta',{}).get('model_classifier','')[:24])
print(dict(Counter(f['properties']['type'] for f in fs).most_common(10))); print(dict(Counter(f['properties']['classifier_source'] for f in fs)))
json.dump(res, open(f'/tmp/wilhelm_{m}.json','w'))
