"""Product 2.3: robust top-N ranking + plausibility gate + v2_verify check."""
import os, sys, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import austria_processor as ap, v2_verify as vv, quality_flags as qf


class O:
    def __init__(s, t, hmax, p90, hmean, area, mm=False, conf=1.0):
        s.obj_type = t; s.height_max = hmax; s.height_p90 = p90; s.height_mean = hmean
        s.area_sqm = area; s.is_manmade = mm; s.confidence = conf; s.rf_confidence = conf
        s.ndvi_mean = 0.5


objs = [O('crop', 81.01, 3.0, 7.82, 56), O('crop', 80.45, 60.0, 21.66, 20),
        O('water', 78.13, 10.0, 15.41, 23), O('tree', 73.73, 18.8, 7.85, 1439),
        O('tree', 66.14, 20.51, 11.58, 240), O('tree', 57.53, 54.52, 17.38, 20),   # mast-like
        O('tree', 48.57, 27.19, 19.84, 5205), O('tree', 31.2, 28.0, 20.0, 300),
        O('roof', 22.0, 20.0, 15.0, 400, True), O('mast', 45, 40, 30, 12, True),
        O('grass', 2.5, 0.4, 0.3, 900), O('tree', 85, 84, 60, 3000), O('fence', 14, 13, 5, 40, True)]
top = ap._top_candidates(objs, 10)
types = [(o.obj_type, o.height_max) for o, _, _ in top]
assert ('crop', 81.01) not in types and ('water', 78.13) not in types
assert ('tree', 85) not in types and ('tree', 57.53) not in types and ('fence', 14) not in types
assert ap._robust_height(objs[3]) == 28.8
assert [hr for _, hr, _ in top] == sorted([hr for _, hr, _ in top], reverse=True)
mm = ap._top_candidates(objs, 10, manmade_only=True)
assert [o.obj_type for o, _, _ in mm] == ['mast', 'roof']
assert qf.apply_rules({'obj_type': 'fill', 'height_max_m': 7, 'area_sqm': 300})[0]['severity'] == 'medium'
assert qf.apply_rules({'obj_type': 'wall', 'height_max_m': 9, 'area_sqm': 30})[0]['severity'] == 'high'

row = {'type': 'tree', 'height_max_m': 30, 'height_robust_m': 30}
doc = {'product_version': vv.PRODUCT_VERSION, 'top_10_objects': [row], 'top_10_trees': [dict(row)],
       'top_manmade_objects': []}
r = vv.Report(); vv.check_v23_top_ranking(doc, r); assert r['ok'], r.summary()
bad = copy.deepcopy(doc); bad['top_10_trees'][0]['height_max_m'] = 81
r = vv.Report(); vv.check_v23_top_ranking(bad, r); assert not r['ok']
bad = copy.deepcopy(doc); del bad['top_10_objects'][0]['height_robust_m']
r = vv.Report(); vv.check_v23_top_ranking(bad, r); assert not r['ok']
old = copy.deepcopy(bad); old['product_version'] = '2.2'
r = vv.Report(); vv.check_v23_top_ranking(old, r); assert r['ok'] and not r['checks']
print('ALL OK')
