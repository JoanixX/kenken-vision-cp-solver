import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from kenken.pipeline import extract_structure, read_instance, solve_image
from kenken.model import solve_joint

gt = json.load(open('dataset/synthetic/syn_00002.json'))
print('--- syn_00002.jpg Analysis ---')
print('n =', gt['n'])
print('GT solution:\n', gt['solution'])

img = cv2.imread('dataset/synthetic/syn_00002.jpg')
st = extract_structure(img)
inst = read_instance(st)

gt_cages = gt['cages']
print(f'GT jaulas count: {len(gt_cages)}, Pred jaulas count: {len(inst.cages)}')

res = solve_image('dataset/synthetic/syn_00002.jpg', method='auto')
print('\nSolver fallback used:', res.fallback_used)
print('Pred solution:\n', res.grid)

res_joint = solve_joint(inst)
print('\nChosen candidates en joint:')
for c_idx, chosen in res_joint.chosen_candidates.items():
    real_c = gt_cages[c_idx] if c_idx < len(gt_cages) else None
    gt_t = real_c['target'] if real_c else '?'
    gt_op = real_c['op'] if real_c else '?'
    top1 = inst.candidates[c_idx][0]
    c_t = chosen['target']
    c_op = chosen['op']
    c_lp = chosen['logp']
    t1_t = top1['target']
    t1_op = top1['op']
    match_gt = (c_t == gt_t and c_op == gt_op)
    print(f'Jaula {c_idx:2d}: cells={inst.cages[c_idx].cells}')
    print(f'   GT: {gt_t}{gt_op} | Elegido: {c_t}{c_op} (logp={c_lp:.2f}) | Top1: {t1_t}{t1_op} | Match GT: {match_gt}')
