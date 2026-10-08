import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from kenken.pipeline import solve_image, extract_structure, read_instance
from kenken.model import solve, solve_joint

def evaluate_sample(jf):
    gt = json.loads(jf.read_text('utf-8'))
    img_name = gt.get('image', {}).get('file')
    img_p = jf.parent / img_name if img_name else jf.with_suffix('.png')
    if not img_p.exists():
        img_p = jf.with_suffix('.jpg')
    if not img_p.exists():
        return None
        
    img = cv2.imread(str(img_p))
    if img is None:
        return None
        
    st = extract_structure(img)
    inst = read_instance(st)
    
    # 1. Comprobar si las lecturas reales están en candidates
    gt_cages = gt['cages']
    all_gt_in_cands = True
    all_gt_in_top1 = True
    cages_match_count = 0
    
    for i, c in enumerate(inst.cages):
        if i >= len(gt_cages):
            all_gt_in_cands = False
            all_gt_in_top1 = False
            break
        real = gt_cages[i]
        real_target, real_op = real['target'], real['op']
        
        # Top-1
        cands_i = inst.candidates.get(i) or [{}]
        top1 = cands_i[0]
        if top1.get('target') == real_target and top1.get('op') == real_op:
            cages_match_count += 1
        else:
            all_gt_in_top1 = False
            
        # Top-k
        cands = inst.candidates.get(i) or []
        in_cands = any(cand.get('target') == real_target and cand.get('op') == real_op for cand in cands)
        if not in_cands:
            all_gt_in_cands = False

    # 2. Resolver directo
    res_direct = solve(inst)
    
    # 3. Resolver con pipeline (auto)
    res_pipe = solve_image(img, method='auto')
    
    sol_matches_gt = False
    if res_pipe.solved and res_pipe.grid is not None:
        sol_matches_gt = np.array_equal(res_pipe.grid, gt.get('solution'))
        
    return {
        'file': img_p.name,
        'n': gt['n'],
        'num_cages': len(gt_cages),
        'all_gt_in_top1': all_gt_in_top1,
        'all_gt_in_cands': all_gt_in_cands,
        'direct_solved': res_direct.solved,
        'fallback_used': res_pipe.fallback_used,
        'pipe_solved': res_pipe.solved,
        'sol_matches_gt': sol_matches_gt,
    }

# Analizar 60 muestras variadas
samples = sorted(Path('dataset/synthetic').glob('*.json'))[:40] + \
          sorted(Path('dataset/synthetic_hard').glob('*.json'))[:20]

stats = [evaluate_sample(s) for s in samples]
stats = [s for s in stats if s is not None]

total = len(stats)
direct_ok = sum(1 for s in stats if not s['fallback_used'] and s['pipe_solved'] and s['sol_matches_gt'])
direct_divergent = sum(1 for s in stats if not s['fallback_used'] and s['pipe_solved'] and not s['sol_matches_gt'])
direct_infeasible = sum(1 for s in stats if not s['fallback_used'] and not s['pipe_solved'])

fallback_total = sum(1 for s in stats if s['fallback_used'])
fallback_recovered_gt = sum(1 for s in stats if s['fallback_used'] and s['sol_matches_gt'])
fallback_divergent = sum(1 for s in stats if s['fallback_used'] and not s['sol_matches_gt'])

# En los casos donde el fallback se activó, ¿estaba el GT en los candidatos?
fb_when_gt_in_cands = sum(1 for s in stats if s['fallback_used'] and s['all_gt_in_cands'])
fb_when_gt_in_cands_and_correct = sum(1 for s in stats if s['fallback_used'] and s['all_gt_in_cands'] and s['sol_matches_gt'])

fb_when_gt_NOT_in_cands = sum(1 for s in stats if s['fallback_used'] and not s['all_gt_in_cands'])
fb_when_gt_NOT_in_cands_divergent = sum(1 for s in stats if s['fallback_used'] and not s['all_gt_in_cands'] and not s['sol_matches_gt'])

print(f'=== ESTUDIO ESTADÍSTICO DE INFERENCIA CONJUNTA (MAP) ({total} Muestras) ===')
print(f'1. Casos de Lectura Directa (sin fallback): {total - fallback_total}/{total} ({(total - fallback_total)/total*100:.1f}%)')
print(f'   - Coincide con Ground Truth: {direct_ok}')
print(f'   - Divergente de Ground Truth: {direct_divergent}')
print()
print(f'2. Casos con Fallback a Inferencia Conjunta (MAP): {fallback_total}/{total} ({fallback_total/total*100:.1f}%)')
print(f'   - Recuperó el Ground Truth: {fallback_recovered_gt} ({fallback_recovered_gt/max(1, fallback_total)*100:.1f}%)')
print(f'   - Divergente del Ground Truth (solución alternativa): {fallback_divergent} ({fallback_divergent/max(1, fallback_total)*100:.1f}%)')
print()
print(f'3. Análisis Causal de Divergencia en Fallback:')
print(f'   - Cuando el Ground Truth SÍ estaba en el top-k: {fb_when_gt_in_cands} casos')
print(f'     -> Recuperó el Ground Truth: {fb_when_gt_in_cands_and_correct}/{max(1, fb_when_gt_in_cands)} ({fb_when_gt_in_cands_and_correct/max(1, fb_when_gt_in_cands)*100:.1f}%)')
print(f'   - Cuando el Ground Truth NO estaba en el top-k: {fb_when_gt_NOT_in_cands} casos')
print(f'     -> Generó solución alternativa válida (divergente): {fb_when_gt_NOT_in_cands_divergent}/{max(1, fb_when_gt_NOT_in_cands)} ({fb_when_gt_NOT_in_cands_divergent/max(1, fb_when_gt_NOT_in_cands)*100:.1f}%)')
