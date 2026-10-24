"""Cross-check TIGRE (R2-Gaussian conventions) vs foj3d/projector.py.
Prints per-angle relative L2 difference and correlation of projections of the
engine GT and of an off-centre ball (catches axis/flip/orientation errors),
then CGLS-60 input PSNR on engine P50 for both backends."""
import json, sys

import numpy as np
import torch

sys.path.insert(0, '/content/3D-Field-of-Junctions')
import ct_pipeline, metrics
from strict_protocol import load_gt, get_angles

res = {}
gt = load_gt('engine')
N = 256
lin = np.linspace(-1, 1, N, dtype=np.float32)
z, y, x = np.meshgrid(lin, lin, lin, indexing='ij')
ball = (((x - 0.4) ** 2 + (y + 0.2) ** 2 + (z - 0.1) ** 2) <= 0.3 ** 2).astype(np.float32)

ang = np.array([0.0, 0.7, 2.1, 4.0])
for name, vol in (('engine', gt), ('ball_offcentre', ball)):
    import tigre, projector as pj
    pt = tigre.Ax(np.transpose(vol, (2, 1, 0)).copy(), ct_pipeline.tigre_geometry(256),
                  ang.astype(np.float32))[:, ::-1, :]
    v = torch.from_numpy(vol)[None, None].cuda()
    pp = pj.forward_project(v, ang, det_hw=256, det_half=2.0).cpu().numpy()
    rows = []
    for i in range(len(ang)):
        a, b = pt[i].ravel(), pp[i].ravel()
        rows.append(dict(angle=float(ang[i]), max_tigre=float(a.max()), max_torch=float(b.max()),
                         rel_l2=float(np.linalg.norm(a - b) / np.linalg.norm(a)),
                         corr=float(np.corrcoef(a, b)[0, 1]),
                         corr_transposed=float(np.corrcoef(a, pp[i].T.ravel())[0, 1]),
                         corr_flipped_v=float(np.corrcoef(a, pp[i][::-1].ravel())[0, 1])))
        print(name, rows[-1], flush=True)
    res[name] = rows

for backend in ('tigre', 'torch'):
    angs = get_angles('jitter')
    proj = ct_pipeline.simulate(gt, angs, 50.0, 0, backend=backend)
    rec = ct_pipeline.cgls(proj, angs, [15, 60], backend=backend)
    res[f'cgls_{backend}'] = {k: dict(psnr_unclamped=metrics.psnr(gt, v),
                                      psnr_clamped=metrics.psnr(gt, np.clip(v, 0, 1)))
                              for k, v in rec.items()}
    print(backend, res[f'cgls_{backend}'], flush=True)
json.dump(res, open('validate_tigre.json', 'w'), indent=2)
print('DONE')
