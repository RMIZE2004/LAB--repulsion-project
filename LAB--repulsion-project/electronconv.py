from __future__ import annotations
import numpy as np
from typing import Optional

from PECONVPYTHON import (rtp_to_xyz_chem, xyz_to_rtp_chem, angle_matrix_deg, forces_coulomb,)

def electronconv(new_arr: np.ndarray, charge: np.ndarray, N2: int, rthphi: np.ndarray, done: int,  Z: int, l:int, angles: np.ndarray, *, step: float = .3, maxit: int=600, tol_digits: int =8, live_plot: bool=True, target_energy: Optional[float]= None,):
    q= np.array(charge, dtype=float).copy()
    PE_values = np.zeros((2, 100), dtype=float)
def pair_energy(xyz_local: np.ndarray, q_local: np.ndarray) -> float:
        q_eff = q_local.copy()
        q_eff[q_eff ==.5] =1.0
        q_eff[q_eff ==1.2]= 1.5
        diff = xyz_local[:, None, :] - xyz_local[None, :, :]
        rij = np.linalg.norm(diff, axis=2)
        iu =np.triu_indices(Z, 1)
        return float(np.sum(np.outer(q_eff, q_eff)[iu] / np.clip(rij[iu], 1e-12, None)))
xyz = rtp_to_xyz_chem(rthphi)
angles = angle_matrix_deg(xyz)
done=1
for it in range(1, maxit+1):
      if it % 600 ==0:
            step *= 0.9
      angles = angle_matrix_deg(xyz)
      col_sum= np.sum(angles, axis=0)
      mask = (col_sum> 430.0) & (q<1.2)
      q[mask]=0.5
      F= forces_coulomb(xyz, q)
      if Z> 1:
            trial = xyz.copy()
            step_part = xyz[1:,:] + step *F[1:,:]
            step_part /= np.linalg.norm(step_part, axis=1, keepdims=True)
            trial[1:,:] = step_part
      else:
            trial= xyz
      rtp_new = xyz_to_rtp_chem(trial)
      rtp_new[2, :] = np.where(rtp_new[2,:]<0, rtp_new[2,:]+ 2*np.pi, rtp_new[2,:])
      lhs = np.round(rthphi[:,1:], tol_digits)
      rhs = np.round(rtp_new[1:, :], tol_digits)
      if np.array_equal(lhs, rhs):
           xyz= trial
           rthphi = rtp_new
           l = it
           break
      xyz = trial
      rthphi = rtp_new
      l = it
      PETOT = pair_energy(xyz, q)
      if it < 100:
            PE_values[0, it-1] = PETOT
      done = 3
PETOT = pair_energy(xyz, q)
if it <= 100:
        PE_values[0, it-1] = PETOT
else:
        PE_values[1, -1] = PETOT
if live_lot and _HAS_PLOT:
     figsphere_main(rthphi, Z, charges=q, done=done, l=l, angles=angle_matrix_deg(xyz), PE_values=PE_values, target_energy=target_energy)
return PE_values, done, rthphi, Z, l, angle_matrix_deg(xyz), q
        