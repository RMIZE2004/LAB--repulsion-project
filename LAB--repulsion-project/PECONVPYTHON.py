from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Optional
import time
#All data classes, combinations, math library, future annotations imported now"
#next is all future relevant conversion functions
def sph_to_cart(r, theta, phi)-> np.ndarray:
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])
def cart_to_sph(x, y, z)-> np.ndarray:
    r=np.sqrt(x**2 + y**2 + z**2)
    safe_r = r if r != 0 else 1.0
    theta=np.arccos(np.clip(z/safe_r, -1.0, 1.0))
    phi=np.arctan2(y, x)
    if phi < 0:
        phi += 2 * np.pi
    return np.array([r, theta, phi])
def rtp_to_xyz_chem(rtp: np.ndarray)-> np.ndarray:
    r, theta, phi = rtp
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=1)
def xyz_to_rtp_chem(xyz: np.ndarray)-> np.ndarray:
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    r=np.linalg.norm(xyz, axis=1)
    safe_r = np.where(r == 0, 1.0, r)
    theta=np.arccos(np.clip(z / safe_r, -1.0, 1.0))
    phi=np.arctan2(y, x)
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    return np.stack([r, theta, phi], axis=0)
 #Random angle setup (with lone pairs included, but only bonding pairs for now)
def rand_angles_chem(Z: int, n_lone: int=0, n_bond: int | None=None, seed:int | None=None) -> np.ndarray:
    rng=np.random.default_rng(seed)
    rnds=np.round(rng.random((2, Z)), 2)
 #calculate unique random values for phi and theta
    seen=set()
    for j in range(Z):
     while rnds[0,j] in seen:
         rnds[0,j]= np.round(rng.random(), 2)
     seen.add(rnds[0,j])
    c=np.zeros((2,Z),dtype=float)
    c[0,:]= 2*np.pi*rnds[0,:]
    c[1,:]= np.pi*rnds[1,:]
    if Z ==5:
        c[1,0]= np.pi/2; c[0,0]=0
        c[1,1:3]= np.pi/2.3+ 1*np.pi*rnds[1,1:3]
        c[0,1]= np.pi/4 + .16*np.pi*rng.random()
        c[1,3]=0.0
        c[1,4]=np.pi-.2*rng.random() 
    return c    
#Int_Angs to calculate all relevant angles between pairs
def cosine_angle_matrix_chem(rtp : np.ndarray)-> np.ndarray:
   theta=rtp[1,:]
   phi=rtp[2,:]
   st, ct =np.sin(theta), np.cos(theta)
   sp, cp =np.sin(phi), np.cos(phi)
   cos_dphi = np.outer(cp, cp) + np.outer(sp, sp)
   cosa = np.outer(ct, ct) + np.outer(st, st) *cos_dphi
   return np.clip(cosa, -1.0, 1.0)
   #this function will be used to calculate the forces used for optimization
def forces_coulomb(xyz: np.ndarray, charge: np.ndarray, extra: None = None) -> np.ndarray:
   Z=xyz.shape[0]
   q=np.ones(Z) if charge is None else np.asarray(charge)
   #difference vectors for all locations
   diff=xyz[:, None, :] -xyz[None,:,:]
   # norm of distance for each set of pairs
   rij=np.linalg.norm(diff, axis=2)
   #infinity for the diagnonal in order to calculate 0 force on itself
   np.fill_diagonal(rij, np.inf)
   #inverse radius in order to account for inverse square law and unit vector
   inv_r3=1.0/ np.clip(rij, 1e-15, None)**3
   #charge products (irrelevant in base geometry)
   qq = np.outer(q, q)
   F = np.sum(diff*inv_r3[:,:,None]*qq[:,:,None], axis=1)
   return F
def energy_coulomb(xyz: np.ndarray, charges : np.ndarray | None = None) -> float:
    Z= xyz.shape[0]
    q= np.ones(Z, dtype=float) if charges is None else np.asarray(charges, dtype=float)
    diff = xyz[:, None, :] - xyz[None, :, :]
    rij= np.linalg.norm(diff, axis=2)
    iu= np.triu_indices(Z, 1)
    return float(np.sum((np.outer(q,q)[iu] / np.clip(rij[iu], 1e-12, None))))
MIN_ENERGY_Z: dict[int, float]= {2: 0.500000, 3: float(np.sqrt(3.0)), 4: float(3*np.sqrt(3/2)), 5: 6.474691, 6: 9.985281}
def energy_per_Z(Z: int) -> float | None:
    return MIN_ENERGY_Z.get(Z, None)

@dataclass
class VSEPRdone:
    PE_values: np.ndarray
    xyz: np.ndarray
    rtp: np.ndarray
    done: int
    Z: int
    iters: int
    angles_deg: np.ndarray
    charges: np.ndarray
    attempts: int
    iters_total: int = 0
    backtracks_total: int = 0
    stop_reason: str=""
    last_step: float=0.0
def angle_matrix_deg(xyz: np.ndarray) -> np.ndarray:
    u = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    dot= np.clip(u @ u.T, -1.0, 1.0)
    return np.degrees(np.arccos(dot))    

def optimize_vsepr_on_sphere(Z:int, *, max_iter: int=5000, step_size: float= .05, stepdown_every: int=50, stepdown_factor: float= 0.90, tol_digits: int= 8, attempts: int=50, seed: int | None = None, charge: np.ndarray | None = None, 
                             energy_window: float = 1e-8, on_update: Optional[Callable[[dict], None]]=None, update_every: int =1, viewer_delay: float=.02,) -> VSEPRdone:
    rng = np.random.default_rng(seed)
    q= np.ones(Z) if charge is None else np.asarray(charge, dtype=float)
    target = energy_per_Z(Z)
    eps_target = energy_window
    best = None 
    total_iters=0
    total_backtracks=0
    for attempt in range(1,attempts+1):
        c = rand_angles_chem(Z, seed=int(rng.integers(0, 2**31 -1)))    
        theta = c[1, :].copy()
        phi = c[0,:].copy()
        theta[0]= np.pi / 2
        r= np.ones(Z)
        rtp = np.stack([r, theta, phi], axis=0)
        xyz= rtp_to_xyz_chem(rtp)
        prev_rtp= rtp.copy()
        new_rtp= rtp.copy()
        d= step_size
        step = d
        pe_trace: list[float]= []
        pe_prev = np.inf
        energy_tol= 1e-8
        g_prev = None
        x_prev = None
        last_step_used = 0.0
        polishing= False
        armijo_c= 1e-3
        ls_max_bts= 20
        max_polish_iters= 300
        polish_counter=0
        if on_update is not None:
            PE0= np.zeros((2,100), dtype=float)
            dot0 = np.clip(xyz @ xyz.T, -1.0, 1.0)
            ang0= np.degrees(np.arccos(dot0))
            rtp0 = xyz_to_rtp_chem(xyz)
            rtp0[2, :] = np.where(rtp0[2,:]<0, rtp0[2,:]+2*np.pi, rtp0[2,:])
            on_update({"rtp": rtp0, "Z":Z, "charges": q, "done_flag": 1, "PE_values": PE0, "angles": ang0, "target_energy": target})
            if on_update is not None: 
                PE0= np.zeros((2,100), dtype=float)
                pe0= energy_coulomb(xyz, q)
                PE0[0, 0] =pe0
                dot0= np.clip(xyz @ xyz.T, -1.0, 1.0)
                ang0= np.degrees(np.arccos(dot0))
                rtp0 = xyz_to_rtp_chem(xyz)
                rtp0[2, :] = np.where(rtp0[2, :]<0, rtp0[2, :]+ 2*np.pi, rtp0[2,:])
                on_update({"rtp":rtp0, "Z": Z, "charges":q, "done_flag":1, "PE_values": PE0, "angles": ang0, "target_energy": target})
                if viewer_delay and viewer_delay > 0:
                    time.sleep(viewer_delay)
        for it in range(1, max_iter+1): 
            if it % stepdown_every ==0:
                d*= stepdown_factor
                step = min(step, d)
            F= forces_coulomb(xyz, q)
            Fx_dot_x = np.sum(F*xyz, axis=1, keepdims= True)
            Ft = F - Fx_dot_x *xyz
            g = Ft[1:,:].copy()
            if g_prev is not None and x_prev is not None:
                s= (xyz[1:,:] - x_prev).reshape(-1)
                y= (g - g_prev).reshape(-1)
                sy = float(s @ y)
                ss = float(s @ s)
                if sy > 1e-16: 
                    bb= ss/ sy
                    step = float(np.clip(bb, 1e-12, 10*d))
                else:
                    step = d
            else:
                step = d             
            #gradient of change from the tangential forces
            #grad_norm= float(np.linalg.norm(Ft[1:,:]))
            #if grad_norm < 1e-10:
                # rtp = xyz_to_rtp_chem(xyz)
                # dot= np.clip(xyz @ xyz.T, -1.0, 1.0)
                #angles_deg = np.degrees(np.arccos(dot))
                # return VSEPRdone(PE_values=np.asarray(pe_trace), xyz=xyz, rtp=rtp, done=2, Z=Z, iter=it, angles_deg=angles_deg, charges=q)
            pe = energy_coulomb(xyz, q)
            accepted= False
        
            max_backtracks= ls_max_bts
            bt=0
            g2= float(np.sum(Ft[1:,:]**2))
            g2_eff = max(float(np.sum(g*g)), 1e-14)
            grad_norm_current= float(np.sqrt(g2_eff))
            c= armijo_c if Z != 2 else 1e-5
            if Z==2:
                cang = float(np.clip(np.dot(xyz[0], xyz[1]), -1.0, 1.0))
                ang = float(np.arccos(cang))
            while bt< max_backtracks:
                if target is not None and abs(pe-target)<= energy_window or (Z == 2 and (np.pi- ang)< 1e-3):
                    new_rtp = xyz_to_rtp_chem(xyz)
                    new_rtp[2,:] = np.where(new_rtp[2,:]<0, new_rtp[2,:]+2*np.pi, new_rtp[2,:])
                    dot = np.clip(xyz @ xyz.T, -1.0, 1.0)
                    angles_deg = np.degrees(np.arccos(dot))
                    total_iters += it
                    return VSEPRdone(PE_values=np.asarray(pe_trace), xyz=xyz, rtp=new_rtp, done=2, Z=Z, iters=it, angles_deg= angles_deg, charges=q, attempts=attempt, iters_total=total_iters, backtracks_total=total_backtracks)
                trial =xyz.copy()
                if Z ==2:
                    t= Ft[1,:]
                    n = float(np.linalg.norm(t))

                    if n < 1e-12:
                        gdir = -xyz[0]
                        t = gdir - float(np.dot(gdir, xyz[1])) * xyz[1]
                        n = float(np.linalg.norm(t))
                
                        if n < 1e-16:
                            t= np.zeros_like(xyz[1])
                            n = 0.0
                    if n > 0.0:
                            delta = np.clip(.25* (np.pi -ang), 1e-4, .2)
                            step_eff = delta / n
                            trial[1:,:]= xyz[1:, :] + step_eff*t
                    else:
                            trial[1:,:]= xyz[1:,:]
                else:
                        trial[1:,:] = xyz[1:,:] + (step) * Ft[1:,:]
                trial /= np.linalg.norm(trial, axis=1 , keepdims=True)
                pe_trial = energy_coulomb(trial, q)
                
                if pe_trial < pe- c* step * g2_eff:
                    xyz = trial
                    pe = pe_trial
                    accepted = True
                    last_step_used = step
                    break
                step *= .99
                bt += 1
            total_backtracks += bt
            if not accepted:
                if grad_norm_current < 1e-6:
                    d *= 0.95
                else:
                    d *= .99
                if d < 1e-15 and not polishing:
                    d=max(d, 1e-20)
                    polishing= True
                    step = d
                    armijo_c= 1e-3
                    ls_max_bts= 20
                    polish_counter=0
                    stall_counter=0
                    if it <=100:
                        pe_trace.append(pe)
                    continue
            
                if it <= 100:
                    pe_trace.append(pe)
                continue
            
          

            new_rtp= xyz_to_rtp_chem(xyz)
            new_rtp[2, :] = np.where(new_rtp[2, :]<0, new_rtp[2,:]+2 * np.pi, new_rtp[2,:])
            angles_stable= np.array_equal(np.round(prev_rtp[:,1:], tol_digits), np.round(new_rtp[:,1:], tol_digits),)
            F_now = forces_coulomb(xyz, q)
            Fx_dot_x_now= np.sum(F_now*xyz, axis=1, keepdims=True)
            Ft_now= F_now -Fx_dot_x_now * xyz
            
            grad_norm = float(np.linalg.norm(Ft_now[1:,:]))
            x_prev = xyz[1:,:].copy()
            g_prev = g.copy()
            if it<=100:
                pe_trace.append(pe)
            if on_update is not None and (it % update_every ==0):
                _PE = np.zeros((2,100), dtype=float)
                if pe_trace:
                    base = np.asarray(pe_trace, dtype=float)
                    n= min(base.size, 100)
                    _PE[0, :n] = base[:n]
                ang_live = np.degrees(np.arccos(np.clip(xyz @ xyz.T, -1.0, 1.0)))
                on_update({"rtp": new_rtp, "Z": Z, "charges":q, "done_flag":1,"PE_values": _PE, "angles": ang_live, "target_energy": target})
                if viewer_delay and viewer_delay > 0:
                    time.sleep(viewer_delay)
            grad_tol= 5e-4
            tiny_step_gate = 1e-6
            if (not polishing) and (grad_norm <= grad_tol):
                polishing= True
                d= min(d, max(1e-6*d, 1e-8))
                step=min(step, d)
                armijo_c= 0.0
                ls_max_bts= 50
                
                polish_counter=0
                stall_counter=0
                continue
            if polishing:
                polish_counter += 1
                tiny_step_tol= 1e-5
                tiny_dE_tol= max(1e-13, .1*energy_window)
                if (last_step_used <=tiny_step_tol) and (pe_prev - pe <= tiny_dE_tol):
                    stall_counter = (stall_counter + 1) if 'stall_counter' in locals() else 1
                else:
                    stall_counter=0
                if (stall_counter >= 25) or (polish_counter>= max_polish_iters):
                    dot=np.clip(xyz @ xyz.T, -1.0, 1.0)
                    angles_deg=np.degrees(np.arccos(dot))
                    total_iters += it
                    return VSEPRdone(
                        PE_values= np.asarray(pe_trace), xyz=xyz, rtp=new_rtp, done=2, stop_reason="polishing", Z=Z, iters=it, angles_deg=angles_deg, charges=q, attempts=attempt, iters_total=total_iters, backtracks_total=total_backtracks)
        
            if (target is not None) and (pe<= target+eps_target):
                dot=np.clip(xyz@xyz.T, -1.0, 1.0)
                angles_deg=np.degrees(np.arccos(dot))
                total_iters += it
                
    
                return VSEPRdone(
                    PE_values= np.asarray(pe_trace), xyz=xyz, rtp=new_rtp, done=2, stop_reason="abs(pe-target)", Z=Z, iters=it, angles_deg=angles_deg, charges=q, attempts=attempt, iters_total=total_iters, backtracks_total=total_backtracks)
            prev_rtp=new_rtp

            pe_prev=pe
        total_iters += it
        last_it= it
        pe_final= energy_coulomb(xyz, q)
        new_rtp= xyz_to_rtp_chem(xyz)
        new_rtp[2,:]= np.where(new_rtp[2,:]<0, new_rtp[2,:]+2*np.pi, new_rtp[2,:])
        if (on_update is not None) and (it % update_every == 0):
            _PE = np.zeros((2, 100), dtype= float)
            if pe_trace: 
                base = np.asarray(pe_trace, dtype=float)
                n= min(base.size, 100)
                _PE=np.zeros((2,100), dtype =float)
                if pe_trace:
                    base = np.asarray(pe_trace, dtype=float)
                    n= min(base.size, 100)
                    _PE[0, :n] = base[:n]
            dot_live = np.clip(xyz@ xyz.T, -1.0, 1.0)
            ang_live = np.degrees(np.arccos(dot_live))
            on_update({"rtp": new_rtp, "Z": Z, "charges": q, "done_flag": 1, "PE_values": _PE, "angles": ang_live, "target_energy": target})
            if viewer_delay and viewer_delay> 0:
                time.sleep(viewer_delay)
        if best is None or pe_final< energy_coulomb(best.xyz, q):
            dot = np.clip(xyz @xyz.T, -1.0, 1.0)
            angles_deg= np.degrees(np.arccos(dot))
            best= VSEPRdone(PE_values=np.asarray(pe_trace), xyz= xyz.copy(), rtp=new_rtp.copy(), done=0, stop_reason="fallback_best", Z=Z, iters= last_it, iters_total= total_iters, angles_deg=angles_deg, charges=q, attempts=attempt,)
        if target is not None and pe_final<= target+eps_target:
            dot = np.clip(xyz @ xyz.T, -1.0, 1.0)
            angles_deg= np.degrees(np.arccos(dot))
            return VSEPRdone(PE_values=np.asarray(pe_trace), xyz=xyz, rtp=new_rtp, done=2, Z=Z, iters=max_iter, angles_deg=angles_deg, charges=q, attempts=attempt, iters_total=total_iters, backtracks_total=total_backtracks) 
        
    return best


        
      
        

def print_angles(xyz: np.ndarray, labels= None) -> None:
    ang = angle_matrix_deg(xyz)
    Z=xyz.shape[0]
    if labels is None:
        labels= [str(i) for i in range(Z)]
    for i in range(Z):
        for j in range(i+1, Z):
            print(f"{labels[i]}-labels{j}: {ang[i, j]: .2f}")
import argparse
import matplotlib.pyplot as plt
from visualfigsphere import figsphere_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-Z", type=int, default=5, help="Number of electron groups")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--energy_window", type=float, default=1e-12)
    args, _ = parser.parse_known_args()
    if 'get_ipython' in globals():
        Z=2
        seed=42
        energy_window=1e-8
    else:
        Z=args.Z
        seed=args.seed
        energy_window=args.energy_window
    
    plt.ion()

    fig = plt.figure(figsize=(7.5, 7.0), facecolor = "white")
    ax_main = fig.add_subplot(111, projection ="3d")
    plt.show(block=False)
    fig.canvas.flush_events()
    show_moves = 30
    frames = {"n": 0}
    # initialize artists as an empty list so downstream code always receives a 1D sequence
    store = {"artists": None}
    def on_update(state: dict):
        if frames["n"] < show_moves:
            # figsphere_main may sometimes return empty/None artists which can cause
            # downstream calls (e.g. to set_box_aspect) to receive an invalid shape.
            # Protect against that by ensuring we always keep a list-like artists object.
        
            _, _, returned = figsphere_main(
                state["rtp"],
                state["Z"],
                charges=state["charges"],
                done=state["done_flag"],
                ax_main=ax_main,
                PE_values=state["PE_values"],
                angles=state["angles"],
                target_energy=state["target_energy"],
                artists=store["artists"],
            )
            store["artists"] = returned
            ax_main.figure.canvas.draw_idle()
            plt.pause(.001)
            
            
            
            frames["n"] += 1
            time.sleep(.05)
    sol=optimize_vsepr_on_sphere(Z, seed=42, energy_window=1e-8, on_update=on_update, update_every=1, viewer_delay=0.0, attempts= 1)
    angles_from_PECONVPYTHON = angle_matrix_deg(sol.xyz)
    PE_values_from_PECONVPYTHON = np.zeros((2,100), dtype=float)
    base= np.asarray(sol.PE_values, dtype=float).ravel() if sol.PE_values is not None else np.array([], dtype=float)
    n= min(base.size, 100)
    PE_values_from_PECONVPYTHON[0, :n]= base[:n]
    _, _, store["artists"] = figsphere_main(sol.rtp, sol.Z, charges=sol.charges, done=2, ax_main=ax_main, PE_values=PE_values_from_PECONVPYTHON, angles= angles_from_PECONVPYTHON, target_energy= energy_per_Z(Z), artists= store["artists"])
    plt.ioff()
    plt.show()
    pe_final = energy_coulomb(sol.xyz, sol.charges)
    print(f"Converged flag = {sol.done} in {sol.iters} iterations")
    target_E= energy_per_Z(Z)
    print(f"Final potential energy: {pe_final: .6f} (target≈{target_E if target_E is not None else 'N/A':.6f})")
    print(f"Stop reason:{sol.stop_reason}")
    print("\nCentral angles at the origin (degrees):")
    print_angles(sol.xyz)