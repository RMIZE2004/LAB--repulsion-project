#all helpers below
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, CheckButtons


import PECONVPYTHON as conv
from visualfigsphere import figsphere_main


def _angle_matrix_deg(xyz: np.ndarray) -> np.ndarray:
    u = xyz / np.linalg.norm(xyz, axis= 1, keepdims=True)
    dot = np.clip( u @u.T, -1.0, 1.0 )
    return np.degrees(np.arccos(dot))
def _target_for_Z(Z: int) -> float | None:

    return { 2: .500000, 3 : float(np.sqrt(3.0)), 4: float(3* np.sqrt(3/2)), 5: 6.474691, 6: 9.985281}.get(Z, None)

class VSEPR_UI:
    def _angles_from_rtp(self, rtp3xZ: np.ndarray) -> np.ndarray:
        phi = rtp3xZ[2, :].copy()
        theta = rtp3xZ[1, :].copy()
        return np.vstack([phi, theta])
    def __init__(self, z0: int =4, seed: int =42, energy_window: float= 1e-8, show_moves: int =30):
        
        self.Z= z0
        self.lp_mask = np.zeros(self.Z, dtype=bool)
        self.lp_factor = 1.3
        self.seed= seed
        self.energy_window= energy_window
        self.show_moves= show_moves
        self._frames= 0
        self._base_seed_rtp: dict[int, np.ndarray] ={}
        self.fig = plt.figure(figsize=(8.5, 7.0), facecolor="white")
        self.ax_main = self.fig.add_axes([0.05, 0.05, 0.68, 0.9], projection='3d')
        self.ax_main.set_box_aspect([1, 1, 1])
        for ax in (self.ax_main,):
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            self.ax_main.set_xlim3d([-1.5, 1.5]); self.ax_main.set_ylim3d([-1.5, 1.5]); self.ax_main.set_zlim3d([-1.5, 1.5])
            self.ax_main.autoscale(enable= False)
            self.ax_main.view_init(elev=45, azim=45)

            self.artists = None

            panel_left = .78
            self.ax_title = self.fig.add_axes([panel_left, 0.92, 0.2, 0.05]); self.ax_title.axis('off')
            self.ax_title.text(0, 0.6, "VSEPR UI", fontsize =12)
            
            
            self.ax_z= self.fig.add_axes([panel_left, 0.78, 0.20, 0.15])
            self.rbuttons = RadioButtons(self.ax_z, labels = ["2", "3", "4","5", "6"],active=["2", "3", "4", "5", "6"].index(str(z0)))
            self.rbuttons.on_clicked(self._on_select_Z)
            self._z_widget = self.rbuttons
            
                
            self.ax_solve = self.fig.add_axes([panel_left, 0.60, 0.10, 0.05])
            self.btn_solve = Button(self.ax_solve, "Solve")
            self.btn_solve.on_clicked(lambda _evt: self.solve_and_draw(self.Z))

            self.ax_spin = self.fig.add_axes([panel_left+.11, .60, .09, .05])
            self.btn_spin = Button(self.ax_spin, "Spin On/Off")
            self.btn_spin.on_clicked(self._toggle_spin)
            self.spinning= False
            self._spin_timer = None
            self._azim = 45.0

            self.ax_msg = self.fig.add_axes([panel_left, .05, .20, .50])
            self._lp_axes_rect = [.56, .86, .20, .12]
            self.ax_lp = self.fig.add_axes(self._lp_axes_rect)
            self.ax_lp.set_title("Lone Pairs", fontsize= 9)
            lp_labels = [f"P{i+1}" for i in range(self.Z)]
            self.chk_lp = CheckButtons(self.ax_lp, labels = lp_labels, actives= [False]*self.Z)
            self.chk_lp.on_clicked(self._on_toggle_lp)
            
            self.ax_apply_lp = self.fig.add_axes([panel_left, 0.47, 0.20, 0.05])
            self.btn_apply_lp = Button(self.ax_apply_lp, "Apply LPs and Solve")
            self.btn_apply_lp.on_clicked(lambda _evt: self.solve_and_draw(self.Z))
            self._log(f"Ready. Current Z = {self.Z}. Click Solve.")
            self._pe_mem: dict[int, dict[str, np.ndarray]] ={}
            plt.show(block=False)
    def _on_select_Z(self, selection):
        
        self.Z= int(selection)
        
        self._pe_mem[self.Z] = {"base": np.array([], dtype=float), "lp": np.array([], dtype= float)}
        self.lp_mask = np.zeros(self.Z, dtype= bool)
        self._setup_lp_checkboxes()
        self._log(f"Selected Z = {self.Z}. Click Solve to re-run")
    
       
    def _setup_lp_checkboxes(self):
        for attr in ("ax_lp", "ax_apply_lp"):
            ax = getattr(self, attr, None)
            if ax is not None:
                try:
                    self.fig.delaxes(ax)
                except Exception:
                    try:
                        ax.remove()
                    except Exception:
                        pass
            
        self.ax_lp = self.fig.add_axes(self._lp_axes_rect)
        self.ax_lp.set_title("Lone Pairs", fontsize= 9)
        lp_labels = [f"P{i+1}" for i in range(self.Z)]
        self.chk_lp = CheckButtons(self.ax_lp, labels = lp_labels, actives= [False]*self.Z)
        self.chk_lp.on_clicked(self._on_toggle_lp)
        panel_left = .78 
        self.ax_apply_lp = self.fig.add_axes([panel_left, 0.47, 0.20, 0.05])
        self.btn_apply_lp = Button(self.ax_apply_lp, "Apply LPs and Solve")  
        self.btn_apply_lp.on_clicked(lambda _evt: self.solve_and_draw(self.Z))
        self.fig.canvas.draw_idle()
    def _on_toggle_lp(self, label):
        idx = int(label[1:]) -1
        if 0<= idx < self.Z:
            self.lp_mask[idx] = ~ self.lp_mask[idx]
    def _toggle_spin(self, _evt):
        self.spinning = not self.spinning
        if self.spinning:
            self._start_spin()
            self._log("Spin: ON")
        else:
            self._stop_spin()
            self._log("Spin: OFF")
    def _start_spin(self):
        if self._spin_timer is None:
            self._spin_timer = self.fig.canvas.new_timer(interval= 50)
            self._spin_timer.add_callback(self._spin_tick)
        self._spin_timer.start()
    def _stop_spin(self):
        if self._spin_timer is not None:
            self._spin_timer.stop()
    def _spin_tick(self):  
        self._azim = (self._azim + 0.6) % 360.0
        self.ax_main.view_init(elev =45, azim= self._azim)
        self.fig.canvas.draw_idle()
    def _live_on_update(self, state:dict):
        if self._frames < self.show_moves:
            n_lp = int(self.lp_mask.sum())
            trace = np.asarray(state.get("PE_values", []), dtype=float).ravel()
            N= 100
            PE_values = np.zeros((2,N), dtype=float)
            if trace.size:
                n = min(trace.size, N)
                if n_lp:

                    base_mem = self._pe_mem.get(self.Z, {}).get("base", np.array([], dtype=float))
                    if base_mem.size:
                        m = min(base_mem.size, N)
                        PE_values[0, :m] = base_mem[:m]
                        if m < N:
                            PE_values[1, m:] = trace[n-1]
                    PE_values[1, :n] = trace[:n]
                    if n < N:
                        PE_values[1, n:] = trace[n-1]
                else:
                    PE_values[0, :n] = trace[:n]
                    if n < N:
                        PE_values[0, n:] = trace[n-1]        
            
            
            target_E = _target_for_Z(state["Z"])
            _, _, self.artists = figsphere_main(state["rtp"], state["Z"], charges= state["charges"], done= state["done_flag"], ax_main= self.ax_main, n_lone_pairs=n_lp, PE_values= PE_values, angles = state.get("angles", None), target_energy= target_E, artists= self.artists,)
            self._update_potential_axis()
            self.fig.canvas.draw_idle()
            plt.pause(0.001)
            self._frames +=1
    def solve_and_draw(self, Z: int):
        self._stop_spin()
        self._log(f"Solving for Z = {Z}...")
        self._frames = 0
        
        if self.artists is not None and self.artists.get("Z", Z) != Z:
            for k in ("ax_pot", "ax_tbl", "ax_text"):
                ax = self.artists.get(k)
                if ax is not None:
                    try:
                        ax.remove()
                    except Exception:
                        try:
                            self.fig.delaxes(ax)
                        except Exception:
                            pass
            self.artists = None
        if self.artists is None:
            self.ax_main.cla()
            self.ax_main.set_box_aspect([1, 1, 1])
            self.ax_main.set_xticks([]); self.ax_main.set_yticks([]); self.ax_main.set_zticks([])
            self.ax_main.set_xlim3d([-1.5, 1.5]); self.ax_main.set_ylim3d([-1.5, 1.5]); self.ax_main.set_zlim3d([-1.5, 1.5])
            self.ax_main.autoscale(enable= False)
            self.ax_main.view_init(elev= 45, azim = self._azim)
        
        
        
        charges = np.ones(self.Z, dtype=float)
        charges[self.lp_mask] = 1.3
        
        want_lp = bool(np.any(self.lp_mask))
        try:
            if not want_lp:
                sol = conv.optimize_vsepr_on_sphere(Z, seed=self.seed, energy_window=self.energy_window, on_update=self._live_on_update, update_every=1, viewer_delay= 0.0, attempts =1, charge= charges)
                self._base_seed_rtp[Z] = sol.rtp.copy()
            else:
                seed_rtp = self._base_seed_rtp.get(Z, None)
                if seed_rtp is not None:
                    orig_rand = conv.rand_angles_chem
                    def _seed_rand_angles(_Z, *args, **kwargs):
                        if _Z == Z:
                            return self._angles_from_rtp(seed_rtp)
                        return orig_rand(_Z, *args, **kwargs)
                    conv.rand_angles_chem = _seed_rand_angles                  
                    try:
                        sol = conv.optimize_vsepr_on_sphere(Z, seed=self.seed, energy_window=self.energy_window, on_update=self._live_on_update, update_every=1, viewer_delay= 0.0, attempts =1, charge= charges)
                    finally:
                        conv.rand_angles_chem = orig_rand
                else:
                    sol = conv.optimize_vsepr_on_sphere(Z, seed=self.seed, energy_window=self.energy_window, on_update=self._live_on_update, update_every=1, viewer_delay= 0.0, attempts =1, charge= charges)
        except Exception as e:
            self._log(f"Solve failed: {e}")
            sol = conv.optimize_vsepr_on_sphere(Z, seed=self.seed, energy_window=self.energy_window, on_update=None, update_every=1, viewer_delay= 0.0, attempts =1, charge= charges)
        angles = _angle_matrix_deg(sol.xyz)
        trace = np.asarray(sol.PE_values, dtype=float).ravel() if sol.PE_values is not None else np.array([], dtype=float)
        want_lp = bool(np.any(self.lp_mask))
        mem = self._pe_mem.setdefault(self.Z, {"base": np.array([], dtype = float), "lp": np.array([], dtype= float)})
        if want_lp:
            mem["lp"] = trace.copy()
        else:
            mem["base"] = trace.copy()
            mem["lp"] = mem.get("lp", np.array([], dtype = float))
        PE_values = np.zeros((2,100), dtype =float)
        if mem["base"].size:
            n0 = min(mem["base"].size, 100); PE_values[0, :n0] = mem["base"][:n0]
            if n0<100:
                PE_values[0, n0:] = mem["base"][n0-1]
        if mem["lp"].size:
            n1 = min(mem["lp"].size, 100); PE_values[1, :n1] = mem["lp"][:n1]
            if n1<100:
                PE_values[1, n1:] = mem["lp"][n1-1]

        target_E = _target_for_Z(Z)
        n_lp = int(self.lp_mask.sum())
        _, _, self.artists = figsphere_main(sol.rtp, sol.Z, charges= sol.charges, done=2, ax_main= self.ax_main, n_lone_pairs= n_lp, PE_values= PE_values, angles= angles, target_energy= target_E, artists=self.artists)
        self._update_potential_axis()
        self.fig.canvas.draw_idle()
        if sol.done >=2 :
            self.spinning = True
            self._start_spin()
            self._log(f"Converged in {sol.iters} iterations. Spinning...")
        else:
            self._log(f"Done flag = {sol.done}. No spin (not fully converged)")
    
            
    def _spin_tick(self):
        if not plt.fignum_exists(self.fig.number):
            self._stop_spin()
            return
        self._azim = (self._azim + 0.6) % 360.0
        self.ax_main.view_init(elev =45, azim= self._azim)
        self.fig.canvas.draw_idle()
    def _update_potential_axis(self):
        if not self.artists:
            return
        ax = self.artists.get("ax_pot")
        if ax is None:
            return
        all_ys = []
        for line in ax.lines:
            line.set_linestyle("None")
            line.set_marker("o")
            line.set_markersize(4)
           
        target_E = _target_for_Z(self.Z)
        ymin= target_E-.05
        span = abs(ymin)*1.2 if ymin !=0.0 else 1.5
        ymax = ymin + span
        ax.set_ylim(ymin, ymax)
        ax.figure.canvas.draw_idle()
    def _log(self, text: str):
        self.ax_msg.cla(); self.ax_msg.axis("off")
        self.ax_msg.text(0.0, 1.0, text, fontsize= 10, va = "top")
        self.fig.canvas.draw_idle()
if __name__ == "__main__":
    import matplotlib
        
    ui = VSEPR_UI(z0 =4, seed =42, energy_window = 1e-8, show_moves=30)
    plt.show()
                

