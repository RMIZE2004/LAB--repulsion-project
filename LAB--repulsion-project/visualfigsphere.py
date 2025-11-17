from __future__ import annotations
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from scipy.spatial import ConvexHull
def _geometry_name(Z: int, n_lp: int) -> str:
    table = {2: {0: "Linear"}, 3: {0:"Trigonal Planar", 1: "Bent"}, 4: {0:"Tetrahedral", 1:  "Trigonal pyramidal", 2: "Bent"},5: {0: "Trigonal Bipyramidal", 1: "Seesaw", 2: "T-shaped", 3: "Linear"}, 6: {0: "Octahedral", 1: "Square pyramidal", 2:"square planar", 3: "T shaped", 4: "linear"}}
    return table.get(Z, {}).get(n_lp, "Unknown")
try:
   from scipy.spatial import ConvexHull
   _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
def rtp_to_xyz_chem(rtp):
    r, theta, phi = rtp
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=1)
def _set_axes_equal_3d(ax: plt.Axes) -> None:
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range= x_limits[1] - x_limits[0]
    y_range= y_limits[1] - y_limits[0]
    z_range= z_limits[1] - z_limits[0]
    max_range= max(x_range, y_range, z_range)
    x_middle= np.mean(x_limits)
    y_middle= np.mean(y_limits)
    z_middle= np.mean(z_limits)
    ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])
def figsphere_main(rtp:np.ndarray, Z: int, *, charges: np.ndarray | None = None, done: int=0, ax_main: plt.Axes | None = None, show_convex_hull: bool = True, view= (45, 45), clear_when_done_lt_2: bool = True, sphere_alpha: float =0.4, n_lone_pairs: int =0, PE_values: np.ndarray | None= None, angles: np.ndarray | None = None, target_energy: float | None = None, artists: dict | None= None,) -> Tuple[plt.Figure, plt.Axes, dict]:
    xyz = rtp_to_xyz_chem(rtp)

    if charges is None:
        charges = np.ones(Z, dtype=float)
    else:
        charges= np.asarray(charges, dtype=float)
    point_colors= np.where(charges <1.2, "red", "green")
    if artists is None:
        artists= {}
        
        if ax_main is None:
            fig = plt.figure(figsize=(7.5, 7.0), facecolor="white")
            ax_main = fig.add_subplot(111, projection='3d')
            ax_main.set_position([.05, .05, .68, .82])
        else:
            fig = ax_main.figure
        ax_main.set_box_aspect([1, 1, 1])
        ax_main.set_xticks([]); ax_main.set_yticks([]); ax_main.set_zticks([])
        
        ax_main.view_init(elev=view[1], azim=view[0])
        
        u = np.linspace(0, np.pi, 101)
        v= np.linspace(0, 2 * np.pi, 101)
        xx= np.outer(np.sin(u), np.cos(v))
        yy= np.outer(np.sin(u), np.sin(v))
        zz= np.outer(np.cos(u), np.ones_like(v))
        surf= ax_main.plot_surface(xx, yy, zz, color = "lightgray", alpha=sphere_alpha, shade= True, antialiased= True, zorder = 0, linewidth= 0)
        ax_main.set_xlim3d([-1.5, 1.5]); ax_main.set_ylim3d([-1.5, 1.5]); ax_main.set_zlim3d([-1.5, 1.5])
        ax_main.autoscale(enable=False)
        rays, pts, labels_txt = [], [], []
        for i in range(Z):
            (ln,)= ax_main.plot([0, xyz[i,0]], [0, xyz[i,1]], [0, xyz[i, 2]], color=point_colors[i], linewidth= 2.5, zorder= 3)
            (pt,) = ax_main.plot([xyz[i,0]], [xyz[i, 1]], [xyz[i,2]], "o", markersize= 12, color=point_colors[i], markeredgecolor="white", markeredgewidth=1.5, zorder=4)
            label = f"{i+1}" if charges[i] < 1.2 else f"Lone Pair{i+1-Z}"
            lbl= ax_main.text(xyz[i,0], xyz[i,1], xyz[i,2], label, color=point_colors[i], fontsize=10, zorder=4)
            rays.append(ln); pts.append(pt); labels_txt.append(lbl)

        hull_tri= None
        if show_convex_hull and _HAS_SCIPY and Z>=4:
        
            
            try:
                
                hull = ConvexHull(xyz)
                hull_tri= ax_main.plot_trisurf(xyz[:,0], xyz[:,1], xyz[:,2], triangles= hull.simplices, color='cyan', alpha= 0.3, edgecolor= 'gray', linewidth= 0.0, zorder=1)
             
            except Exception:
               hull_tri= None
           



        
       
      
        ax_pot = fig.add_axes([.80, .06, .17, .23])
        ax_pot.set_title("Potential energy", fontsize= 9)
        pot_base_line, = ax_pot.plot([], [], "o-", color="blue", label="Base")
        pot_mod_line, = ax_pot.plot([], [], "o-", color= "orange", label= "lone pairs added")
        pot_tgt_line= ax_pot.axhline(target_energy, color= 'black', linestyle = "--", label= "Target") if target_energy is not None else None
        base = PE_values[0, :] if PE_values.shape[0] >= 1 else np.array([])
        mod = PE_values[1, :] if PE_values.shape[0] >= 2 else np.array([])
        base = base[base != 0]
        mod = mod[mod !=0]
        x_base = np.arange(1, base.size +1) if base.size else np.array([])
        x_mod = np.arange(1, mod.size +1) if mod.size else np.array([])
        pot_base_line.set_data(x_base, base)
        pot_mod_line.set_data(x_mod, mod)
        nmax = max(x_base.size, x_mod.size, 100)
        ax_pot.set_xlim(0, nmax)
        ax_pot.relim()
        ax_pot.autoscale_view(scalex = False, scaley = True)
        handles, labels = [], []
        if base.size:
            handles.append(pot_base_line); labels.append("Base")
        if mod.size:
            handles.append(pot_mod_line); labels.append("LP")
        old_leg = ax_pot.get_legend()
        if old_leg:
            old_leg.remove()
        if handles:
            ax_pot.legend(handles, labels, loc ="lower center", bbox_to_anchor= (.5, -.35), ncol= len(handles), frameon= False)

        ax_text= fig.add_axes([.80, .32, .17, .16]); ax_text.axis("off")

        ax_tbl = fig.add_axes([.06, .72, .25, .24]) 
        ax_tbl.axis("off")
        rowlabels = [f"P{i+1}" for i in range(Z)]
        collabels = [f"P{i+1}" for i in range(Z)]
        if angles is not None:     
            if Z!= 4:
                A= np.round(angles, 0).astype(int).astype(str) 
            else:
                A= np.round(angles, 1).astype(str)
            
            tbl = ax_tbl.table(cellText=A, rowLabels=rowlabels, colLabels=collabels, loc='center')
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.scale(1.0, 1.1)
        else:
           
            rowlabels = [f"P{i+1}" for i in range(Z)]
            collabels = [f"P{i+1}" for i in range(Z)]
            tbl = ax_tbl.table(cellText = [[""]*Z]*Z, rowLabels=rowlabels, colLabels=collabels, loc = 'center')
            tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 1.1)
            artists["tbl"]= tbl

        
        geom = _geometry_name(Z, n_lone_pairs)if done in (2, 3) else ""
            
        geom_txt=ax_main.text2D(.02, .05, geom, transform= ax_main.transAxes, fontsize=20, color="blue", alpha=0.5)
       
        
        ax_main.set_xlim3d([-1.5, 1.5]); ax_main.set_ylim3d([-1.5,1.5]); ax_main.set_zlim3d([-1.5, 1.5])
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001)
        artists.update(dict(fig=fig, ax_main= ax_main, ax_pot=ax_pot, ax_text = ax_text, ax_tbl = ax_tbl, surf= surf, rays= rays, pts= pts, labels= labels_txt, hull=hull_tri if ('hull_tri' in locals()) else None, pot_base= pot_base_line, pot_mod=pot_mod_line, pot_tgt=pot_tgt_line, tbl=tbl, geom_txt= geom_txt, Z=Z, charges= charges.copy()))
        return fig, ax_main, artists
    
    else:
        fig= artists["fig"]; ax_main = artists["ax_main"]
        for i in range(Z):
            ln= artists["rays"][i]
            ln.set_data([0, xyz[i,0]], [0, xyz[i,1]])
            ln.set_3d_properties([0, xyz[i, 2]])
            pt = artists["pts"][i]
            pt.set_data([xyz[i,0]], [xyz[i,1]])
            pt.set_3d_properties([xyz[i,2]])
            lbl = artists["labels"][i]
            lbl.set_position((xyz[i,0], xyz[i,1]))
            lbl.set_3d_properties(xyz[i,2])
            need_restyle = ("charges" not in artists) or artists["charges"].shape != charges.shape or np.any(artists["charges"] != charges)
            if need_restyle:
                for i in range(Z):
                    color = "red" if charges[i] < 1.2 else "green"
                    artists["rays"][i].set_color(color)
                    artists["pts"][i].set_color(color)
                    artists["pts"][i].set_markeredgecolor("white")
                    artists["pts"][i].set_markeredgewidth(1.5)
                    label_txt = f"{i+1}" if charges[i] < 1.2 else f"Lone Pair {i+1-Z}"
                    artists["charges"] = charges.copy()
        if artists["hull"] is not None and show_convex_hull and _HAS_SCIPY and Z>=4:
            try: artists["hull"].remove()
            except Exception: pass
            try:
                hull = ConvexHull(xyz)
                artists["hull"]= ax_main.plot_trisurf(xyz[:,0], xyz[:, 1], xyz[:,2], triangles= hull.simplices, color= "cyan", alpha= 0.3, edgecolor= 'gray', linewidth= 0.0, zorder=1)
            except Exception:
                artists["hull"] = None
        
        base= PE_values[0,:] if PE_values.shape[0]>=1 else np.array([])
        mod = PE_values[1,:] if PE_values.shape[0] >=2 else np.array([])
        base= base[base != 0]
        x_base= np.arange(1, base.size+1) if base.size else np.array([])
        x_mod= np.arange(1, mod.size+1) if mod.size else np.array([])
        artists["pot_base"].set_data(x_base, base)
        artists["pot_mod"].set_data(x_mod, mod)
        artists["ax_pot"].set_xlim(0, max(100, base.size if base.size else 100))
        artists["ax_pot"].relim(); artists["ax_pot"].autoscale_view(scalex=False, scaley=True)
        if angles is not None:     
            if Z!= 4:
                A= np.round(angles, 0).astype(int).astype(str) 
            else:
                A= np.round(angles, 1).astype(str)
            ax_tbl = artists["ax_tbl"]
            ax_tbl.cla(); ax_tbl.axis("off")

            rowlabels = [f"P{i+1}" for i in range(Z)]
            collabels = [f"P{i+1}" for i in range(Z)]
            tbl = ax_tbl.table(cellText=A, rowLabels=rowlabels, colLabels=collabels, loc='center')
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.scale(1.0, 1.1)
            artists["tbl"]= tbl
        
        geom= _geometry_name(Z, n_lone_pairs)if done in (2, 3) else""
        artists["geom_txt"].set_text(geom)
        artists["geom_txt"].set_position((.02, .05))
            
        
        ax_main.set_xlim3d([-1.5, 1.5]); ax_main.set_ylim3d([-1.5,1.5]); ax_main.set_zlim3d([-1.5, 1.5])
        
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        return fig, ax_main, artists

        