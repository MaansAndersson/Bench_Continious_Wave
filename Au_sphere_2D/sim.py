import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from meep import materials

# radii
r = 50e-3

sizex = 240e-3
sizey = 240e-3

size_2d = (sizex, sizey)
# pixels / um
resolution = 400

pixel_size = 1/resolution
plane_th = 1*r
# original thickness = 0.2*sizex
pml_th = 1; #0.040
fullx = sizex + 2*pml_th
fully = sizey + 2*pml_th
cell = mp.Vector3(fullx, fully, 0)

pml_layers = [mp.PML(thickness=pml_th, direction=mp.X), mp.PML(thickness=pml_th, direction=mp.Y)]

nonpml_vol = mp.Volume(mp.Vector3(), size=mp.Vector3(sizex, sizey))

wl = 0.633
cfreq = 1/wl

comp = mp.Ex

sources = [mp.Source(mp.ContinuousSource(frequency=cfreq, is_integrated=True),
                     size=mp.Vector3(fullx, 0, 0),
                     component=comp,
                     center=mp.Vector3(0, -0.5*sizey))]


geometry = [mp.Sphere(material=mp.Medium(epsilon=-11.685, D_conductivity=-2*np.pi*1.58*1.2672/11.685),
            center=mp.Vector3(0, 0),
            radius=r),
            ]
# Au permitivity at 0.633
#real part -11.685
#imaginary part 1.2672

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution,
                    #force_complex_fields=True,
                    Courant=0.3,
                    split_chunks_evenly=False,)

f = plt.figure(dpi=150)
sim.plot2D(ax = f.gca())
if mp.am_master():
    plt.show()

sim.init_sim()
sim.solve_wh(1e-5, 10000, 50, 1, mp.GMRES)

ex_norm =  np.real(sim.get_array(vol=nonpml_vol, component=mp.Ex))

ex_en = np.abs(ex_norm)**2

plt.figure()
plt.imshow(ex_en, cmap=cm.jet, interpolation='none')
plt.colorbar()

if mp.am_master():
    plt.show()
