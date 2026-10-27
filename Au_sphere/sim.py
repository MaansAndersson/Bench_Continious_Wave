import meep as mp
import numpy as np
from meep.materials import Au

# computational cell: (3D)
sx = 0.20
sy = 0.20
sz = 0.20

dpml = 0.15

cell = mp.Vector3(sx+2*dpml,sy+2*dpml,sz+2*dpml)

# PML boundary conditions with dpml um all around
pml_layers = [mp.PML(dpml)]

# resolution: number of pixels per um
resolution = 100

# Courant number
Courant = 0.30

# symmetries
symmetries = [mp.Mirror(mp.Z)]

# setting frequency
fcen = 1/0.515

## geometry (adding AuNP sphere - default material set to water later)
r_sphere = 0.05

# material properties manually according to AuNP size and frequency of source:
# Au permitivity at 0.515 um (pulled from Johnson and Christy 1972)
# real part -3.5733
# imaginary part 2.9088

eps = -3.5733 + 2.9088*1j

geometry = [mp.Sphere(material=mp.Medium(epsilon=eps.real, D_conductivity=2*np.pi*fcen*eps.imag/eps.real),
            center=mp.Vector3(),
            radius=r_sphere),
            ]

# location source:
source_loc = mp.Vector3(-1*(r_sphere+0.01),0,0)
#
sources = [mp.Source(mp.ContinuousSource(frequency=fcen,is_integrated=True),
                    component=mp.Ex,
                    center=source_loc,
                    amplitude=1)]

#  setting up simulation

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution,
                    default_material=mp.Medium(index=1.333),
                    force_complex_fields=False, #True,
                    symmetries=symmetries,
                    Courant=Courant,
                    split_chunks_evenly=False)

sim.init_sim()
# MeepHoltz: solve_wh(tol, maxiters, L, periods, solver); res 100 needs ~450 GMRES iterations
sim.solve_wh(1e-7,10000,50)
#sim.solve_cw(1e-7,10000,2)  # needs force_complex_fields=True; breaks down for this eps < 0 sphere

## output for ParaView: one HDF5 file + an XDMF descriptor (open the .xdmf in ParaView)
import os
import h5py

outdir = "output"
os.makedirs(outdir, exist_ok=True)

# full cell (incl. PML) on MEEP's centered grid
vol = mp.Volume(center=mp.Vector3(), size=cell)
x, y, z, _ = sim.get_array_metadata(vol=vol)
data = {name: sim.get_array(component=c, vol=vol)
        for name, c in [("ex", mp.Ex), ("ey", mp.Ey), ("ez", mp.Ez),
                        ("hx", mp.Hx), ("hy", mp.Hy), ("hz", mp.Hz)]}
data["eps"] = np.real(sim.get_epsilon())
data["absE"] = np.sqrt(data["ex"]**2 + data["ey"]**2 + data["ez"]**2)

if mp.am_master():
    h5name = "au_sphere.h5"
    with h5py.File(os.path.join(outdir, h5name), "w") as f:
        for name, a in data.items():
            # arrays are (x,y,z) in C order; XDMF wants the slowest index first, so store as (z,y,x)
            f.create_dataset(name, data=np.ascontiguousarray(a.T))
        f.create_dataset("x", data=x); f.create_dataset("y", data=y); f.create_dataset("z", data=z)

    nx, ny, nz = len(x), len(y), len(z)
    attrs = "\n".join(
        f'      <Attribute Name="{name}" AttributeType="Scalar" Center="Node">\n'
        f'        <DataItem Dimensions="{nz} {ny} {nx}" NumberType="Float" Precision="8" Format="HDF">'
        f'{h5name}:/{name}</DataItem>\n'
        f'      </Attribute>'
        for name in data)
    with open(os.path.join(outdir, "au_sphere.xdmf"), "w") as f:
        f.write(f"""<?xml version="1.0" ?>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="au_sphere" GridType="Uniform">
      <Topology TopologyType="3DCoRectMesh" Dimensions="{nz} {ny} {nx}"/>
      <Geometry GeometryType="ORIGIN_DXDYDZ">
        <DataItem Dimensions="3" Format="XML">{z[0]} {y[0]} {x[0]}</DataItem>
        <DataItem Dimensions="3" Format="XML">{z[1]-z[0]} {y[1]-y[0]} {x[1]-x[0]}</DataItem>
      </Geometry>
{attrs}
    </Grid>
  </Domain>
</Xdmf>
""")
    print(f"wrote {outdir}/{h5name} and {outdir}/au_sphere.xdmf, grid {nx}x{ny}x{nz}")
