import matplotlib.pyplot as plt
import ifcopenshell
import uuid
import numpy as np
import numba

from base_classes.base_classes import Voxelized, export_json
from base_classes.scene_model import SceneModel


#####################################################################################################################################################
def initialize_sim(encoding, mp_sh_vx, model):
    encoding_n = np.pad(encoding, pad_width=1, mode="constant", constant_values=False)
    mapping_props = {"IfcWall": {"a": 0.28 * 10 ** -6, "T": 10 + 273.15},
                     "IfcWallStandardCase": {"a": 0.28 * 10 ** -6, "T": 10 + 273.15},
                     "IfcSlab": {"a": 0.994 * 10 ** -6, "T": 10 + 273.15},
                     "IfcSite": {"a": 10 * 10 ** -6, "T": 5 + 273.15},
                     "IfcMember": {"a": 0.082 * 10 ** -6, "T": 10 + 273.15},
                     "IfcDoor": {"a": 0.082 * 10 ** -6, "T": 10 + 273.15},
                     "IfcBeam": {"a": 0.082 * 10 ** -6, "T": 10 + 273.15},
                     "IfcOpeningElement": {"a": 0.5 * 10 ** -6, "T": 10 + 273.15},
                     "IfcWindow": {"a": 0.5 * 10 ** -6, "T": 10 + 273.15}}
    a = np.zeros(encoding_n.shape)
    T = np.zeros(encoding_n.shape)
    T[T == 0] = 273.15
    a[a == 0] = 20.0 * 10 ** -6
    spaces = Voxelized.get_cavity(encoding_n)

    for el in mp_sh_vx.keys():
        # print(model.by_guid(el).is_a())
        for tp in mapping_props.keys():
            if model.by_guid(el).is_a(tp):
                for cube in mp_sh_vx[el]:
                    x, y, z = cube
                    a[x][y][z] = mapping_props[tp]["a"]
                    T[x][y][z] = mapping_props[tp]["T"]
        # if model.by_guid(el).is_a("IfcWall") or model.by_guid(el).is_a("IfcWallStandardCase"):
        #     for cube in mp_sh_vx[el]:
        #         a[cube[0]][cube[1]][cube[2]] = 0.28  # * 10 ** -6
        #         T[cube[0]][cube[1]][cube[2]] = 10 + 273.15
        #
        # elif model.by_guid(el).is_a("IfcSlab"):
        #     for cube in mp_sh_vx[el]:
        #         a[cube[0]][cube[1]][cube[2]] = 0.994  # * 10 ** -6
        #         T[cube[0]][cube[1]][cube[2]] = 10 + 273.15
        # elif model.by_guid(el).is_a("IfcSite"):
        #     for cube in mp_sh_vx[el]:
        #         a[cube[0]][cube[1]][cube[2]] = 10  # ** -6
        #         T[cube[0]][cube[1]][cube[2]] = 5 + 273.15
        #
        # elif model.by_guid(el).is_a("IfcMember") or model.by_guid(el).is_a("IfcDoor") or model.by_guid(el).is_a(
        #         "IfcBeam"):
        #     for cube in mp_sh_vx[el]:
        #         a[cube[0]][cube[1]][cube[2]] = 0.082  # * 10 ** -6
        #         T[cube[0]][cube[1]][cube[2]] = 10 + 273.15
        #
        # elif model.by_guid(el).is_a("IfcOpeningElement") or model.by_guid(el).is_a("IfcWindow"):
        #     for cube in mp_sh_vx[el]:
        #         a[cube[0]][cube[1]][cube[2]] = 0.5  # * 10 ** -6
        #         T[cube[0]][cube[1]][cube[2]] = 10 + 273.15
        #         # print(a[cube[0]][cube[1]][cube[2]])

    for cube in Voxelized.encoding_to_voxel_grid(spaces).sparse_indices:
        x, y, z = cube
        a[x][y][z] = 20 * 10 ** -6
        T[x][y][z] = 20 + 273.15
        # print(a[cube[0]][cube[1]][cube[2]])
    return a, T, encoding_n


@numba.jit("f8[:,:,:,](f8[:,:,:])", nopython=True, nogil=True)
def apply_boundary_conditions(ns):
    lz, ly, lx = ns.shape
    ns[:][:][lz - 1] = 273.15
    ns[:][ly - 1][:] = 273.15
    ns[lx - 1][:][:] = 273.15
    ns[:][:][0] = 273.15
    ns[:][0][:] = 273.15
    ns[0][:][:] = 273.15
    ns[20:125, 20:125, 24:55] = 273.15 + 70
    ns[120:130, 35:45, 25] = 273.15 + 70
    ns[120:130, 115:120, 25] = 273.15 + 70
    # ns[75][10][25] = 273.15 + 70
    # ns[25][11][25] = 273.15 + 70
    # ns[25][12][25] = 273.15 + 70
    # ns[25][13][25] = 273.15 + 70
    # ns[26][10][26] = 273.15 + 70
    # ns[26][11][26] = 273.15 + 70
    # ns[26][12][26] = 273.15 + 70
    # ns[26][13][26] = 273.15 + 70
    return ns


@numba.jit("f8[:,:,:,:](f8[:,:,:,:],f8[:,:,:],f8,f8,f8)", nopython=True, nogil=True)
def solve_heat(heatmap, a, dt, dx, times):
    cs = heatmap[0].copy()  # current state
    lz, ly, lx = cs.shape
    print(lz, ly, lx)
    cf = 0  # current frame
    for t in range(0, times):
        # ns = cs.copy()
        cs = heatmap[cf].copy()  # current state
        ns = cs.copy()
        for i in range(1, lx - 1):
            for j in range(1, ly - 1):
                for k in range(1, lz - 1):
                    ns[i][j][k] = cs[i][j][k] + a[i][j][k] * dt / dx ** 2 * (cs[i][j + 1][k] + cs[i][j - 1][k] + \
                                                                             cs[i + 1][j][k] + cs[i - 1][j][k] + \
                                                                             cs[i][j][k + 1] + cs[i][j][k - 1] - \
                                                                             6 * cs[i][j][k])
                    print(t, k, j, i, ns[i][j][k])
                    # file.write(str(t)+" "+str(k)+" "+str(j)+" "+str(i)+" " +str(ns[i][j][k]))
        ############boundary conditions
        ns = apply_boundary_conditions(ns)
        # ns[:][:][lz - 1] = 273.15
        # ns[:][ly - 1][:] = 273.15
        # ns[lx - 1][:][:] = 273.15
        # ns[:][:][0] = 273.15
        # ns[:][0][:] = 273.15
        # ns[0][:][:] = 273.15
        # ns[20:125][20:125][24:55] = 273.15 + 70
        # ns[25][35:45][25] = 273.15 + 70
        # ns[120:130][115:120][25] = 273.15 + 70
        # ns[:][:][lz - 1] = 273.15
        # ns[:][ly - 1][:] = 273.15
        # ns[lx - 1][:][:] = 273.15
        # ns[:][:][0] = 273.15
        # ns[:][0][:] = 273.15
        # ns[0][:][:] = 273.15
        # ns[4][1][3] = 273.15 + 50
        # ns[5][1][3] = 273.15 + 50
        # ns[6][1][3] = 273.15 + 50
        # ns[7][1][3] = 273.15 + 50
        cs = ns.copy()
        # if t % f == 0:
        cf = cf + 1
        heatmap[cf] = cs

    return heatmap


def visualise_sim(heat_frames, f):
    my_cmap = plt.get_cmap('inferno')
    for h in [10, 15, 20, 24, 25, 26, 30, 40]:
        fig, ax = plt.subplots(figsize=(10, 10))
        a = ax.contourf(heat_frames[f][h], 100, cmap=my_cmap,
                        vmin=0, vmax=70)
        cbar = fig.colorbar(a)
        cbar.set_label('Temp [$^\circ C$]', fontsize=15)
        ax.set_title('Height = {:.2f} '.format(h), fontsize=20)

        plt.show()


#####################################################################################################################################################
file_path = r"input_files/AC20-FZK-Haus.ifc"

model = ifcopenshell.open(file_path)
sm = SceneModel(model)
sm.pitch = 0.1
sm.get_voxel_mapping()
mp_sh_vx, map_vxl_to_shape = sm.map_sh_v, sm.map_v_sh
encoding = sm.get_voxel_encoding()
##################################################################
a, T, encoding_n = initialize_sim(encoding, mp_sh_vx, model)
##################################################################
times = 4
times_snapshot = 4

f = int(times / times_snapshot)
heat_frames = np.zeros([times_snapshot + 1, *encoding_n.shape])
heat_frames[0] = T
##################################################################
dt = 10
dx = 0.1
heat_frames = solve_heat(heat_frames, a, dt, dx, times)
heat_frames -= 273.15
print(heat_frames)
export_json(sm.model_name + "_heatmap_" + str(uuid.uuid4()) + ".json", heat_frames)
visualise_sim(heat_frames, f)
