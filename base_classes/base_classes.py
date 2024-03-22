import networkx as nx
from rtree import index
import trimesh
import ifcopenshell.geom
import ifcopenshell.util.shape
import json
import ifcopenshell.util.element
import ifcopenshell.api
import ifcopenshell.util.system
import trimesh.collision
import multiprocessing
import ifcopenshell
import uuid
import numpy as np
from scipy import ndimage
import functools

settings = ifcopenshell.geom.settings()
settings.set(settings.USE_WORLD_COORDS, True)

p = index.Property()
p.dimension = 3
p.dat_extension = 'data'
p.idx_extension = 'index'


def fill(f):
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        print("Start filling")
        # print(list(self.selection))
        iterator = ifcopenshell.geom.iterator(settings, self.model, multiprocessing.cpu_count(),
                                              include=list(map(lambda x: self.model.by_guid(x), list(self.selection))))
        if iterator.initialize():
            while True:
                # print(iterator)
                f(self, iterator, *args, **kwargs)
                if not iterator.next():
                    break
        print("Filled")

    return wrapper


def export_json(filenames, datas):
    if isinstance(filenames, list) or isinstance(filenames, tuple):
        for pair in zip(filenames, datas):
            filename = pair[0]
            data = pair[1]
            print(filename)
            print(type(data))
            if isinstance(list(data.keys())[0], tuple):
                with open(filename, "w") as outfile:
                    json.dump({str(k): v for k, v in data.items()}, outfile, cls=NpEncoder)
            else:
                with open(filename, "w") as outfile:
                    json.dump(data, outfile, cls=NpEncoder)
            print("Export finished")

    else:
        with open(filenames, "w") as outfile:
            json.dump(datas, outfile, cls=NpEncoder)
        print("Export finished")


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, tuple):
            return str(obj)
        if isinstance(obj, set):
            return list(obj)
        return super(NpEncoder, self).default(obj)


class Shape:
    '''
    Class Shape models the accumulated geometric information of IFC-triangulation object
    :param guid - GlobalId of Ifc-Entity
    :param ifc_shape - Ifc-native entity-shape(geometry)
    :attr  ptch - pitch for voxelization
    :attr  verts - flattened list of vertices coordinates
    :attr  faces - flattened list of triangle faces based on indices of vertices
    :attr  matrix - transformation matrix (translation and rotation) of the entity shape
    :attr  grouped_verts - array of ordered coordinates of the triangulation vertices
    :attr  grouped_faces - array of ordered indices of vertices contained in triangle face
    :attr  mesh - Trimesh triangulation object of the shape(based on vertices and faces)
    :attr  aabb - axis aligned bounding box of the object in format (xmin,ymin,zmin,xmax,ymax,zmax)
    '''
    ptch = 0.1

    def __init__(self, guid, ifc_shape):
        assert isinstance(guid, str)
        assert isinstance(ifc_shape, ifcopenshell.ifcopenshell_wrapper.TriangulationElement)
        self.guid = guid
        self.ifc_shape = ifc_shape
        self.verts: list[float] = self.ifc_shape.geometry.verts
        self.faces: list[int] = self.ifc_shape.geometry.faces
        self.matrix = self.ifc_shape.transformation.matrix.data
        self.grouped_verts: list[list] = [[self.verts[i], self.verts[i + 1], self.verts[i + 2]] for i in
                                          range(0, len(self.verts), 3)]
        self.grouped_faces: list[list] = [[self.faces[i], self.faces[i + 1], self.faces[i + 2]] for i in
                                          range(0, len(self.faces), 3)]
        self.mesh: trimesh.base.Trimesh = trimesh.Trimesh(vertices=self.grouped_verts, faces=self.grouped_faces)
        # print(type(self.mesh))
        # self.voxelized = self.mesh.voxelized(pitch=Element.ptch)
        try:
            self.aabb: tuple[np.array, np.array] = ifcopenshell.util.shape.get_bbox(self.grouped_verts)
        except:
            self.aabb = (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))

    def __eq__(self, other):
        return self.guid == other.guid

    def get_voxelized(self):
        return self.mesh.voxelized(pitch=Shape.ptch)


class Element:
    '''
        Class Element models the accumulated geometric and semantic information of IFC entity
        :param guid - GlobalId of Ifc-Entity
        :param model - opened Ifc-model-entity
        :attr  entity - whole Ifc-entity from the Ifc-Model
        :attr  ifc_class - the ifc-class of the entity
        :attr  shape - ifc-native triangulation
        :attr  geometry - Shape-object based on the entity
        :attr  system - IfcSystem - entity
        '''

    def __init__(self, guid, model):
        assert isinstance(guid, str)
        assert isinstance(model, ifcopenshell.file)
        self.guid = guid
        self.model = model
        self.entity = self.model.by_guid(self.guid)
        self.ifc_class = self.entity.is_a()
        if self.entity.Representation:
            self.shape = ifcopenshell.geom.create_shape(settings, self.entity)
            # print(type(self.shape))
            self.geometry = Shape(self.guid, self.shape)
        if ifcopenshell.util.system.get_element_systems(self.entity):
            self.system = ifcopenshell.util.system.get_element_systems(self.entity)[0]
        else:
            self.system = None

    def __eq__(self, other) -> bool:
        return self.guid == other.guid

    def __str__(self) -> str:
        return "Entity with name '" + str(self.entity.Name) + "' of class '" + str(
            self.ifc_class) + "' part of system " + str(self.system.Name)


class Voxelized:
    '''
                    Class SceneModel models the triangulated scene based on an ifc-model
                    :param mesh_path - path of the mesh to be voxelized
                    :param pitch - voxelization pitch
                    :attr  voxelized_scene - voxelized scene
                    :method  export_binvox - exports the voxelized scene in binvox format
                    :method  get_encoding - returns the encoding of the voxelized scene
                    :staticmethod  get_encoding_graph - get a networkx Graph object with all neighboring voxels
                    :staticmethod  encoding_to_voxel_grid - transforms the encoding to a Trimesh.VoxelGrid object
                    :staticmethod  get_filled_encoding - returns the encoding with all the holes filled (voxel flood)
                    :staticmethod  get_building_envelope - returns the encoding to the most outer layer of voxels(the shell)
                    :staticmethod  get_cavity - returns the encoding that corresponds to the cavity inside a voxel scene
                    '''

    def __init__(self, mesh_path, pitch):  # ,model):
        assert isinstance(mesh_path, str)
        assert isinstance(pitch, float)
        self.mesh_path = mesh_path
        self.pitch = pitch

    # self.model=model

    def get_voxelized(self):  #######binvox doesn't work yet
        print("get_voxelized")
        mesh = trimesh.load_mesh(self.mesh_path)
        self.voxelized_scene = mesh.voxelized(self.pitch, method="binvox",
                                              binvox_path=r"C:\Users\hbale\Downloads\binvox.exe")
        print(dir(self.voxelized_scene))
        print(type(self.voxelized_scene))
        return self.voxelized_scene

    def export_binvox(self):
        print("export_binvox")
        if self.voxelized_scene:
            self.voxelized_scene.export(str(uuid.uuid4()), 'binvox')
        else:
            self.get_voxelized()
            self.voxelized_scene.export(str(uuid.uuid4()), 'binvox')

    def export_encoding(self) -> str:
        print("export_encoding")
        filename = "encoding_" + str(uuid.uuid4()) + ".json"
        print(filename)
        export_json(filename, self.encoding)
        return filename

    def get_encoding(self) -> np.array:
        print("get_encoding")
        if self.voxelized_scene:
            self.encoding = self.voxelized_scene.encoding.dense.tolist()
            return self.encoding
        else:
            self.get_voxelized()
            self.encoding = self.voxelized_scene.encoding.dense.tolist()
            return self.encoding

    @staticmethod
    def get_encoding_graph(encoding: np.array) -> nx.classes.graph.Graph:
        # def ngbr_indices(p, bnd):
        #     p_x, p_y, p_z = p
        #     p_u = (p_x, p_y, p_z + 1)
        #     p_d = (p_x, p_y, p_z - 1)
        #     p_f = (p_x, p_y + 1, p_z)
        #     p_b = (p_x, p_y - 1, p_z)
        #     p_r = (p_x + 1, p_y, p_z)
        #     p_l = (p_x - 1, p_y, p_z)
        #     ngb = [p_u, p_d, p_f, p_b, p_r, p_l]
        #     for n in ngb:
        #         if n[0] < 0 or n[0] > bnd[0] - 2 or n[1] < 0 or n[1] > bnd[1] - 2 or n[2] < 0 or n[2] > bnd[2] - 2:
        #             ngb.remove(n)
        #     return ngb
        graph = nx.Graph()
        lst_shp = (len(encoding), len(encoding[0]), len(encoding[0][0]))
        print(lst_shp)
        voxel_grid = Voxelized.encoding_to_voxel_grid(encoding)
        sparse_indices = voxel_grid.sparse_indices
        list_3d_in_tuple = lambda x: (x[0], x[1], x[2])

        for sparse_index in sparse_indices:
            # print(si)
            graph.add_node(list_3d_in_tuple(sparse_index))
            # print(graph.number_of_edges())
            act_voxel = Voxel(sparse_index, lst_shp)
            # si_ngb = ngbr_indices(si, lst_shp)
            for p in act_voxel.get_voxel_neighbours():  # si_ngb:
                # print(" ",p)
                try:
                    if encoding[p.x][p.y][p.z]:
                        graph.add_node(p)
                        graph.add_edge(list_3d_in_tuple(sparse_index), p)
                except:
                    print(sparse_index, " ", p)
        return graph

    @staticmethod
    def encoding_to_voxel_grid(encoding: np.array) -> trimesh.voxel.VoxelGrid:
        print("encoding_to_voxel_grid")
        return trimesh.voxel.VoxelGrid(np.array(encoding, dtype=bool))

    @staticmethod
    def get_filled_encoding(encoding: np.array) -> np.array:
        print("get_filled_encoding")
        return ndimage.binary_fill_holes(encoding)

    @staticmethod
    def get_building_envelope(encoding: np.array) -> np.array:
        print("get_building_envelope")
        return np.logical_xor(Voxelized.get_filled_encoding(encoding),
                              ndimage.binary_erosion(Voxelized.get_filled_encoding(encoding)))

    @staticmethod
    def get_cavity(encoding: np.array) -> np.array:
        print("get_cavity")
        return np.logical_xor(Voxelized.get_filled_encoding(encoding), encoding)


class Voxel:
    def __init__(self, coordinates, bounds):
        self.coordinates = coordinates
        self.bounds = bounds
        self.x = self.coordinates[0]
        self.y = self.coordinates[1]
        self.z = self.coordinates[2]
        self.bnd_x = self.bounds[0]
        self.bnd_y = self.bounds[1]
        self.bnd_z = self.bounds[2]
        self.neighbour_up = Voxel((self.x, self.y, self.z + 1), self.bounds)
        self.neighbour_down = Voxel((self.x, self.y, self.z - 1), self.bounds)
        self.neighbour_front = Voxel((self.x, self.y + 1, self.z), self.bounds)
        self.neighbour_back = Voxel((self.x, self.y - 1, self.z), self.bounds)
        self.neighbour_right = Voxel((self.x + 1, self.y, self.z), self.bounds)
        self.neighbour_left = Voxel((self.x - 1, self.y, self.z), self.bounds)

    def get_voxel_neighbours(self):
        for n in (
                self.neighbour_up, self.neighbour_down, self.neighbour_front, self.neighbour_back, self.neighbour_right,
                self.neighbour_left):
            if Voxel.valid_voxel(n):
                yield n

    @staticmethod
    def valid_voxel(voxel):
        return (0 < voxel.x < voxel.bnd_x - 1) or (0 < voxel.y < voxel.bnd_y - 1) or (
                0 < voxel.z < voxel.bnd_z - 1)  # not(voxel.x < 0 or voxel.x > voxel.bnd_x - 2 or voxel.y < 0 or voxel.y > voxel.bnd_y - 2 or voxel.z < 0 or voxel.z > voxel.bnd_z - 2)

    def get_point(self, encoding):
        voxel_grid = Voxelized.encoding_to_voxel_grid(encoding)
        return voxel_grid.indices_to_points(self.coordinates)
