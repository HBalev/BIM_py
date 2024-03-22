from rtree import index
import trimesh
import trimesh.collision
import uuid
import numpy as np

from base_classes.spatial_model import SpatialModel
from base_classes.base_classes import p, fill, export_json, Shape, Voxelized


class SceneModel(SpatialModel):
    pitch: float = 0.5

    def __init__(self, model):
        super(SceneModel, self).__init__(model)
        self.scene: trimesh.scene.scene.Scene = trimesh.scene.Scene()
        self.scene_flag = False
        self.path_scene_mesh_exported: str = ""
        self.mapped = False

    @fill
    def all_geometric_entities(self, iterator):
        print("get_geometric_entities")
        self.g_tree.add_element(iterator.get_native())
        ##################################################
        shape = iterator.get()
        current_guid = shape.guid
        geometry = Shape(current_guid, shape)
        mesh = geometry.mesh
        self.scene.add_geometry(mesh, current_guid)
        ###################################################
        rtree_bbx = tuple([*geometry.aabb[0], *geometry.aabb[1]])
        self.idx3d.insert(shape.id, rtree_bbx)

    @fill
    def trimesh_scene(self, iterator):
        shape = iterator.get()
        current_guid = shape.guid
        geometry = Shape(current_guid, shape)
        mesh = geometry.mesh
        self.scene.add_geometry(mesh, current_guid)

    def scene_filling(self):
        self.trimesh_scene()
        self.scene_flag = True

    def export_scene(self):
        print("export_scene")
        if not self.scene_flag:
            self.scene_filling()
        self.path_scene_mesh_exported = ".".join((self.model_name + "_scene_" + str(uuid.uuid4()), "stl"))
        ##############################################################
        rotate_around_y = np.array(
            [[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        self.scene.apply_transform(rotate_around_y).export(self.path_scene_mesh_exported)

    def get_voxel_encoding(self) -> np.array:
        if not self.scene_flag:
            self.scene_filling()
        self.export_scene()
        self.voxelized = Voxelized(self.path_scene_mesh_exported, self.pitch)
        self.voxelized.get_voxelized()
        self.encoding = self.voxelized.get_encoding()
        export_json(self.model_name + "_encoding_" + str(uuid.uuid4()) + ".json", self.encoding)
        return self.encoding

    def mapping_procedure(self):
        map_shape_to_vxl = {}
        map_vxl_to_shape = {}
        get_point_coordinates = lambda e: (e[0][0], e[0][1], e[0][2])
        get_index_coordinates = lambda e: (e[1][0], e[1][1], e[1][2])
        # print(vxl_scene.points_to_indices(np.array([0.0,0.0,0.0])))
        map_coord_indices = zip(
            self.voxelized.voxelized_scene.indices_to_points(self.voxelized.voxelized_scene.sparse_indices),
            self.voxelized.voxelized_scene.sparse_indices)
        for e in map_coord_indices:
            x, y, z = get_point_coordinates(e)
            bounds_vector = np.repeat(np.array([-self.pitch / 2, self.pitch / 2]), 3)
            coordinate_vector = np.tile(np.array([x, y, z]), 2)
            # x = e[0][0]
            # y = e[0][1]
            # z = e[0][2]
            # bbox = (x - self.pitch / 2, y - self.pitch / 2, z - self.pitch / 2, x + self.pitch / 2, y + self.pitch / 2,z + self.pitch / 2)
            bbox = bounds_vector + coordinate_vector
            elements = map(lambda x: self.model.by_id(x).GlobalId, list(self.idx3d.intersection(bbox)))
            # map_vxl_to_shape[(e[1][0], e[1][1], e[1][2])] = []
            map_vxl_to_shape[get_index_coordinates(e)] = []
            for el in elements:
                map_vxl_to_shape[get_index_coordinates(e)].append(el)
                if el not in map_shape_to_vxl.keys():
                    map_shape_to_vxl[el] = [get_index_coordinates(e)]
                else:
                    map_shape_to_vxl[el].append(get_index_coordinates(e))
        filename_shape_to_vxl = self.model_name + "_shape_to_vxl_" + str(uuid.uuid4()) + ".json"
        filename_vxl_to_shape = self.model_name + "_vxl_to_shape_" + str(uuid.uuid4()) + ".json"
        print(filename_vxl_to_shape)
        print(filename_shape_to_vxl)
        export_json((filename_shape_to_vxl, filename_vxl_to_shape,), (map_shape_to_vxl, map_vxl_to_shape))
        self.map_sh_v = map_shape_to_vxl
        self.map_v_sh = map_vxl_to_shape
        self.mapped = True

    def get_voxel_mapping(self):
        print("get_voxel_mapping")
        reqs = {"path_rtree_index": "rtree_filling", "scene_flag": "scene_filling",
                "path_scene_mesh_exported": "export_scene"}
        for flag in reqs.keys():
            if not getattr(self, flag):
                getattr(self, reqs[flag])()
        # if not (self.path_rtree_index and self.scene_flag and self.path_scene_mesh_exported):
        #     self.rtree_filling()
        #     self.scene_filling()
        #     self.export_scene()
        # elif self.path_rtree_index and self.scene_flag:
        #     self.export_scene()
        # elif self.scene_flag and self.path_scene_mesh_exported:
        #     self.rtree_filling()
        # elif self.path_rtree_index:
        #     self.scene_filling()
        #     self.export_scene()
        # elif self.scene_flag:
        #     self.rtree_filling()
        #     self.export_scene()
        # elif self.path_scene_mesh_exported:
        #     self.rtree_filling()
        print(8)
        # self.fill_rtree_spatial()
        # self.rtree_spatial()
        # self.trimesh_scene()
        # self.get_trimesh_scene()
        # self.export_scene()
        self.idx3d = index.Index(p.filename, properties=p)
        self.voxelized = Voxelized(self.path_scene_mesh_exported, self.pitch)
        self.voxelized.get_voxelized()
        self.mapping_procedure()
        # encoding = self.voxelized.get_encoding()
        # map_shape_to_vxl = {}
        # map_vxl_to_shape = {}
        # get_point_coordinates = lambda e: (e[0][0], e[0][1], e[0][2])
        # get_index_coordinates = lambda e: (e[1][0], e[1][1], e[1][2])
        # # print(vxl_scene.points_to_indices(np.array([0.0,0.0,0.0])))
        # map_coord_indices = zip(
        #     self.voxelized.voxelized_scene.indices_to_points(self.voxelized.voxelized_scene.sparse_indices),
        #     self.voxelized.voxelized_scene.sparse_indices)
        # for e in map_coord_indices:
        #     x, y, z = get_point_coordinates(e)
        #     bounds_vector = np.repeat(np.array([-self.pitch / 2, self.pitch / 2]), 3)
        #     coordinate_vector = np.tile(np.array([x, y, z]), 2)
        #     # x = e[0][0]
        #     # y = e[0][1]
        #     # z = e[0][2]
        #     # bbox = (x - self.pitch / 2, y - self.pitch / 2, z - self.pitch / 2, x + self.pitch / 2, y + self.pitch / 2,z + self.pitch / 2)
        #     bbox = bounds_vector + coordinate_vector
        #     elements = map(lambda x: self.model.by_id(x).GlobalId, list(self.idx3d.intersection(bbox)))
        #     # map_vxl_to_shape[(e[1][0], e[1][1], e[1][2])] = []
        #     map_vxl_to_shape[get_index_coordinates(e)] = []
        #     for el in elements:
        #         map_vxl_to_shape[get_index_coordinates(e)].append(el)
        #         if el not in map_shape_to_vxl.keys():
        #             map_shape_to_vxl[el] = [get_index_coordinates(e)]
        #         else:
        #             map_shape_to_vxl[el].append(get_index_coordinates(e))
        # filename_shape_to_vxl = "shape_to_vxl" + str(uuid.uuid4()) + ".json"
        # filename_vxl_to_shape = "vxl_to_shape" + str(uuid.uuid4()) + ".json"
        # print(filename_vxl_to_shape)
        # print(filename_shape_to_vxl)
        # export_json((filename_shape_to_vxl, filename_vxl_to_shape,), (map_shape_to_vxl, map_vxl_to_shape))
        # return (map_shape_to_vxl, map_vxl_to_shape)

    def get_building_envelope_model(self) -> set[str]:
        print("get_building_envelope_model")
        if not self.mapped:
            self.get_voxel_mapping()
        encoding = self.voxelized.get_encoding()
        shell = Voxelized.get_building_envelope(encoding)
        voxel_grid_shell = Voxelized.encoding_to_voxel_grid(shell)
        building_envelope = set()

        for idx in voxel_grid_shell.sparse_indices:
            # if (idx[0], idx[1], idx[2]) in map_vxl_shape.keys():
            if tuple(idx) in self.map_v_sh.keys():
                ls = self.map_v_sh[tuple(idx)]
                if ls:
                    for k in ls:
                        building_envelope.add(k)
        filename = "building_envelope_" + str(uuid.uuid4()) + ".json"
        print(filename)
        export_json(filename, building_envelope)
        return building_envelope
