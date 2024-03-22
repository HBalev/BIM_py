from rtree import index
import trimesh
import trimesh.collision
import uuid
import numpy as np

from base_classes.spatial_model import SpatialModel
from base_classes.base_classes import p, fill, export_json, Shape, Voxelized


class SceneModel(SpatialModel):
    '''
                Class SceneModel models the triangulated scene based on an ifc-model
                :param model - opened Ifc-model-entity
                :attr  num_objs - number of objects in the scene
                :attr  ptch - voxelization pitch of the model
                :attr  map_entity_bounds - dictionary with guid->aabb
                :attr  scene - Trimesh Scene-object to be filled with mesh representations of the objects
                :attr  idx3d - rtree-spatial-tree object
                :attr  g_tree - ifc-geometric-tree object
                :attr  path_scene_mesh_exported - string filename
                :method  all_geometric_entities - fill simultaniously self.idx3d, self.scene and self.g_tree
                :method  ifc_tree - fills self.g_tree with geometry
                :method  rtree_filling - fills the self.idx3d-rtree with aabb-s of the selected entities and updates the path(flag) of the .index-file
                :method  scene_filling - fills the self.scene with meshes and updates the flag
                :method  export_scene - exports the self.scene to a .stl
                :method  get_voxel_mapping - returns two dictionaries that map entity to a list of voxels that lie within the entity and the opposite mapping- voxel to entity
                :method  get_building_envelope_model - returns a list of GlobalIds of all entities that belong to the geometric building envelope
                '''
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
        map_coord_indices = zip(
            self.voxelized.voxelized_scene.indices_to_points(self.voxelized.voxelized_scene.sparse_indices),
            self.voxelized.voxelized_scene.sparse_indices)
        for e in map_coord_indices:
            x, y, z = get_point_coordinates(e)
            bounds_vector = np.repeat(np.array([-self.pitch / 2, self.pitch / 2]), 3)
            coordinate_vector = np.tile(np.array([x, y, z]), 2)
            bbox = bounds_vector + coordinate_vector
            elements = map(lambda x: self.model.by_id(x).GlobalId, list(self.idx3d.intersection(bbox)))
            map_vxl_to_shape[get_index_coordinates(e)] = []
            for el in elements:
                map_vxl_to_shape[get_index_coordinates(e)].append(el)
                if el not in map_shape_to_vxl.keys():
                    map_shape_to_vxl[el] = [get_index_coordinates(e)]
                else:
                    map_shape_to_vxl[el].append(get_index_coordinates(e))
        filename_shape_to_vxl = self.model_name + "_shape_to_vxl_" + str(uuid.uuid4()) + ".json"
        filename_vxl_to_shape = self.model_name + "_vxl_to_shape_" + str(uuid.uuid4()) + ".json"
        export_json((filename_shape_to_vxl, filename_vxl_to_shape,), (map_shape_to_vxl, map_vxl_to_shape))
        self.map_sh_v = map_shape_to_vxl
        self.map_v_sh = map_vxl_to_shape
        self.mapped = True

    def get_voxel_mapping(self):
        reqs = {"path_rtree_index": "rtree_filling", "scene_flag": "scene_filling",
                "path_scene_mesh_exported": "export_scene"}
        for flag in reqs.keys():
            if not getattr(self, flag):
                getattr(self, reqs[flag])()
        self.idx3d = index.Index(p.filename, properties=p)
        self.voxelized = Voxelized(self.path_scene_mesh_exported, self.pitch)
        self.voxelized.get_voxelized()
        self.mapping_procedure()
        
    def get_building_envelope_model(self) -> set[str]:
        if not self.mapped:
            self.get_voxel_mapping()
        encoding = self.voxelized.get_encoding()
        shell = Voxelized.get_building_envelope(encoding)
        voxel_grid_shell = Voxelized.encoding_to_voxel_grid(shell)
        building_envelope = set()

        for idx in voxel_grid_shell.sparse_indices:
            if tuple(idx) in self.map_v_sh.keys():
                ls = self.map_v_sh[tuple(idx)]
                if ls:
                    for k in ls:
                        building_envelope.add(k)
        filename = "building_envelope_" + str(uuid.uuid4()) + ".json"
        export_json(filename, building_envelope)
        return building_envelope
