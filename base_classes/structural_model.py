import networkx as nx
from rtree import index
import trimesh
import trimesh.collision
import multiprocessing
import ifcopenshell
import uuid

from base_classes.spatial_model import SpatialModel
from base_classes.base_classes import p, Element, export_json


class StructuralModel(SpatialModel):
    '''
    Base class for all classes that process geometric data and contain the spatial relationships between the entities as graph, that can be processed
    '''
    def __init__(self, model, selection):
        super(StructuralModel, self).__init__(model, selection)
        assert isinstance(selection, list)
        self.collision_flag = False
        self.entities: dict = {}
        self.graph: nx.classes.graph.Graph = nx.Graph()
        self.graph.add_nodes_from(self.selection)

    def process_element(self, ent_guid, scd, cm_considered, cm_collisioned):
        ###################################################################################################
        shape_considered = Element(ent_guid, self.model).geometry
        cm_considered.add_object(name=ent_guid, mesh=shape_considered.mesh)
        aabb_considered = shape_considered.aabb
        bbx = tuple([*aabb_considered[0], *aabb_considered[1]])
        for hit in self.idx3d.intersection(bbx):
            ent = self.model.by_id(hit)
            entity_guid = ent.GlobalId
            if entity_guid != ent_guid and entity_guid in scd:
                self.graph.add_edge(entity_guid, ent_guid, attrs={"aabb": True})
                shape_collisioned = Element(entity_guid, self.model).geometry  # self.entities[ent.GlobalId].geometry
                cm_collisioned.add_object(name=entity_guid, mesh=shape_collisioned.mesh)
        return cm_considered, cm_collisioned

    def wrap_trimesh_collision(self, cm_considered, cm_collisioned):
        result_collision = cm_considered.in_collision_other(cm_collisioned, return_data=True, return_names=True)
        if result_collision[0]:
            for e in result_collision[1]:
                self.graph.add_edge(e[0], e[1], attrs={"collision": True})

    def rtree_trimesh_collision_procedure(self):
        for ent_guid in self.selection:  
            cm_considered = trimesh.collision.CollisionManager()
            cm_collisioned = trimesh.collision.CollisionManager()
            cm_considered, cm_collisioned = self.process_element(ent_guid, self.selection, cm_considered,
                                                                 cm_collisioned)
            self.wrap_trimesh_collision(cm_considered, cm_collisioned)

    def aabb_trimesh_collision_procedure(self):
        if not self.path_rtree_index:
            self.rtree_filling()
        self.idx3d = index.Index(p.filename, properties=p)
        self.rtree_trimesh_collision_procedure()
        self.collision_flag = True

    def export_graph(self):
        filename = self.model_name + "_graph_edgelist" + str(uuid.uuid4()) + ".json"
        export_json(filename, nx.to_edgelist(self.graph))
