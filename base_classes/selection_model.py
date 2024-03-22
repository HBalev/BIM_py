from rtree import index
import ifcopenshell.geom
import ifcopenshell.util.shape
import ifcopenshell.util.element
import ifcopenshell.api
import ifcopenshell.util.system
import ifcopenshell

from base_classes.base_classes import p
from base_classes.structural_model import StructuralModel


class SelectionModel(StructuralModel):
    '''
            Class SelectionModel models the spatial connections on a selection of ifc-entities
            :param selection - list of GlobalIds of selected entities
            :param model - opened Ifc-model-entity
            :attr  entities - list of Element-objects based on the selection list
            :attr  graph - networkx Graph-object containing all selected entities and their connections
            :attr  g_tree - ifc-geometric-tree object
            :attr  idx3d - rtree-spatial-tree object
            :method  ifc2_3_graph - extracts graph connections based on ifc-model and Distribution Ports
            :method  export_graph - exports .json - version of the graph formatted as dictionary of list and edgelist with edge attributes
            :method  fill_rtree_spatial - fills the self.idx3d-rtree with aabb-s of the selected entities
            :method  fill_ifc_tree - fills the self.g_tree-ifc-tree with geometry
            :method  aabb_trimesh_collision_procedure - fills the self.graph with connections obtained from aabb-intersection and Trimesh-collision analysis
            '''
    def __init__(self, model, selection):
        super(SelectionModel, self).__init__(model, selection)
        self.g_tree: ifcopenshell.geom.main.tree = ifcopenshell.geom.tree()
        self.ifc_graph_flag: bool = False

    def ifc2_3_graph(self):
        raw_edges = [(k.RelatedPort, k.RelatingPort) for k in self.model.by_type('IfcRelConnectsPorts')]
        port_to_el = {k.RelatingPort: k.RelatedElement for k in self.model.by_type('IfcRelConnectsPortToElement')}
        edges = [(port_to_el[e[0]], port_to_el[e[1]]) for e in raw_edges]
        for k in edges:
            fst = k[0].GlobalId
            scd = k[1].GlobalId
            self.graph.add_node(fst)
            self.graph.add_node(scd)
            self.graph.add_edge(fst, scd, attrs={'ifc': True})
        self.ifc_graph_flag = True

    def aabb_trimesh_collision_procedure(self):
        print("aabb_trimesh_collision_procedure")
        reqs = {"ifc_graph_flag": "ifc2_3_graph", "path_rtree_index": "rtree_filling"}
        for flag in reqs.keys():
            if not getattr(self, flag):
                getattr(self, reqs[flag])()
        self.idx3d = index.Index(p.filename, properties=p)
        self.rtree_trimesh_collision_procedure()
