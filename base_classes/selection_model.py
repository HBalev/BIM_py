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
        # if not (self.ifc_graph_flag and self.path_rtree_index):
        #     self.ifc2_3_graph()
        #     self.rtree_filling()
        # elif self.path_rtree_index:
        #     self.ifc2_3_graph()
        # elif self.ifc_graph_flag:
        #     self.rtree_filling()
        self.idx3d = index.Index(p.filename, properties=p)
        self.rtree_trimesh_collision_procedure()
