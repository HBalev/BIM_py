import rtree
from rtree import index
import ifcopenshell.geom
import ifcopenshell.util.shape
import ifcopenshell.util.element
import ifcopenshell.api
import ifcopenshell.util.system
import ifcopenshell
import uuid
import numpy as np

from base_classes.base_classes import p, fill, Shape


class SpatialModel:
    tol = 0.5

    def __init__(self, model, selection=None):
        assert isinstance(model, ifcopenshell.file)
        self.model = model
        self.model_name = self.model.by_type("IfcProject")[0].Name
        if selection is None:
            self.selection = [el.GlobalId for el in self.model.by_type("IfcElement")]
            self.dimension = 1
        elif not isinstance(selection[0], list):
            self.selection = selection
            self.dimension = 1
        else:
            self.selections = selection
            self.selection = [el for sel in self.selections for el in sel]
            self.dimension = len(self.selections)
        p.filename = self.model_name + "_" + str(uuid.uuid4())
        self.idx3d: rtree.index.Index = index.Index(p.filename, properties=p)
        self.path_rtree_index = None
        self.g_tree: ifcopenshell.geom.main.tree = ifcopenshell.geom.tree()

    @fill
    def rtree_spatial(self, iterator):
        shape = iterator.get()
        current_guid = shape.guid
        geometry = Shape(current_guid, shape)
        vector_bbx = np.array([*geometry.aabb[0], *geometry.aabb[1]])
        vector_tolerance = np.repeat(np.array([-self.tol, self.tol]), 3)
        rtree_bbx = tuple(vector_bbx + vector_tolerance)
        self.idx3d.insert(shape.id, rtree_bbx)

    def rtree_filling(self):
        self.rtree_spatial()
        self.idx3d.close()
        print("Idx closed")
        self.path_rtree_index = p.filename
        print(self.path_rtree_index)

    @fill
    def ifc_tree(self, iterator):
        print("get_ifc_tree")
        self.g_tree.add_element(iterator.get_native())
