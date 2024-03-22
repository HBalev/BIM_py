import trimesh
import trimesh.collision
import itertools

from base_classes.structural_model import StructuralModel


class ModelsClash(StructuralModel):
    '''
    Class that gets a model and selection groups and analyses the spatial collisions between the groups
    '''
    def __init__(self, model, selection):
        super(ModelsClash, self).__init__(model, selection)
        self.pairs = itertools.combinations(self.selections, 2)

    def rtree_trimesh_collision_procedure(self):
        for pair in self.pairs:
            fst, scd = pair
            cm_considered = trimesh.collision.CollisionManager()
            cm_collisioned = trimesh.collision.CollisionManager()
            for ent_guid in fst:
                cm_considered, cm_collisioned = self.process_element(ent_guid, scd, cm_considered, cm_collisioned)
            ####################################################################################################################
            self.wrap_trimesh_collision(cm_considered, cm_collisioned)

    def generate_pairs(self, indices: tuple):
        '''
        Function that generates the colliding pair between the two groups named in indices
        '''
        fst = self.selections[indices[0]]
        scd = self.selections[indices[1]]
        for guid in fst:
            for el in self.graph.neighbors(guid):
                if el in scd:
                    yield (guid, el)
