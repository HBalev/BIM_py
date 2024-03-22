import ifcopenshell.geom
import ifcopenshell.util.shape
import ifcopenshell.util.element
import ifcopenshell.api
import ifcopenshell.util.system
import ifcopenshell
import uuid

from base_classes.models_clash import ModelsClash

create_guid = lambda: ifcopenshell.guid.compress(uuid.uuid1().hex)


def cut_holes(model, pairs):
    '''
    Function that cuts openings in construction entities with the form given by modelled void objects
    '''
    drchbr = []
    for pair in pairs:
        construction_el_guid, void_guid = pair
        durchbruch = model.by_guid(void_guid)
        drchbr.append(durchbruch)
        bep_place = durchbruch.ObjectPlacement
        bep_rep = durchbruch.Representation
        owner_history = durchbruch.OwnerHistory
        name = durchbruch.Name
        opening_element = model.createIfcOpeningElement(create_guid(), owner_history, name, "An awesome opening",
                                                        None, bep_place, bep_rep, None)
        model.add(opening_element)
        ifcrelvoid = model.createIfcRelVoidsElement(create_guid(), owner_history, None, None,
                                                    model.by_guid(construction_el_guid), opening_element)
        model.add(ifcrelvoid)
    for el in set(drchbr):
        model.remove(el)
    return model


def provision_for_void(file_path):
    '''
    Implementation of realising openings in construction entities with the form given by modelled void objects
    '''
    model = ifcopenshell.open(file_path)
    sel1 = [k.GlobalId for k in model.by_type("IfcElement") if k.is_a('IfcWall') or k.is_a('IfcSlab')]
    sel2 = [k.GlobalId for k in model.by_type("IfcElement") if
            k.is_a('IfcBuildingElementProxy') and k.PredefinedType == 'PROVISIONFORVOID']
    selections = [sel1, sel2]
    mc = ModelsClash(model, selections)
    mc.aabb_trimesh_collision_procedure()
    mdl = cut_holes(model, list(mc.generate_pairs((0, 1))))
    mdl.write(file_path + "_pfv_" + '.ifc')


provision_for_void("input_files/test.ifc")
