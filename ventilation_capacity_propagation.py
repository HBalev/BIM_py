import pandas as pd
import networkx as nx
import ifcopenshell.geom
import ifcopenshell.util.shape
import ifcopenshell.util.element
import ifcopenshell.api
import ifcopenshell.util.system
import ifcopenshell
import uuid

from base_classes.selection_model import SelectionModel


##################################################################################################################################
def preprocess_capacity_propagation(model):
    port_to_el = {k.RelatedElement: k.RelatingPort for k in model.by_type('IfcRelConnectsPortToElement')}

    selection = [m.GlobalId for m in port_to_el.keys()]
    return selection


def change_property_value_pset(element: ifcopenshell.entity_instance, psets_dict: dict,
                               model: ifcopenshell.file):
    for pset_name in psets_dict.keys():
        add_pset_flag = True
        if element.IsDefinedBy:
            for element_definition in element.IsDefinedBy:
                if "RelatingPropertyDefinition" in dir(element_definition):
                    relating_property_definition = element_definition.RelatingPropertyDefinition
                    if relating_property_definition.is_a("IfcPropertySet"):
                        if relating_property_definition.Name == pset_name:
                            add_pset_flag = False
                            ifcopenshell.api.run("pset.edit_pset", model, pset=relating_property_definition,
                                                 properties=psets_dict[pset_name])
        if add_pset_flag:
            pset = ifcopenshell.api.run("pset.add_pset", model, product=element, name=pset_name)
            ifcopenshell.api.run("pset.edit_pset", model, pset=pset,
                                 properties=psets_dict[pset_name])


def process_table(table_path: str, empty_value: str, properties: dict[str, dict], table_data_column_nr: int) -> dict[
    str, dict]:  ###HARDCODED
    print("process_table")
    tab = pd.read_excel(table_path, index_col=None, header=None)
    # sav={}
    # rav={}
    # eav={}
    # oav={}
    # tpe={}
    print(tab.shape)
    for k in range(1, tab.shape[0]):
        row = tab.iloc[k]
        print(row)
        guid = row[0]
        for i in range(1, table_data_column_nr):  # tab.shape[1]):
            if row[i] != empty_value and guid in properties.keys():
                # print(guid, row[i])
                str_val = row[i]
                val = float(str_val.split(" m³/h")[0].replace(',', '.'))
                if i == 1:
                    properties[guid]['SupplyAirVolume'] = val
                elif i == 2:
                    properties[guid]['ReturnAirVolume'] = val
                elif i == 3:
                    properties[guid]['ExhaustAirVolume'] = val
                elif i == 4:
                    properties[guid]['OutsideAirVolume'] = val
        if guid in properties.keys():
            properties[guid]['SystemType'] = row[5]
    return properties


def propagation_procedure(graph: nx.classes.graph.Graph, source: list, target: list, properties: dict[str, dict]) -> \
        dict[str, dict]:
    '''
    Function that propagates a capacity value through the graph from the source vertices to the target vertices and writing the results in the properties dictionary
    :param graph - graph object based on which the propagation between source and target is executed
    :param source - selection of objects that start the propagation
    :param target - selection of objects that end the propagation
    :param properties - dictionary with keys-guids and value a dictionary of property values
    '''
    print("propagation_procedure")
    mapping_sys_name = {"Zuluft": "SupplyAirVolume", "Abluft": "ReturnAirVolume", "Fortluft": "ExhaustAirVolume",
                        "Außenluft": "OutsideAirVolume"}
    # propagate=lambda original,value : original+value
    for cc in nx.connected_components(graph):
        stations = ((src, trgt) for src in source for trgt in target if src in cc and trgt in cc)
        for src, trgt in stations:
            sh_path = nx.shortest_path(graph, source=src, target=trgt)
            sh_path.remove(src)
            print(src, trgt, sh_path)
            for sys_name in mapping_sys_name.keys():
                if sys_name in properties[src]["SystemType"]:
                    category_bin = mapping_sys_name[sys_name]
                    prop_value = properties[src][category_bin]
                    # print(tkt)
                    for n in sh_path:
                        # if n != src:
                        properties[n][category_bin] = properties[n][category_bin] + prop_value
                        # print(sav[n])
            # if "Zuluft" in properties[src]["SystemType"]:
            #     prop_value = properties[src]["SupplyAirVolume"]
            #     # print(tkt)
            #     for n in sh_path:
            #         #if n != src:
            #             properties[n]["SupplyAirVolume"] = properties[n]["SupplyAirVolume"] + prop_value
            #             # print(sav[n])
            # elif "Abluft" in properties[src]["SystemType"]:
            #     prop_value = properties[src]["ReturnAirVolume"]
            #     # print(tkt)
            #     for n in sh_path:
            #         #if n != src:
            #             properties[n]["ReturnAirVolume"] = properties[n]["ReturnAirVolume"] + prop_value
            # elif "Fortluft" in properties[src]["SystemType"]:
            #     prop_value = properties[src]["ExhaustAirVolume"]
            #     # print(tkt)
            #     for n in sh_path:
            #         #if n != src:
            #             properties[n]["ExhaustAirVolume"] = properties[n]["ExhaustAirVolume"] + prop_value
            # elif "Außenluft" in properties[src]["SystemType"]:
            #     prop_value = properties[src]["OutsideAirVolume"]
            #     # print(tkt)
            #     for n in sh_path:
            #         #if n != src:
            #             properties[n]["OutsideAirVolume"] = properties[n]["OutsideAirVolume"] + prop_value
    return properties


def cleanup_edges(graph: nx.classes.graph.Graph, model: ifcopenshell.file) -> nx.classes.graph.Graph:  ####HARDCODED
    print("cleanup_edges")
    '''
        Helper function that removes edges in the graph object based on semantic contradictions between both ends of edges 
        :param graph - graph object which contradicting edges are removed
        :param model - ifc model

        '''
    extract_system_name = lambda s: s.Name.split("Lüftung LÜ_")[1].split(' ')[0]  # hardcoded
    extract_system = lambda model, n: ifcopenshell.util.system.get_element_systems(model.by_guid(n))
    system_name_contradiction = lambda model, fst, scd: ("Zuluft" in extract_system(model, fst)[
        0].Name and "Abluft" in model.by_guid(scd).Name) or (
                                                                "Abluft" in extract_system(model, fst)[
                                                            0].Name and "Zuluft" in model.by_guid(scd).Name)
    name_name_contradiction = lambda model, fst, scd: (
            "Zuluft" in model.by_guid(fst).Name and "Abluft" in model.by_guid(scd).Name)
    for e in graph.edges():
        n1 = e[0]
        n2 = e[1]
        if extract_system(model, n1) and extract_system(model, n2):  # case both nodes have ifc-system
            sys1 = extract_system_name(extract_system(model, n1)[0])
            sys2 = extract_system_name(extract_system(model, n2)[0])
            if sys1 != sys2 and not (model.by_guid(n1).is_a("IfcEnergyConversionDevice") or model.by_guid(n2).is_a(
                    "IfcEnergyConversionDevice") or model.by_guid(n1).is_a("IfcFlowMovingDevice") or model.by_guid(
                n2).is_a("IfcFlowMovingDevice")):
                graph.remove_edge(n1, n2)
        elif extract_system(model, n1):  # case just the first node has a system
            if system_name_contradiction(model, n1,
                                         n2):  # ("Zuluft" in extract_system(model, n1)[0].Name and "Abluft" in model.by_guid(n2).Name) or (Abluft" in extract_system(model, n1)[0].Name and "Zuluft" in model.by_guid(n2).Name):
                graph.remove_edge(n1, n2)
        elif extract_system(model, n2):  # case just the second node has a system
            if system_name_contradiction(model, n2,
                                         n1):  # ("Zuluft" in extract_system(model, n2)[0].Name and "Abluft" in model.by_guid(n1).Name) or ("Abluft" in extract_system(model, n2)[0].Name and "Zuluft" in model.by_guid(n1).Name):
                graph.remove_edge(n1, n2)
        else:
            if name_name_contradiction(model, n1, n2) or name_name_contradiction(model, n2,
                                                                                 n1):  # ("Zuluft" in model.by_guid(n1).Name and "Abluft" in model.by_guid(n2).Name) or ("Zuluft" in model.by_guid(n2).Name and "Abluft" in model.by_guid(n1).Name):
                graph.remove_edge(n1, n2)
    print(type(graph))
    return graph


def capacity_propagation_ventilation(file_path, table_path):
    model = ifcopenshell.open(file_path)
    selection = preprocess_capacity_propagation(model)
    sm = SelectionModel(model, selection)
    sm.aabb_trimesh_collision_procedure()
    graph = cleanup_edges(sm.graph, model)
    #############################################################################################################
    #################################Initialisation##############################################################
    properties = {}
    for el in graph.nodes:
        properties[el] = {"SupplyAirVolume": 0, "ReturnAirVolume": 0, "ExhaustAirVolume": 0, "OutsideAirVolume": 0,
                          "SystemType": ""}
    properties = process_table(table_path, '-999,0 m³/h', properties, 4)
    #############################################################################################################
    #################################Propagation#################################################################
    properties = propagation_procedure(graph, [k.GlobalId for k in model.by_type("IfcFlowTerminal")],
                                       [k.GlobalId for k in model.by_type("IfcEnergyConversionDevice")], properties)
    #############################################################################################################
    #################################Prepare new file############################################################
    for guid in properties.keys():
        change_property_value_pset(model.by_guid(guid), {"Flowrates": properties[guid]}, model)
    new_model_filename = str(uuid.uuid4()) + '.ifc'
    model.write(new_model_filename)
    return new_model_filename


##################################################################################################################################################

capacity_propagation_ventilation()
