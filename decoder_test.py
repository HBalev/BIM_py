from base_classes.base_classes import Element
import ifcopenshell
import numpy as np
from skimage.transform import resize
import keras
import json


class NpEncoder(json.JSONEncoder):
    '''
    Helper class for exporting .json that handles non-serializable data types
    '''
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

def standardise_shape(shape):
    vxl = shape.voxelized(ptch)
    enc = np.array(vxl.encoding.dense.tolist()).astype(int)
    enc_with_ptch = transform(enc, ptch)
    padded_to_shape = resize(enc_with_ptch, [cube_dim, cube_dim, cube_dim], mode='edge', anti_aliasing=True)
    return padded_to_shape

def transform(a,pitch):
    b=np.zeros(a.shape)
    b[a==True]=pitch
    return b

model=ifcopenshell.open()

with open("encoding_classes.json") as json_file:
    mp_cls_to_enc=json.load(json_file)

X=[]
y=[]
tol=0.0001
cube_dim=16
labels=[]
for el in model.by_type("IfcElement"):
       el_whole=Element(el.GlobalId,model)
       shp_original=el_whole.geometry.mesh
       max_dim=np.max(shp_original.bounds[1]-shp_original.bounds[0])
       ptch=np.round(max_dim)/cube_dim
       if ptch>tol:
            padded_to_shape=standardise_shape(shp_original)
            labels.append(el.is_a())
            X.append(padded_to_shape)

X_voxel_data=np.array(X)
loaded_model = keras.saving.load_model("model2.keras")
predictions=np.round(loaded_model.predict(X_voxel_data))
predictions_list=predictions.tolist()

for k in predictions_list:
    for s in mp_cls_to_enc.keys():
        if all(x == y for x, y in zip(mp_cls_to_enc[s], k)):
            print(s)
            print(labels[predictions_list.index(k)])




