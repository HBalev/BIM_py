from  base_classes.base_classes import Element
import ifcopenshell
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv3D, MaxPooling3D, Flatten
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import json
import pickle

def standardise_shape(shape):
    vxl = shape.voxelized(ptch)
    enc = np.array(vxl.encoding.dense.tolist()).astype(int)
    enc_with_ptch = transform(enc, ptch)
    padded_to_shape = resize(enc_with_ptch, [cube_dim, cube_dim, cube_dim], mode='edge', anti_aliasing=True)
    return padded_to_shape

def rotate_shape(shp_original,rotation_matrix):
    return shp_original.copy().apply_transform(rotation_matrix)

def transform(a,pitch):
    b=np.zeros(a.shape)
    b[a==True]=pitch
    return b

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

filename=r"C:\Users\hbale\Downloads\DigitalHub_FM-LFT_v2.ifc"
model=ifcopenshell.open(filename)
labels=list(set([el.is_a() for el in model.by_type("IfcElement")]))
onehot_encoder = OneHotEncoder(sparse_output=False)
encoded_labels_onehot = onehot_encoder.fit_transform(np.array(labels).reshape(-1, 1))
mp_cls_to_enc={p[0]:p[1] for p in zip(labels,encoded_labels_onehot)}
with open("encoding_classes.json", "w") as outfile:
    json.dump(mp_cls_to_enc, outfile, cls=NpEncoder)
print("Export finished")
X=[]
y=[]
tol=0.0001
cube_dim=16
for el in model.by_type("IfcElement"):
       el_whole=Element(el.GlobalId,model)
       rotate_around_y = np.array(
           [[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
       rotate_around_x = np.array(
           [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
       rotate_around_z = np.array(
           [[0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
       rotations=[rotate_around_y,rotate_around_x,rotate_around_z]
       shp_original=el_whole.geometry.mesh
       shapes_as_list=[shp_original,shp_original,shp_original]
       max_dim=np.max(shp_original.bounds[1]-shp_original.bounds[0])
       ptch=np.round(max_dim)/cube_dim
       shapes_variants=list(map(rotate_shape,shapes_as_list,rotations))+[shp_original]#[shp_original,shape_copy_y,shape_copy_x,shape_copy_z]
       if ptch>tol:
           for shape in shapes_variants:
                padded_to_shape=standardise_shape(shape)
                X.append(padded_to_shape)
                y.append(mp_cls_to_enc[el_whole.ifc_class])

X_voxel_data=np.array(X)
y_labels_categorical=np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X_voxel_data, y_labels_categorical, test_size=0.2, random_state=42)

with open('X_train3.pkl', 'wb') as f:  # open a text file
     pickle.dump(X_train, f)
with open('X_test3.pkl', 'wb') as f:  # open a text file
     pickle.dump(X_test, f)
with open('y_train3.pkl', 'wb') as f:  # open a text file
     pickle.dump(y_train, f)
with open('y_test3.pkl', 'wb') as f:  # open a text file
     pickle.dump(y_test, f)


with open('X_train3.pkl', 'rb') as f:
    X_train = pickle.load(f)

with open('X_test3.pkl', 'rb') as f:
    X_test = pickle.load(f)

with open('y_train3.pkl', 'rb') as f:
    y_train = pickle.load(f)

with open('y_test3.pkl', 'rb') as f:
    y_test = pickle.load(f)


# Build the model
input_shape = (cube_dim, cube_dim, cube_dim, 1)  # Assuming single channel voxel images

# Define number of classes
num_classes = len(labels)  # Assuming 8 classes

# Define your model
model = Sequential()
model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)

model.save("model2.keras")

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)