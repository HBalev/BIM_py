from  base_classes.base_classes import Element
import ifcopenshell
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv3D, MaxPooling3D, Flatten
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from skimage.transform import resize

def transform(a,pitch):
    b=np.zeros(a.shape)
    b[a==True]=pitch
    return b


filepath=r""
model=ifcopenshell.open(filepath)
labels=list(set([el.is_a() for el in model.by_type("IfcElement")]))

onehot_encoder = OneHotEncoder(sparse_output=False)
encoded_labels_onehot = onehot_encoder.fit_transform(np.array(labels).reshape(-1, 1))
mp_cls_to_enc={p[0]:p[1] for p in zip(labels,encoded_labels_onehot)}
X=[]#np.array([])
y=[]#np.array([])
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
       shp_original=el_whole.geometry.mesh
       max_dim=np.max(shp_original.bounds[1]-shp_original.bounds[0])
       ptch=np.round(max_dim)/cube_dim
       shape_copy_y=shp_original.copy().apply_transform(rotate_around_y)
       shape_copy_x = shp_original.copy().apply_transform(rotate_around_x)
       shape_copy_z = shp_original.copy().apply_transform(rotate_around_z)
       shapes_variants=[shp_original,shape_copy_y,shape_copy_x,shape_copy_z]
       if ptch>tol:
           for shape in shapes_variants:
                   print(el)
                   print(max_dim)
                   print(ptch)
                   vxl=shape.voxelized(ptch)
                   enc=np.array(vxl.encoding.dense.tolist()).astype(int)
                   enc_with_ptch=transform(enc,ptch)
                   padded_to_shape=resize(enc_with_ptch,[cube_dim,cube_dim,cube_dim],mode='edge', anti_aliasing=True)#to_shape(enc_with_ptch,[cube_dim,cube_dim,cube_dim])
                   X.append(padded_to_shape)
                   y.append(mp_cls_to_enc[el_whole.ifc_class])



X_voxel_data=np.array(X)
y_labels_categorical=np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X_voxel_data, y_labels_categorical, test_size=0.2, random_state=42)
import pickle
with open('X_train2.pkl', 'wb') as f:  # open a text file
     pickle.dump(X_train, f)
with open('X_test2.pkl', 'wb') as f:  # open a text file
     pickle.dump(X_test, f)
with open('y_train2.pkl', 'wb') as f:  # open a text file
     pickle.dump(y_train, f)
with open('y_test2.pkl', 'wb') as f:  # open a text file
     pickle.dump(y_test, f)


with open('X_train2.pkl', 'rb') as f:
    X_train = pickle.load(f)

with open('X_test2.pkl', 'rb') as f:
    X_test = pickle.load(f)

with open('y_train2.pkl', 'rb') as f:
    y_train = pickle.load(f)

with open('y_test2.pkl', 'rb') as f:
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

model.save("model1.keras")

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)