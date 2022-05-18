import os
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from joblib import dump, load
import json

#hyperparameters
BATCH_SIZE=10
LOW_THRESHOLD=75
EPOCHS=80
LEARNING_RATE=.0005

data_dir='/tf/data'
fname = os.path.join(data_dir,'data.csv')
f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

print(header, len(lines))
print(lines[0])

glucose_index = 0
keep_from_index = 2

len_of_each_line = len(lines[1].split(','))

data = np.zeros((len(lines),(len_of_each_line-keep_from_index)))

deletes = []
for i,l in enumerate(lines):
    if l!='':
        data[i,:]  = [float(x) for x in l.split(',')[keep_from_index:]]
    else:
        deletes.append(i)
        print("delete ",i)
data = np.delete(data,deletes, axis=0)
print("samples: ", len(data))
datab = np.copy(data)
print(datab[108:115])

up = True
low_count = 0
low_index = []
low_data_ = []
low_data = []
low_data_map = []
normal_data = []
low_threshold = LOW_THRESHOLD
'''
The way this works is we'll use measures_before to predict. but lookahead is the gap between the last measure_before and the actual low event
so this is handy if you want to predict based on the last 2 hours you have a low in the next 10 minutes, 
you would set lookahead to 2 and measure before to 24
'''
measures_before = 24
lookahead = 2
low_data_index = 0
for index, i in enumerate(datab):
    if i[0] <= low_threshold and up:
        low_count+=1
        up = False
        if index > measures_before + lookahead:
            low_data_.append(datab[index-measures_before-lookahead:index-lookahead])
            low_data_map.append([low_data_index, index])
            low_data_index+=1

    if i[0] >low_threshold and not up:
        up = True

continuous = 0
for index, i in enumerate(datab):
    if i[0] <=low_threshold :
        continuous = 0
    if i[0] > low_threshold :
        continuous+=1
    if continuous > measures_before:
        normal_data.append(datab[index-measures_before-lookahead:index-lookahead])

np.random.shuffle(low_data_map)
np.random.shuffle(normal_data)
for i in range(len(low_data_map)):
    low_data.append(low_data_[low_data_map[i][0]])

splits = [.75,.2,.05]
norm_multiplier = 2
if(sum(splits) != 1):
    raise ValueError('splits must add up to 1')

train_low_i = round(len(low_data)*splits[0])
val_low_i = [train_low_i, train_low_i+round(len(low_data)*splits[1])]
test_low_i = [val_low_i[1], val_low_i[1]+round(len(low_data)*splits[2])]

train_norm_i = train_low_i * norm_multiplier
val_norm_i = [train_norm_i, val_low_i[1] * norm_multiplier]
test_norm_i = [val_norm_i[1], test_low_i[1] * norm_multiplier]

#training data
train_data_low = np.array(low_data[:train_low_i])
train_data_norm = np.array(normal_data[:train_norm_i])

train_labels_low = np.full((train_low_i,), 1)
train_labels_norm = np.full((train_norm_i,), 0)

train_data = np.concatenate((train_data_low, train_data_norm), axis=0)
train_labels = np.concatenate((train_labels_low, train_labels_norm), axis=0)
#validation data
val_data_low = np.array(low_data[val_low_i[0]:val_low_i[1]])
val_data_norm = np.array(normal_data[val_norm_i[0]:val_norm_i[1]])

val_labels_low = np.full((len(val_data_low),), 1)
val_labels_norm = np.full((len(val_data_norm),), 0)

val_data = np.concatenate((val_data_low, val_data_norm), axis=0)
val_labels = np.concatenate((val_labels_low, val_labels_norm), axis=0)
#test data
test_data_low = np.array(low_data[test_low_i[0]:test_low_i[1]])
test_data_norm = np.array(normal_data[test_norm_i[0]:test_norm_i[1]])
test_labels_low = np.full((len(test_data_low),), 1)
test_labels_norm = np.full((len(test_data_norm),), 0)
test_data = np.concatenate((test_data_low, test_data_norm), axis=0)
test_labels = np.concatenate((test_labels_low, test_labels_norm), axis=0)
print("*** LOWS ***")
print("training low data indexes: ","000","->", train_low_i)
print("validati low data indexes: ", val_low_i[0],"->",val_low_i[1])
print("testing  low data indexes: ", test_low_i[0],"->",test_low_i[1])

print("*** NORMS ***")
print("training norm data indexes: ","000","->", train_norm_i)
print("validati norm data indexes: ", val_norm_i[0],"->",val_norm_i[1])
print("testing  norm data indexes: ", test_norm_i[0],"->",test_norm_i[1])

print("*** COMBINED ***")
print("training: ", train_data.shape, train_labels.shape)
print("validati: ", val_data.shape, val_labels.shape)
print("test.   : ", test_data.shape, test_labels.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_data_flat = train_data.reshape(-1,3)

scaler.fit(train_data_flat)

scaler_json = {
    "mean":scaler.mean_.tolist(),
    "var":scaler.var_.tolist()
}
out_file = open("./data/scalerjson", "w")
json.dump(scaler_json, out_file, indent = 6)
  
out_file.close()

for i in range(len(train_data)):
    train_data[i] = scaler.transform(train_data[i])
for i in range(len(val_data)):
    val_data[i] = scaler.transform(val_data[i])

print("data summary")
print(train_data.shape, "\n", train_data)
print(val_data.shape, "\n", val_data)
print(test_data.shape, "\n", test_data)

#assumes exploratory model used to find hyperparams
all_data = np.concatenate((train_data, val_data), axis=0)
all_labels = np.concatenate((train_labels, val_labels), axis=0)
model_with_dropout = Sequential()
model_with_dropout.add(layers.GRU(32, reset_after=False,  return_sequences=True, input_shape=(None, train_data.shape[-1])))
model_with_dropout.add(layers.GRU(64, reset_after=False, dropout=0.2, activation='relu'))
model_with_dropout.add(layers.Dense(1, activation="sigmoid"))
model_with_dropout.compile(loss="binary_crossentropy", optimizer=Adam(LEARNING_RATE))
history_with_dropout = model_with_dropout.fit(all_data, all_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(val_data, val_labels))


#make LSTM capable with batch of 1
#thanks: https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/
model_p = Sequential()
model_p.add(layers.GRU(32,  reset_after=False,  return_sequences=True, input_shape=(None, data.shape[-1]))) #interestingly you don't give it all the samples in shape
model_p.add(layers.GRU(64, reset_after=False,  activation='relu'))
model_p.add(layers.Dense(1, activation="sigmoid"))
#copy weights from trained model
old_weights = model_with_dropout.get_weights()

model_p.set_weights(old_weights)
model_p.compile(loss="binary_crossentropy", optimizer='adam')
#save the network
model_p.save(os.path.join(data_dir,'model_classify_no_reset_after.h5'))

#save the scaler. need to scale new readings on prediction 
dump(scaler, os.path.join(data_dir,'scaler_model_classify.bin'), compress=True)

print("mean: ", scaler.mean_)
print("var:  ", scaler.var_)