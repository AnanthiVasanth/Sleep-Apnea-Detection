# Sleep-Apnea-Detection
Sleep Apnea Detection Using Hybrid Deep Learning Model
The ECG signal is pre-processed and analyzed by the deep learning model to recognize sleep apnea. Among deep learning techniques, convolutional network (MobileNetV1) and hybrid convolutional-recurrent networks: MobileNetV1-LSTM (Long Short-Term Memory) and MobileNetV1-GRU (Gated Recurrent Unit) are implemented. It is found that the hybrid deep model MobileNetV1-GRU achieved the best detection results with an accuracy of 90.07%.


SOURCE CODE:
SIGNAL PREPROCESSING

!pip install biosppy
!pip install cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed import biosppy.signals.tools as st
from biosppy.signals.ecg import correct_rpeaks, hamilton_segmenter from scipy.signal import medfilt
from cpu_count import cpu_count from tqdm import tqdm
base_dir = "/content/drive/MyDrive/apnea/apnea-ecg/1.0.0" fs = 100
sample = fs * 60 # 1 min's sample points before = 2 # forward interval (min)
after = 2 # backward interval (min) hr_min = 20
hr_max = 300
num_worker = 35 if cpu_count() > 35 else cpu_count() - 1 def worker(name, labels):
X = []
y = []
groups = []
signals = wfdb.rdrecord(os.path.join(base_dir, name), channels=[0]).p_signal[:, 0] for j in tqdm(range(len(labels)), desc=name, file=sys.stdout):
if j < before or (j + 1 + after) > len(signals) / float(sample):
continue
signal = signals[int((j - before) * sample):int((j + 1 + after) * sample)]
signal, _, _ = st.filter_signal(signal, ftype='FIR', band='bandpass', order=int(0.3 * fs), frequency=[3, 45], sampling_rate=fs)
 

rpeaks, = hamilton_segmenter(signal, sampling_rate=fs)
rpeaks, = correct_rpeaks(signal, rpeaks=rpeaks, sampling_rate=fs, tol=0.1)
if len(rpeaks) / (1 + after + before) < 40 or len(rpeaks) / (1 + after + before) > 200: continue
rri_tm, rri_signal = rpeaks[1:] / float(fs), np.diff(rpeaks) / rri_signal = medfilt(rri_signal, kernel_size=3)
ampl_tm, ampl_siganl = rpeaks / float(fs), signal[rpeaks] hr = 60 / rri_signal
if np.all(np.logical_and(hr >= hr_min, hr <= hr_max)): X.append([(rri_tm, rri_signal), (ampl_tm, ampl_siganl)]) y.append(0. if labels[j] == 'N' else 1.) groups.append(name)
return X, y, groups

MOBILENETV1 MODEL

dir= "/content/drive/MyDrive/apnea"
ir = 3 # INTERPOLATION RATE(3HZ)
time_range= 60 # 60-s INTERVALS OF ECG SIGNALS
weight=1e-3
# NORMALIZATION: # MIN-MAX METHOD APPLIED FOR SCALING
scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr))def load_data(): tm = np.arange(0, (time_range), step=(1) / float(ir))
# time metric for interpolation # load and interpolate r-r intervals and r-peak amplitudes with open(os.path.join(dir,"/content/drive/MyDrive/apnea/apneaecg/apnea-ecg.pkl"), 'rb') as f:
apnea_ecg = pickle.load(f) x = []
X = apnea_ecg["o_train"] Y=apnea_ecg["y_train"] for i in range(len(X)):
(rri_tm, rri_signal), (amp_tm, amp_signal) = X[i] rri_interp_signal = splev(tm, splrep(rri_tm,
 

scaler(rri_signal), k=3), ext=1) amp_interp_signal = splev(tm, splrep(amp_tm, scaler(amp_signal), k=3), ext=1) x.append([rri_interp_signal, amp_interp_signal]) x = np.array(x, dtype="float32")
x = np.expand_dims(x,1)
x_final=np.array(x, dtype="float32").transpose((0,3,1,2)) return x_final, Y
def mobilenetV1():
input_shape = (180, 1, 2)
inputs = tf.keras.Input(shape=input_shape)
model1 = Conv2D(32, (3, 1), strides=(1, 1), padding='same')(inputs) model1 = BatchNormalization()(model1)
model1 = ReLU()(model1)
model1 = DepthwiseConv2D((3, 1), padding='same')(model1) model1 = BatchNormalization()(model1)
model1 = ReLU()(model1)
model1 = Conv2D(64, (1, 1), padding='same')(model1) model1 = BatchNormalization()(model1)
model1 = ReLU()(model1)
model1 = DepthwiseConv2D((3, 1), strides=(2, 2), padding='same')(model1)
model1 = BatchNormalization()(model1) model1 = ReLU()(model1)
model1 = Conv2D(128, (1, 1), padding='same')(model1) model1 = BatchNormalization()(model1)
model1 = ReLU()(model1)
model1 = DepthwiseConv2D((3, 1), strides=(2, 2), padding='same')(model1)
model1 = BatchNormalization()(model1) model1 = ReLU()(model1)
model1 = Conv2D(256, (1, 1), padding='same')(model1) model1 = BatchNormalization()(model1)
 

model1 = ReLU()(model1)
model1 = DepthwiseConv2D((3, 1), strides=(2, 2), padding='same')(model1)
model1 = BatchNormalization()(model1) model1 = ReLU()(model1)
model1 = GlobalAveragePooling2D()(model1) model1 = Dense(4, activation="relu") (model1) outputs = Dense(2, activation='softmax')(model1)
model = tf.keras.Model(inputs=inputs, outputs=outputs) return model
def lr_schedule(epoch, lr):
if epoch > 30 and \ (epoch - 1) % 10 == 0:
lr *= 0.1
print("Learning rate: ", lr) return lr
# we have labels(Y) in a binary way 0 for normal and 1 for apnea patients Y = tf.keras.utils.to_categorical(Y, num_classes=2)
# we used k-fold cross-validation for more reliable experiments: kfold = StratifiedKFold(n_splits=5, shuffle=True,random_state=7) # separate train& test and then compile model
for train, test in kfold.split(X, Y.argmax(1)):
model = mobilenetV1()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
# define callback for early stopping:
lr_scheduler = LearningRateScheduler(lr_schedule) callback1 = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#10% of Data used for validation
for train, test in kfold.split(X, Y.argmax(1)):
model = mobilenetV1()
model.compile(optimizer="adam", loss="categorical_crossentropy",
 

metrics=['accuracy'])
# define callback for early stopping:
lr_scheduler = LearningRateScheduler(lr_schedule) callback1 = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#10% of Data used for validation: X1,x_val,Y1,y_val=train_test_split(X[train],Y[train],test_size=0.10)
history = model.fit(X1, Y1, batch_size=128, epochs=100, validation_data=(x_val, y_val), callbacks=[callback1,lr_scheduler])
model.save(os.path.join("model.mobilenetV1.h5")) loss, accuracy = model.evaluate(X[test], Y[test]) y_score = model.predict(X[test])
y_predict= np.argmax(y_score, axis=-1) y_training = np.argmax(Y[test], axis=-1)
C = confusion_matrix(y_training, y_predict, labels=(1, 0)) TP, TN, FP, FN = C[0, 0], C[1, 1], C[1, 0], C[0, 1]
acc, sn, sp = 1. * (TP + TN) / (TP + TN + FP + FN), 1. * TP / (TP
+ FN), 1. * TN / (TN + FP)
f2=f1_score(y_training, y_predict) print(“Accuracy\tSensitivity\specificity\F1score”) print(acc*100\t\tsn*100\t\tsp*100\t\t*f2*100)
loss, accuracy = model.evaluate(x_val, y_val) # test the model print("Test loss: ", loss)
print("Test accuracy: ", accuracy) # save prediction score
y_score = model.predict(x_val)

MOBILENETV1-LSTM MODEL

def mobilenetlstm():
model= Sequential() model.add(Reshape((90,2,2),input_shape=(180,1,2))) model.add(Conv2D(96, kernel_size=(11,1), strides=(1,1),
 

padding="same", activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(weight), bias_regularizer=l2(weight),input_shape=(180,1,2))) model.add(BatchNormalization()) model.add(MaxPooling2D(pool_size=(3,1)))
model.add(Conv2D(256, kernel_size=(5,1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(weight),
bias_regularizer=l2(weight))) model.add(BatchNormalization()) model.add(MaxPooling2D(pool_size=(3,1))) model.add(Conv2D(384, kernel_size=(3,1), strides=(1,1),
padding="same", activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(weight),
bias_regularizer=l2(weight))) model.add(BatchNormalization())
model.add(Conv2D(384, kernel_size=(3,1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(weight),
bias_regularizer=l2(weight))) model.add(BatchNormalization())
model.add(Conv2D(256, kernel_size=(3,1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(weight),
bias_regularizer=l2(weight))) model.add(BatchNormalization()) model.add(MaxPooling2D(pool_size=(3,1),strides=(2,1)))
model.add(Conv2D(256, kernel_size=(3,1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(weight),
bias_regularizer=l2(weight))) model.add(BatchNormalization()) model.add(Permute((2,1,3)))
 

model.add(Reshape((2,4*256))) model.add(LSTM(64, return_sequences=True)) model.add(Flatten())
model.add(Dense(18, activation="relu")) model.add(Dense(2, activation="softmax")) return model

MOBILENETV1-GRU MODEL

def mobilenetGRU():
model= Sequential() model.add(Reshape((90,2,2),input_shape=(180,1,2))) model.add(Conv2D(96, kernel_size=(11,1), strides=(1,1), padding="same", activation="relu", kernel_initializer="he_normal",
kernel_regularizer=l2(weight), bias_regularizer=l2(weight),input_shape=(180,1,2))) model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,1)))
model.add(Conv2D(256, kernel_size=(5,1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(weight),
bias_regularizer=l2(weight))) model.add(BatchNormalization()) model.add(MaxPooling2D(pool_size=(3,1))) model.add(Conv2D(384, kernel_size=(3,1), strides=(1,1),
padding="same", activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(weight),
bias_regularizer=l2(weight))) model.add(BatchNormalization())
model.add(Conv2D(384, kernel_size=(3,1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(weight),
bias_regularizer=l2(weight))) model.add(BatchNormalization())
 

model.add(Conv2D(256, kernel_size=(3,1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(weight),
bias_regularizer=l2(weight))) model.add(BatchNormalization()) model.add(MaxPooling2D(pool_size=(3,1),strides=(2,1)))
model.add(Conv2D(256, kernel_size=(3,1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(weight),
bias_regularizer=l2(weight))) model.add(BatchNormalization()) model.add(Permute((2,1,3))) model.add(Reshape((2,4*256))) model.add(GRU(64, return_sequences=True)) model.add(Flatten())
model.add(Dense(18, activation="relu")) model.add(Dense(2, activation="softmax")) return model
 


