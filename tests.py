import cv2; import random; import os; from importlib import reload

# returns the confusion matrix for output prediction output from network
# each row is assumed to be a datapoint
def gen_confusion_mat(y_real, y_pred, pprint=True):
    y_pred_indices = []
    y_real_indices = []
    for i in range(len(y_pred)):
        y_pred_indices.append(y_pred[i].argmax())
        y_real_indices.append(y_real[i].argmax())
    confusion_mat = confusion_matrix(y_real_indices, y_pred_indices)
    if pprint: print(confusion_mat)
    return confusion_mat

# returns a dictionary of recall (senstivity) for each class from confusion matrix
def calc_recall(confusion_mat, labels):
    recall_dict = {}
    for i, arr in enumerate(confusion_mat):
        recall_dict[labels[i]] = float(arr[i])/sum(arr)
    return recall_dict

# returns overall accuracy based on a confusion matrix
def calc_acc(confusion_mat):
    TP_sum = np.trace(confusion_mat)
    all_sum = np.sum(confusion_mat)
    return float(TP_sum)/all_sum

# returns a flat vector instead of an image matrix. Also converts image to gray
def flatten_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('float32')
    gray = gray/255
    return gray.flatten()

base_dict = {} # A
opt_B_dict = {}
opt_C_dict = {}
for i in range(10): # run multiple times and average
    from numpy.random import seed # see https://machinelearningmastery.com/reproducible-results-neural-networks-keras/
    state = 99+i # state changes for each run
    seed(state)
    save = True # whether or not save confusion matrix of each approach to file

    import numpy as np
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.optimizers import RMSprop
    from keras.datasets import cifar10
    from keras.utils import to_categorical
    from keras.losses import categorical_crossentropy
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle
    from keras import backend as K

    # change keras backend to theano -- tf is not producing consistent output
    def set_keras_backend(backend):

        if K.backend() != backend:
            os.environ['KERAS_BACKEND'] = backend
            reload(K)
            assert K.backend() == backend

    set_keras_backend('theano')


    # TODO: add flags

    # load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    labels = list(set(list(y_train.flatten()))) # 0,1,2,3,4,5,6,7,8,9
    print('classes: ', labels)

    print('before flattening...')
    print('Input shape: ', x_train[0].shape)
    num_classes = len(labels)
    y_train, y_test = to_categorical(y_train, num_classes), to_categorical(y_test, num_classes)
    # generate grayscale flattened arrays
    x_train = np.array([flatten_img(img) for img in x_train])
    x_test = np.array([flatten_img(img) for img in x_test])
    print('after flattening...')
    print('Input shape: ', x_train[0].shape)
    # split datasets
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                        test_size = 0.2, random_state=state, shuffle=True)

    # the parameters
    BATCH = 800
    EPOCHS = 30
    LEARN = 0.001 # learning rate

    # the network (feedforward without convolutional layers)
    model = Sequential()
    model.add(Dense(256, input_shape=(1024,)))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    # compile network
    model.compile(optimizer=RMSprop(lr=LEARN),
                    loss=categorical_crossentropy,
                    metrics=['accuracy'])

    print(model.summary())
    # TODO: set seed for each run. Make a list of seeds and use them one/trial

    # A. FULLY RANDOM
    # shuffle datasets
    x_train, y_train = shuffle(x_train, y_train, random_state=state)

    # run the model
    model.fit(x_train, y_train,
                batch_size=BATCH,
                epochs=EPOCHS,
                validation_data=(x_valid, y_valid),
                verbose=0)

    predictions =  model.predict(x_test)
    conf_mat_A = gen_confusion_mat(y_test, predictions)
    np.savetxt('conf_mat_A.csv', conf_mat_A, fmt='%i', delimiter=',')
    acc_A = calc_acc(conf_mat_A)
    print('accuracy for run A: \n ', acc_A)
    recall_A = calc_recall(conf_mat_A, labels)
    print('recalls for run A: \n ', recall_A)
    base_dict[i] = (acc_A, recall_A, conf_mat_A)

    # B. Place the class with lowest recall in the front and all the way at back (3)
    # shuffle datasets
    # x_train, y_train = shuffle(x_train, y_train, random_state=state)
    # rearrange training data: get all the indices of class 3 and subset x and add half infront and half to the end
    indices_3 = [i for i, arr in enumerate(y_train) if np.argmax(arr) == 3]
    x_train_3, y_train_3 = x_train[indices_3], y_train[indices_3]
    x_train_red, y_train_red = np.delete(x_train, indices_3, 0), np.delete(y_train, indices_3, 0)
    # x_train_B = np.vstack([x_train_3[:len(x_train_3)//2], x_train_red, x_train_3[len(x_train_3)//2:]])
    # y_train_B = np.vstack([y_train_3[:len(y_train_3)//2], y_train_red, y_train_3[len(y_train_3)//2:]])
    # x_train_B = np.vstack([x_train_red, x_train_3])
    # y_train_B = np.vstack([y_train_red, y_train_3])
    x_train_B = np.vstack([x_train_3, x_train_red])
    y_train_B = np.vstack([y_train_3, y_train_red])

    # run the model
    model.fit(x_train_B, y_train_B,
                batch_size=BATCH,
                epochs=EPOCHS,
                validation_data=(x_valid, y_valid),
                verbose=0)

    predictions =  model.predict(x_test)
    conf_mat_B = gen_confusion_mat(y_test, predictions)
    np.savetxt('conf_mat_B.csv', conf_mat_B, fmt='%i', delimiter=',')
    acc_B = calc_acc(conf_mat_B)
    print('accuracy for run B: \n ', acc_B)
    recall_B = calc_recall(conf_mat_B, labels)
    print('recalls for run B: \n ', recall_B)
    opt_B_dict[i] = (acc_B, recall_B, conf_mat_B)


    # C. Train with 1 and 9 alternated (since both a confused with each other)
    # suffle data
    # x_train, y_train = shuffle(x_train, y_train, random_state=state)

    indices_1 = [i for i, arr in enumerate(y_train) if np.argmax(arr) == 1]
    indices_9 = [i for i, arr in enumerate(y_train) if np.argmax(arr) == 9]
    indices_19 = [None]*(min(len(indices_1),len(indices_9))*2) # for alternating
    if len(indices_1) >= len(indices_9):
        diff = len(indices_1) - len(indices_9)
        indices_19[::2] = indices_9
        indices_19[1::2] = indices_1[:-diff]
        indices_19 += indices_1[-diff:] # add the remaining part
    else:
        diff = len(indices_9) - len(indices_1)
        indices_19[::2] = indices_1
        indices_19[1::2] = indices_9[:-diff]
        indices_19 += indices_9[-diff:] # add the remaining part
    x_train_19, y_train_19 = x_train[indices_19], y_train[indices_19]
    x_train_red, y_train_red = np.delete(x_train, indices_19, 0), np.delete(y_train, indices_19, 0)
    x_train_C = np.vstack([x_train_19, x_train_red])
    y_train_C = np.vstack([y_train_19, y_train_red])

    # run the model
    model.fit(x_train_C, y_train_C,
                batch_size=BATCH,
                epochs=EPOCHS,
                validation_data=(x_valid, y_valid),
                verbose=0)

    predictions =  model.predict(x_test)
    conf_mat_C = gen_confusion_mat(y_test, predictions)
    np.savetxt('conf_mat_B.csv', conf_mat_C, fmt='%i', delimiter=',')
    acc_C = calc_acc(conf_mat_C)
    print('accuracy for run B: \n ', acc_C)
    recall_C = calc_recall(conf_mat_C, labels)
    print('recalls for run B: \n ', recall_C)

    opt_C_dict[i] = (acc_C, recall_C, conf_mat_C)



    # D. Train with 3 and 5 alternated (since both a confused with each other)

    # E.


############# SAVE RESUTLS ###################
import pickle

with open('./save_folder/base_dict.pickle', 'wb') as f:
    pickle.dump(base_dict, f)

with open('./save_folder/opt_B_dict.pickle', 'wb') as f:
    pickle.dump(opt_B_dict, f)

with open('./save_folder/opt_C_dict.pickle', 'wb') as f:
    pickle.dump(opt_C_dict, f)

############# ANALYZE THE RESULTS #############
## Average values
avg_recall_base = np.mean(np.vstack([np.array(list(base_dict[key][1].values())) for key in base_dict.keys()]),0)
avg_recall_B = np.mean(np.vstack([np.array(list(opt_B_dict[key][1].values())) for key in opt_B_dict.keys()]),0)
avg_recall_C = np.mean(np.vstack([np.array(list(opt_C_dict[key][1].values())) for key in opt_C_dict.keys()]),0)

avg_recall_base

avg_recall_B

avg_recall_C
