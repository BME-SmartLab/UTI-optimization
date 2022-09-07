'''
Written by Tamas Gabor Csapo <csapot@tmit.bme.hu>
First version Nov 9, 2016
Restructured Feb 4, 2018 - get data
Restructured Sep 19, 2018 - DNN training
Restructured Oct 13, 2018 - CNN training
Restructured Feb 11, 2020 - restructure for data generator
Restructured May 4, 2020 - ResNet-50 experiments

Keras implementation of the UTI raw (with shifts/padding for square shape) representation of
Tamas Gabor Csapo, Gabor Gosztolya, Laszlo Toth, Amin Honarmandi Shandiz, Alexandra Marko,
,,Optimizing the Ultrasound Tongue Image Representation for Residual Network-based Articulatory-to-Acoustic Mapping'', submitted.
'''

import matplotlib
matplotlib.use('agg')


import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as io_wav
from detect_peaks import detect_peaks
import os
import os.path
import gc
import re
import tgt
import csv
import datetime
import scipy
import pickle
import skimage
import cv2
import random
random.seed(17)


from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dropout
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint

# additional requirement: SPTK 3.8 or above in PATH

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ResNet model
from residualnetworks import ResNet50_regression

# do not use all GPU memory
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True 
set_session(tf.Session(config=config))



# read_ult reads in *.ult file from AAA
def read_ult(filename, NumVectors, PixPerVector):
    # read binary file
    ult_data = np.fromfile(filename, dtype='uint8')
    ult_data = np.reshape(ult_data, (-1, NumVectors, PixPerVector))
    return ult_data


# read_psync_and_correct_ult reads *_sync.wav and finds the rising edge of the pulses
# if there was a '3 pulses bug' during the recording,
# it removes the first three frames from the ultrasound data
def read_psync_and_correct_ult(filename, ult_data):
    (Fs, sync_data_orig) = io_wav.read(filename)
    sync_data = sync_data_orig.copy()

    # clip
    sync_threshold = np.max(sync_data) * 0.6
    for s in range(len(sync_data)):
        if sync_data[s] > sync_threshold:
            sync_data[s] = sync_threshold

    # find peeks
    peakind1 = detect_peaks(sync_data, mph=0.9*sync_threshold, mpd=10, threshold=0, edge='rising')
    
    '''
    # figure for debugging
    plt.figure(figsize=(18,4))
    plt.plot(sync_data)
    plt.plot(np.gradient(sync_data), 'r')
    for i in range(len(peakind1)):
        plt.plot(peakind1[i], sync_data[peakind1[i]], 'gx')
        # plt.plot(peakind2[i], sync_data[peakind2[i]], 'r*')
    plt.xlim(2000, 6000)
    plt.show()    
    '''
    
    # this is a know bug: there are three pulses, after which there is a 2-300 ms silence, 
    # and the pulses continue again
    if (np.abs( (peakind1[3] - peakind1[2]) - (peakind1[2] - peakind1[1]) ) / Fs) > 0.2:
        bug_log = 'first 3 pulses omitted from sync and ultrasound data: ' + \
            str(peakind1[0] / Fs) + 's, ' + str(peakind1[1] / Fs) + 's, ' + str(peakind1[2] / Fs) + 's'
        print(bug_log)
        
        peakind1 = peakind1[3:]
        ult_data = ult_data[3:]
    
    for i in range(1, len(peakind1) - 2):
        # if there is a significant difference between peak distances, raise error
        if np.abs( (peakind1[i + 2] - peakind1[i + 1]) - (peakind1[i + 1] - peakind1[i]) ) > 1:
            bug_log = 'pulse locations: ' + str(peakind1[i]) + ', ' + str(peakind1[i + 1]) + ', ' +  str(peakind1[i + 2])
            print(bug_log)
            bug_log = 'distances: ' + str(peakind1[i + 1] - peakind1[i]) + ', ' + str(peakind1[i + 2] - peakind1[i + 1])
            print(bug_log)
            
            raise ValueError('pulse sync data contains wrong pulses, check it manually!')
    
    return ([p for p in peakind1], ult_data)



def get_training_data(dir_file, filename_no_ext, NumVectors = 64, PixPerVector = 842):
    print('starting ' + dir_file + filename_no_ext)

    # read in raw ultrasound data
    ult_data = read_ult(dir_file + filename_no_ext + '.ult', NumVectors, PixPerVector)
    
    try:
        # read pulse sync data (and correct ult_data if necessary)
        (psync_data, ult_data) = read_psync_and_correct_ult(dir_file + filename_no_ext + '_sync.wav', ult_data)
    except ValueError as e:
        raise
    else:
        
        # works only with 22kHz sampled wav
        (Fs, speech_wav_data) = io_wav.read(dir_file + filename_no_ext + '_speech_volnorm.wav')
        assert Fs == 22050

        mgc_lsp_coeff = np.fromfile(dir_file + filename_no_ext + '_speech_volnorm_cut_ultrasound.mgclsp', dtype=np.float32).reshape(-1, order + 1)
        lf0 = np.fromfile(dir_file + filename_no_ext + '_speech_volnorm_cut_ultrasound.lf0', dtype=np.float32)

        (mgc_lsp_coeff_length, _) = mgc_lsp_coeff.shape
        (lf0_length, ) = lf0.shape
        assert mgc_lsp_coeff_length == lf0_length

        # cut from ultrasound the part where there are mgc/lf0 frames
        ult_data = ult_data[0 : mgc_lsp_coeff_length]

        # read phones from TextGrid
        tg = tgt.io.read_textgrid(dir_file + filename_no_ext + '_speech.TextGrid')
        tier = tg.get_tier_by_name(tg.get_tier_names()[0])

        tg_index = 0
        phone_text = []
        for i in range(len(psync_data)):
            # get times from pulse synchronization signal
            time = psync_data[i] / Fs

            # get current textgrid text
            if (tier[tg_index].end_time < time) and (tg_index < len(tier) - 1):
                tg_index = tg_index + 1
            phone_text += [tier[tg_index].text]

        # add last elements to phone list if necessary
        while len(phone_text) < lf0_length:
            phone_text += [phone_text[:-1]]

        print('finished ' + dir_file + filename_no_ext + ', altogether ' + str(lf0_length) + ' frames')

        plt.imshow(ult_data[0])

        return (ult_data, mgc_lsp_coeff, lf0, phone_text)

def get_mgc_lf0(dir_file, filename_no_ext):
    
    mgc_lsp_coeff = np.fromfile(dir_file + filename_no_ext + '_speech_volnorm_cut_ultrasound.mgclsp', dtype=np.float32).reshape(-1, order + 1)
    lf0 = np.fromfile(dir_file + filename_no_ext + '_speech_volnorm_cut_ultrasound.lf0', dtype=np.float32)

    (mgc_lsp_coeff_length, _) = mgc_lsp_coeff.shape
    (lf0_length, ) = lf0.shape
    assert mgc_lsp_coeff_length == lf0_length

    return (mgc_lsp_coeff, lf0)


# from LipReading with slight modifications
# https://github.com/hassanhub/LipReading/blob/master/codes/data_integration.py
################## VIDEO INPUT ##################
def load_video_3D_fc1(path, framesPerSec):
    
    cap = cv2.VideoCapture(path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # make sure that all the videos are the same FPS
    if (np.abs(fps - framesPerSec) > 0.2):
        print('fps:', fps, '(' + path + ')')
        raise

    buf = np.empty((frameCount, frameHeight, frameWidth), np.dtype('float32'))
    fc = 0
    ret = True
    
    while (fc < frameCount  and ret):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # scaling to [0-1]
        buf[fc] = frame.astype('float32') / 255
        fc += 1
    cap.release()

    return buf

def ult_mgc_generator(ult_files_all, mgc_scalers, batch_size, mode="train"):
    ult_batch = np.empty((batch_size, n_width, n_height, 1))
    mgc_batch = np.empty((batch_size, n_mgc))
    i_batch = 0
    epoch = 0
    # loop indefinitely
    while True:
        print('ult_mgc_seq_generator: loop start', mode)
        
        # randomize the order of files
        random.shuffle(ult_files_all)
        
        print('ult_mgc_seq_generator', mode, ult_files_all)
        
        # iterate through ult files
        for n in range(len(ult_files_all)):
            file = ult_files_all[n]
            print('\n ult_mgc_seq_generator: reading file ...', \
                    str(int(100*n/len(ult_files_all)))+'%', mode, file[:-15] + '.ult', end='\r')
            
            try:
                (ult_data, mgc_lsp_coeff, lf0, phone_text) = get_training_data('', file[:-15])
            except ValueError as e:
                print("wrong psync data, check manually!")
            else:
                # shuffle data for CNN
                permutation = np.random.permutation(ult_data.shape[0])
                np.take(ult_data, permutation, axis=0, out=ult_data)
                np.take(mgc_lsp_coeff, permutation, axis=0, out=mgc_lsp_coeff)
                
                for i in range(len(ult_data)):
                    ult_img = ult_data[i]
                    
                    # zero padding for squared size
                    n_height_pad = int(256 * 256 / n_lines)
                    
                    ult_padding = np.zeros((n_lines, n_height_pad))
                    ult_padding[0 : n_lines, 0 : n_pixels] = ult_img
                    
                    # reshape / shift / pad for square size
                    
                    ult_img_padding = np.zeros((256, 256))
                    
                    ult_img_padding[0:64, :] = ult_padding[0:64, 0:256]
                    ult_img_padding[64:128, :] = ult_padding[0:64, 256:512]
                    ult_img_padding[128:192, :] = ult_padding[0:64, 512:768]
                    ult_img_padding[192:256, :] = ult_padding[0:64, 768:1024]
                    
                    # resize to smaller area
                    ult_img_padding = skimage.transform.resize(ult_img_padding, (n_width, n_height), \
                        preserve_range=True, anti_aliasing=True, mode='constant')
                    
                    # reshape for CNN
                    ult_batch[i_batch] = np.reshape(ult_img_padding, (n_width, n_height, 1))
                    
                    
                    # temp
                    if not os.path.isfile('img/UTIraw-padding_SSI_ResNet_datagen_' + \
                            str(n_width) + 'x' + str(n_height) + '_' + speaker + '.png'):
                        plt.imshow(ult_batch[i_batch].squeeze(), cmap='gray')
                        plt.savefig('img/UTIraw-padding_SSI_ResNet_datagen_' + \
                            str(n_width) + 'x' + str(n_height) + '_' + speaker + '.png')
                        # plt.close()
                        
                        # raise
                    
                    mgc_batch[i_batch] = mgc_lsp_coeff[i]
                    
                    i_batch += 1
                    
                    if i_batch == batch_size:
                        # input data to [0-1]
                        ult_batch /= 255
                    
                        # target: normalization to zero mean, unit variance feature by feature
                        for j in range(n_mgc):
                            mgc_batch[:, j] = mgc_scalers[j].transform(mgc_batch[:, j].reshape(-1, 1)).ravel()
                        
                        # yield the batch to the calling function
                        yield (ult_batch, mgc_batch)

                        # empty the buffer
                        ult_batch = np.empty((batch_size, n_width, n_height, 1))
                        mgc_batch = np.empty((batch_size, n_mgc))
                        i_batch = 0

            if n == len(ult_files_all) - 1:
                print('\n\n ult_mgc_seq_generator: end of ult_files', mode, ', epoch ' + str(epoch+1) + '\n\n')
                
                # restart reading files
                n = 0
                
                # randomize again file order
                random.shuffle(ult_files_all)
                
                # empty the buffer
                ult_batch = np.empty((batch_size, n_width, n_height, 1))
                mgc_batch = np.empty((batch_size, n_mgc))
                i_batch = 0
                
                


# Parameters of vocoder
frameLength = 512 # 23 ms at 22050 Hz sampling
frameShift = 270 # 12 ms at 22050 Hz sampling, correspondong to 81.5 fps (ultrasound)
order = 24
alpha = 0.42
stage = 3
n_contf0_mvf = 2
n_mgc = order + 1

# parameters of ultrasound images
framesPerSec = 81.67
type = 'PPBA' # the 'PPBA' directory can be used for training data
n_files = 200
n_lines = 64
n_pixels = 842



# TODO: modify this according to your data path
dir_base = "/shared/data_SSI2018/"
##### training data
# - 2 females: spkr048, spkr049
# - 5 males: spkr010, spkr102, spkr103, spkr104, spkr120

# train on 1 speaker
speakers = ['spkr048']

for n_width in [512, 256, 128, 64, 32, 16, 8]:
    n_height = n_width
    
    for speaker in speakers:
        n_file = 0
        n_max_ultrasound_frames = n_files * 500

        mgc = np.empty((n_max_ultrasound_frames, n_mgc))
        ult_size = 0
        mgc_size_train = 0
        mgc_size_valid = 0
        
        # collect all possible ult files
        ult_files_all = []
        dir_data = dir_base + speaker + "/" + type + "/"
        if os.path.isdir(dir_data):
            for file in sorted(os.listdir(dir_data)):
                if file.endswith('_ultrasound.mp4'):
                    ult_files_all += [dir_data + file]
        
        # randomize the order of files
        random.shuffle(ult_files_all)
        
        # temp: only first 10 sentence
        # ult_files_all = ult_files_all[0:10]

        
        # train: first 90% of sentences
        ult_files_all_train = ult_files_all[0:int(0.9*len(ult_files_all))]
        # valid: last 10% of sentences
        ult_files_all_valid = ult_files_all[int(0.9*len(ult_files_all)):]
        
        # load all training data - only MGC-LSP
        for file in ult_files_all_train:
            mgc_lsp_coeff = np.fromfile(file[:-15] + '_speech_volnorm_cut_ultrasound.mgclsp', \
                dtype=np.float32).reshape(-1, n_mgc)
            mgc_len = len(mgc_lsp_coeff)
            mgc[mgc_size_train : mgc_size_train + mgc_len] = mgc_lsp_coeff
            mgc_size_train += mgc_len
            print('n_frames_all (train): ', mgc_size_train)
        mgc = mgc[0 : mgc_size_train]

        # remaining 10% for validation
        for file in ult_files_all_valid:
            mgc_lsp_coeff = np.fromfile(file[:-15] + '_speech_volnorm_cut_ultrasound.mgclsp', \
                dtype=np.float32).reshape(-1, n_mgc)
            mgc_size_valid += len(mgc_lsp_coeff)
            print('n_frames_all (valid): ', mgc_size_valid)
        
        # input: already scaled to [0,1] range
        
        # target: normalization to zero mean, unit variance
        # feature by feature
        mgc_scalers = []
        for i in range(n_mgc):
            mgc_scaler = StandardScaler(with_mean=True, with_std=True)
            mgc_scalers.append(mgc_scaler)
            mgc_scalers[i].fit(mgc[:, i].reshape(-1, 1))
        
        if n_width in [512]:
            GENERATOR_BATCH = 2
        else:
            GENERATOR_BATCH = 64
        
        NUM_TRAIN_IMAGES = mgc_size_train
        NUM_VALID_IMAGES = mgc_size_valid
        
        # initialize both the training and validation image generators
        trainGen = ult_mgc_generator(ult_files_all_train, mgc_scalers, GENERATOR_BATCH, mode="train")
        validGen = ult_mgc_generator(ult_files_all_valid, mgc_scalers, GENERATOR_BATCH, mode="valid")

        
        # get ResNet model
        model = ResNet50_regression(input_shape = (n_width, n_height, 1), n_output = n_mgc)

        # compile model
        model.compile(loss='mean_squared_error', optimizer='adam')

        print(model.summary())
        

        # early stopping to avoid over-training
        # csv logger
        current_date = '{date:%Y-%m-%d_%H-%M-%S}'.format( date=datetime.datetime.now() )
        model_name = 'models/UTIraw-padding_SSI_ResNet_datagen_' + \
            str(n_width) + 'x' + str(n_height) + '_' + speaker + '_' + current_date

        print(current_date)
        callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=0), \
                     CSVLogger(model_name + '.csv', append=True, separator=';'), \
                     ModelCheckpoint(model_name + '_weights_best.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')]

        # save model
        model_json = model.to_json()
        with open(model_name + '_model.json', "w") as json_file:
            json_file.write(model_json)

        # serialize scalers to pickle
        pickle.dump(mgc_scalers, open(model_name + '_mgc_scalers.sav', 'wb'))

        
        # Run iterative training
        history = model.fit_generator(
            trainGen,
            steps_per_epoch=NUM_TRAIN_IMAGES // GENERATOR_BATCH,
            epochs=100,
            validation_data=validGen,
            validation_steps=NUM_VALID_IMAGES // GENERATOR_BATCH,
            verbose = 1,
            callbacks=callbacks)
        
        # here the training of ResNet is finished

