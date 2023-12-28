import os
import numpy as np
import json
import pywt # wavelet
import mne # (for .edf file of EEG data) or import pyedflib
from scipy import signal, fft
import matplotlib.pyplot as plt
import random
from sklearn.metrics import classification_report, auc




def read_label(label_dir, read_all_classes):
    seizure_dict = {}
    c = 0

    for path in os.listdir(label_dir): # each .json
        label_path = label_dir + path

        ##### json -> add label
        print(label_path)
        with open(label_path, 'r') as f: # read .json
            labels_ = json.load(f)
            labels = labels_['eeg']['annotations']
            for label in labels: # each seizure
                if label['type'] in read_all_classes:
                    seizure_dict[c] = {'file': path[:-9], 'start': label['start'], 'end': label['end'], 'type': label['type']}
                    c += 1
    
    return seizure_dict




def read_data(data_dir, lowcut, highcut, channel_indices, scaling_factor):
    data_dict = {}

    for path in os.listdir(data_dir): # each .edf
        data_path = data_dir + path

        ##### edf -> add data
        print(data_path)
        raw_eeg = mne.io.read_raw_edf(data_path, preload=True, verbose=False) # read .edf
        raw_eeg.filter(lowcut, highcut, method='iir') # band-pass filter
        data = raw_eeg.get_data().astype(np.float32) # data, times = raw_eeg[:, :]

        data_list = []
        for channel_index in channel_indices: # use 18 channels
            data_list.append(data[channel_index[1]] - data[channel_index[0]]) # subtraction 

        data_list = np.array(data_list)
        data = data_list.T * scaling_factor # μV -> pV (1e-12 V)
        data_dict[path[:-4]] = {'data': data} # data_dict[path[:-4]] = {'data': data, 'time': times} 
    
    return data_dict




def band_pass(data_df, lowcut, highcut, sampling_rate, order=4):
    for i in range(len(data_df)): # each edf file
        data_filtered = []
        for j in range(data_df['data'][i].shape[1]): # each channel
            # original
            t = data_df['time'][i] # x-axis: time
            d = data_df['data'][i][:, j] # y-axis: amplitude
            # plt.figure('original')
            # plt.title('original')
            # plt.plot(t, d)
            # plt.show()
                
            # filter coefficient
            nyquist = 0.5 * sampling_rate
            low = lowcut / nyquist
            high = highcut / nyquist

            # filtered
            b, a = signal.butter(order, [low, high], btype='band') # lowpass, highpass, or band
            d_filtered = signal.lfilter(b, a, d) # signal.lfilter or signal.filtfilt
            # plt.figure('filtered')
            # plt.title('filtered')
            # plt.plot(t, d_filtered)
            # plt.show()

            data_filtered.append(d_filtered)

        data_filtered = np.array(data_filtered).T
        data_df['data'][i] = data_filtered # new data

        return data_df




def viz_original(file, seizure_df, data_df, id_channel, times):

    data = data_df[data_df['file']==file]['data'].values[0] [500*3600*0: 500*3600*1] # just 1h of 12h
    time = times [500*3600*0: 500*3600*1] # just 1h of 12h

    print('----------All time: non-seizure(black), seizure(red) ----------')
    plt.plot(time, data[:, id_channel], c='k') # seizure + non-seizure

    seizure_index_list = list(zip(seizure_df[seizure_df['file']==file]['start'], seizure_df[seizure_df['file']==file]['end']))
    for s, e in seizure_index_list: # for adjust 
        # print(s, e) # seizure range
        plt.plot(time[s:e], data[:, id_channel][s:e], c='r') # seizure
    plt.xlabel('time [s]')
    plt.ylabel('Voltage [pV]')
    plt.show()

    # print('----------Detailed: non-seizure(black), seizure(red) ----------')
    # seizure_index_list = list(zip(seizure_df[seizure_df['file']==file]['start'], seizure_df[seizure_df['file']==file]['end']))
    # # print([(i[0]/500, i[1]/500) for i in seizure_index_list])
    # for s, e in seizure_index_list[2:6]: # for adjust 
    #     # print('seiz', s/500, e/500) # seizure range
    #     plt.plot(time[s:e], data[:, id_channel][s:e], c='r') # seizure

    # for i in range(2, 6): # for adjust 
    #     s, e = seizure_index_list[i][1], seizure_index_list[i+1][0]
    #     # print('non-seiz', s/500, e/500) # non-seizure range
    #     plt.plot(time[s:e], data[:, id_channel][s:e], c='k') # non-seizure

    # plt.xlabel('time [s]')
    # plt.ylabel('Voltage [pV]')
    # plt.show()
    



def viz_dwt(file, data_df, id_channel, level, wavelet, times):
    ##### original
    data = data_df[data_df['file']==file]['data'].values[0] [500*0: 500*1] # just 1s
    time = times [500*0: 500*1] # just 1s
    d = data[:, id_channel] # 1 file, 1 channel
    print('sum number:', d.shape)

    fontsize = 15
    ##### original
    plt.figure(figsize=(10, 60))
    plt.subplot(level+2, 1, 1)
    # plt.plot(time, d) # x-axis: sample point
    plt.scatter(np.linspace(0, 1, len(d)), d) # x-axis: second
    plt.title('Original Signals', fontsize=fontsize)
    plt.xlabel('time [s]', fontsize=fontsize)
    plt.ylabel('Voltage [pV]', fontsize=fontsize)

    ##### DWT
    coeffs = pywt.wavedec(data=d, wavelet=wavelet, level=level)

    ##### approximate coefficient
    approximation = coeffs[0]
    print('length of approximate coefficients: {}'.format(approximation.shape))
    plt.subplot(level+2, 1, 2)
    # plt.plot(approximation) # x-axis: sample point
    plt.scatter(np.linspace(0, 1, len(approximation)), approximation) # x-axis: second
    plt.title('Approximate Coefficients', fontsize=fontsize)
    plt.xlabel('time [s]', fontsize=fontsize)
    plt.ylabel('Voltage [pV]', fontsize=fontsize)

    ##### detailed coefficient
    details = coeffs[1:] 
    for i_, detail in enumerate(details):
        print('length of detailed coefficients (level {}): {}'.format(level-i_, detail.shape))
        plt.subplot(level+2, 1, i_+3)
        # plt.plot(detail) # x-axis: sample point
        plt.scatter(np.linspace(0, 1, len(detail)), detail) # x-axis: second
        plt.title('Detailed Coefficients (level {})'.format(level-i_), fontsize=fontsize)
        plt.xlabel('time [s]', fontsize=fontsize)
        plt.ylabel('Voltage [pV]', fontsize=fontsize)
    
    plt.show()




def dwt_seizure_per_window(seizure_df, data_df, window_size, seizure_classes, sampling_rate, wavelet, level):

    coeffs_all_channels_all_steps_all_seizure = []    
    for i in range(len(seizure_df)): ##### each seizure
        # print('----------------------------------------')
        # print('seizure id:', i)

        if seizure_df['type'][i] in seizure_classes: # 只要 ['rhythmic', 'seiz', 'single'], 而不要 ['other']
            s = seizure_df['start'][i]
            e = seizure_df['end'][i]
            seizure_time = int( (e - s) / sampling_rate )

            if seizure_time >= window_size: # 只有大于 window_size(e.g. 1[s] or 2[s]) 的才能执行
                
                coeffs_all_channels_all_steps = []
                for j in range(int(seizure_time/window_size)): ##### each time step, e.g. 1[s] or 2[s] (只要整数部分, 约等于整几秒, e.g. 1.524[s] -> 1[s])
                    # print('--------------------')
                    # print('step id:', j)

                    coeffs_all_channels = []
                    for k in range(18): ##### each channel
                        # print('----------')
                        # print('channel id:', k)

                        ##### original
                        d_ = data_df[data_df['file']==seizure_df['file'][i]]['data'].values[0][:, k][s + j*window_size*sampling_rate : s + (j+1)*window_size*sampling_rate] ##### seizure

                        ##### DWT
                        coeffs_ = pywt.wavedec(data=d_, wavelet=wavelet, level=level) ##### seizure
                        coeffs_.append(0) # add label 0

                        coeffs_all_channels.append(coeffs_)

                    coeffs_all_channels_all_steps.append(coeffs_all_channels)

                coeffs_all_channels_all_steps = np.array(coeffs_all_channels_all_steps) # (step num, channel num, coeff num), e.g. (3, 18, 8) or (1, 18, 8)
                coeffs_all_channels_all_steps_all_seizure.append(coeffs_all_channels_all_steps)
            else:
                continue

    coeffs_seizure = np.concatenate(coeffs_all_channels_all_steps_all_seizure, axis=0) 

    return coeffs_seizure




def dwt_seizure_per_overlap_window(seizure_df, data_df, window_size, seizure_classes, sampling_rate, wavelet, level, moving_size):

    coeffs_all_channels_all_steps_all_seizure = []    
    for i in range(len(seizure_df)): ##### each seizure
        # print('----------------------------------------')
        # print('seizure id:', i)

        if seizure_df['type'][i] in seizure_classes: # 只要 ['rhythmic', 'seiz', 'single'], 而不要 ['other']
            s = seizure_df['start'][i]
            e = seizure_df['end'][i]
            seizure_time = (e - s) / sampling_rate

            if seizure_time >= window_size: # 只有大于 window_size(e.g. 1[s] or 2[s]) 的才能执行

                coeffs_all_channels_all_steps = []
                for j in np.arange(0, seizure_time, moving_size): ##### each time step, e.g. 1[s] or 2[s]

                    if j + window_size <= seizure_time: # 只有未超出癫痫结束时间的才能执行 
                        # print('--------------------')
                        # print('step id:', j)

                        coeffs_all_channels = []
                        for k in range(18): ##### each channel
                            # print('----------')
                            # print('channel id:', k)

                            ##### original
                            d_ = data_df[data_df['file']==seizure_df['file'][i]]['data'].values[0] [:, k] [int(s + j*sampling_rate) : int(s + (j+window_size)*sampling_rate)] ##### seizure

                            ##### DWT
                            coeffs_ = pywt.wavedec(data=d_, wavelet=wavelet, level=level) ##### seizure
                            coeffs_.append(0) # add label 0

                            coeffs_all_channels.append(coeffs_)

                        coeffs_all_channels_all_steps.append(coeffs_all_channels)

                coeffs_all_channels_all_steps = np.array(coeffs_all_channels_all_steps) # (step num, channel num, coeff num), e.g. (3, 18, 8) or (1, 18, 8)
                coeffs_all_channels_all_steps_all_seizure.append(coeffs_all_channels_all_steps)
            else:
                continue

    coeffs_seizure = np.concatenate(coeffs_all_channels_all_steps_all_seizure, axis=0) 

    return coeffs_seizure




def dwt_seizure_per_overlap_window_CHB(df, window_size, sampling_rate, wavelet, level, moving_size):
    coeffs_all_channels_all_steps_all_seizure = []

    for i in df.index: ##### each seizure (each edf file)
        # print('----------------------------------------')
        # print('seizure id:', i)

        s = df['start_time'][i]
        e = df['end_time'][i]
        seizure_time = e - s # [s]

        coeffs_all_channels_all_steps = []
        for j in np.arange(0, seizure_time, moving_size): ##### each time step, e.g. 1[s] or 2[s]

            if j + window_size <= seizure_time: # 只有未超出癫痫结束时间的才能执行 
                # print('--------------------')
                # print('step id:', j)

                coeffs_all_channels = []
                for k in range(23): ##### each channel
                    # print('----------')
                    # print('channel id:', k)

                    ##### original
                    d_ = df['data'][i] [:, k] [int((s + j)*sampling_rate) : int((s + j + window_size)*sampling_rate)] ##### seizure

                    ##### DWT
                    coeffs_ = pywt.wavedec(data=d_, wavelet=wavelet, level=level) ##### seizure
                    coeffs_.append(0) # add label 0

                    coeffs_all_channels.append(coeffs_)

                coeffs_all_channels_all_steps.append(coeffs_all_channels)

        coeffs_all_channels_all_steps = np.array(coeffs_all_channels_all_steps) # (step num, channel num, coeff num), e.g. (40, 23, 7) or (27, 23, 7)
        coeffs_all_channels_all_steps_all_seizure.append(coeffs_all_channels_all_steps)

    coeffs_seizure = np.concatenate(coeffs_all_channels_all_steps_all_seizure, axis=0) 

    return coeffs_seizure




def random_non_seizure_range(total_range, seizure_ranges, window_len): 
    '''
    total_range: 整个数据的大范围,  e.g. (0, 20)
    seizure_ranges: 每个癫痫的小范围, e.g. [(2, 3), (10, 12)]
    window_len: 要选出的癫痫范围的长度, e.g. 5
    '''
    while True:
        s_ = random.randint(total_range[0], total_range[1]-1-window_len) # 随机一个起点
        e_ = s_ + window_len # 计算对应终点
        candidate_range = (s_, e_) # 候选范围
        
        overlap = False # 下面开始循环判断随机的候选范围是否覆盖了任意一个癫痫范围
        for seizure_range in seizure_ranges:
            if candidate_range[0] <= seizure_range[1] and candidate_range[1] >= seizure_range[0]:
                overlap = True # 如果一旦有覆盖，就立刻重新while
                break

        # 如果没有与任意一个癫痫范围覆盖，则选中它并结束while
        if not overlap:
            return candidate_range
            break




def dwt_non_seizure_per_window_one_file(seizure_sum_time, data_df, seizure_df, seizure_classes, window_size, random_non_seizure_range, sampling_rate, wavelet, level):
    # print('total seizure time: {} [s]'.format(seizure_sum_time))

    file = random.choice(data_df['file']) # 随机选择一个文件
    data = data_df[data_df['file']==file]['data'].values[0] # 取出该文件的全部数据

    s = seizure_df[seizure_df['file']==file]['start'] # non-seizure not include other
    e = seizure_df[seizure_df['file']==file]['end']
    # s = seizure_df[(seizure_df['file']==file) & (seizure_df['type'].isin(seizure_classes))]['start'] # non-seizure include other
    # e = seizure_df[(seizure_df['file']==file) & (seizure_df['type'].isin(seizure_classes))]['end']
    
    seizure_ranges = [] # 获取全部癫痫的范围
    for seizure_range in zip(s, e): ####
        seizure_ranges.append(seizure_range)

    coeffs_all_channels_all_steps = []
    for j in range(seizure_sum_time): # each time step, e.g. 1[s] or 2[s] 
        non_seizure_range = random_non_seizure_range((0, len(data)), seizure_ranges, window_size*sampling_rate) # 调用函数随机选出非癫痫的范围

        coeffs_all_channels = []
        for k in range(18): # each channel
            #### original
            d_ = data[non_seizure_range[0]:non_seizure_range[1], k]

            #### DWT
            coeffs_ = pywt.wavedec(data=d_, wavelet=wavelet, level=level) ##### non-seizure
            coeffs_.append(1) # add label 1

            coeffs_all_channels.append(coeffs_)

        coeffs_all_channels_all_steps.append(coeffs_all_channels)

    coeffs_all_channels_all_steps = np.array(coeffs_all_channels_all_steps) # (step num, channel num, coeff num), e.g. (204, 18, 8)
    coeffs_non_seizure = coeffs_all_channels_all_steps

    return coeffs_non_seizure




def dwt_non_seizure_per_window_all_files(seizure_sum_time, data_df, seizure_df, seizure_classes, window_size, random_non_seizure_range, sampling_rate, wavelet, level):
    # print('total seizure time: {} [time step]'.format(seizure_sum_time))

    # s_all = seizure_df['start'] # non-seizure not include other
    # e_all = seizure_df['end']
    s_all = seizure_df[seizure_df['type'].isin(seizure_classes)]['start'] # non-seizure include other
    e_all = seizure_df[seizure_df['type'].isin(seizure_classes)]['end']
    num_seizure_points_all_files = sum(e_all-s_all) # 计算原始的所有文件的癫痫点数总和

    lll = []
    coeffs_all_channels_all_steps_all_file = []
    for i, file in enumerate(data_df['file']):
        # print(file)

        # s = seizure_df[seizure_df['file']==file]['start'] # non-seizure not include other
        # e = seizure_df[seizure_df['file']==file]['end']
        s = seizure_df[(seizure_df['file']==file) & (seizure_df['type'].isin(seizure_classes))]['start'] # non-seizure include other
        e = seizure_df[(seizure_df['file']==file) & (seizure_df['type'].isin(seizure_classes))]['end']

        ratio = sum(e-s) / num_seizure_points_all_files # 计算每个原始的文件占所有文件的比例
        seizure_sum_time_per_file = ratio * seizure_sum_time # 计算新的每个文件的癫痫点数总和 = 上述比例* 要压缩到的新的和

        if i != len(data_df['file'])-1:
            seizure_sum_time_per_file = int(np.floor(seizure_sum_time_per_file)) # 向下取整
            lll.append(seizure_sum_time_per_file)
        else:
            seizure_sum_time_per_file = seizure_sum_time - sum(lll) # 只有当最后一个文件时，补回来之前所有向下取整的小数，为了保持总和不变

        seizure_ranges = [] # 获取全部癫痫的范围
        for seizure_range in zip(s, e): ####
            seizure_ranges.append(seizure_range)
        # print(seizure_ranges)

        data = data_df[data_df['file']==file]['data'].values[0] # 取出该文件的全部数据
        coeffs_all_channels_all_steps = []
        for j in range(seizure_sum_time_per_file): ##### each time step, e.g. 1[s] or 2[s] (按照每个文件的癫痫的数据量比例来采样每个非癫痫的文件)
            non_seizure_range = random_non_seizure_range(total_range=(0, len(data)), seizure_ranges=seizure_ranges, window_len=window_size*sampling_rate) # 调用函数随机选出非癫痫的范围
            # print(non_seizure_range)

            coeffs_all_channels = []
            for k in range(data.shape[1]): ##### each channel
                #### original
                d_ = data[non_seizure_range[0]:non_seizure_range[1], k]

                #### DWT
                coeffs_ = pywt.wavedec(data=d_, wavelet=wavelet, level=level) ##### non-seizure
                coeffs_.append(1) # add label 1

                coeffs_all_channels.append(coeffs_)

            coeffs_all_channels_all_steps.append(coeffs_all_channels)

        coeffs_all_channels_all_steps = np.array(coeffs_all_channels_all_steps) # (step num, channel num, coeff num), e.g. (204, 18, 8)
        coeffs_all_channels_all_steps_all_file.append(coeffs_all_channels_all_steps)

        coeffs_non_seizure = np.concatenate(coeffs_all_channels_all_steps_all_file, axis=0) 

    return coeffs_non_seizure




def dwt_non_seizure_per_window_all_files_CHB(seizure_sum_time, df, window_size, random_non_seizure_range, sampling_rate, wavelet, level):
    # print('total seizure time: {} [s]'.format(seizure_sum_time))

    s_all = df['start_time']
    e_all = df['end_time']
    num_seizure_points_all_files = sum(e_all-s_all) # 计算原始的所有文件的癫痫点数总和

    lll = []
    coeffs_all_channels_all_steps_all_file = []
    for i, file in enumerate(df['file_path']):
        # print(file)

        s = df[df['file_path']==file]['start_time']
        e = df[df['file_path']==file]['end_time']

        ratio = sum(e-s) / num_seizure_points_all_files # 计算每个原始的文件占所有文件的比例
        seizure_sum_time_per_file = ratio * seizure_sum_time # 计算新的每个文件的癫痫点数总和 = 上述比例* 要压缩到的新的和

        if i != len(df['file_path'])-1:
            seizure_sum_time_per_file = int(np.floor(seizure_sum_time_per_file)) # 向下取整
            lll.append(seizure_sum_time_per_file)
        else:
            seizure_sum_time_per_file = seizure_sum_time - sum(lll) # 只有当最后一个文件时，补回来之前所有向下取整的小数，为了保持总和不变

        seizure_ranges = [] # 获取全部癫痫的范围
        for seizure_range in zip(s, e): ####
            seizure_ranges.append(seizure_range)
        # print(seizure_ranges)

        data = df[df['file_path']==file]['data'].values[0] # 取出该文件的全部数据
        coeffs_all_channels_all_steps = []
        for j in range(seizure_sum_time_per_file): ##### each time step, e.g. 1[s] or 2[s] (按照每个文件的癫痫的数据量比例来采样每个非癫痫的文件)
            non_seizure_range = random_non_seizure_range(total_range=(0, len(data)), seizure_ranges=seizure_ranges, window_len=window_size*sampling_rate) # 调用函数随机选出非癫痫的范围
            # print(non_seizure_range)

            coeffs_all_channels = []
            for k in range(data.shape[1]): ##### each channel
                #### original
                d_ = data[non_seizure_range[0]:non_seizure_range[1], k]

                #### DWT
                coeffs_ = pywt.wavedec(data=d_, wavelet=wavelet, level=level) ##### non-seizure
                coeffs_.append(1) # add label 1

                coeffs_all_channels.append(coeffs_)

            coeffs_all_channels_all_steps.append(coeffs_all_channels)

        coeffs_all_channels_all_steps = np.array(coeffs_all_channels_all_steps) # (step num, channel num, coeff num), e.g. (204, 18, 8)
        coeffs_all_channels_all_steps_all_file.append(coeffs_all_channels_all_steps)

        coeffs_non_seizure = np.concatenate(coeffs_all_channels_all_steps_all_file, axis=0) 

    return coeffs_non_seizure




def extract_features(coeffs):
    features = []
    for i, coeff in enumerate(coeffs):
        # feature 1
        features.append(np.mean(coeff)) 

        # # feature 2
        # features.append(np.mean(np.square(coeff))) 

        # feature 3
        features.append(np.std(coeff)) 
        
        # # feature 4
        # if i != len(coeffs[0:used_coeff_num]) - 1: # not last
        #     features.append( np.mean(np.abs(coeffs[i])) / np.mean(np.abs(coeffs[i+1])) ) 
        # else: 
        #     features.append( np.mean(np.abs(coeffs[i])) / 1 )

        # # feature 5
        # features.append(scipy.stats.skew(np.abs(coeff))) # so slow

        # # feature 6
        # peaks, _ = scipy.signal.find_peaks(coeff)
        # features.append(len(peaks)) 

        # # feature 7
        # features.append(np.median(coeff)) 

    return features




def confusion_matrix(y_true, y_pred):
    tp = np.sum((np.array(y_true)==1) & (np.array(y_pred)==1))
    tn = np.sum((np.array(y_true)==0) & (np.array(y_pred)==0))
    fp = np.sum((np.array(y_true)==0) & (np.array(y_pred)==1))
    fn = np.sum((np.array(y_true)==1) & (np.array(y_pred)==0))
    cm = np.array([[tp, fp],
                   [fn, tn]])
    # heatmap
    plt.imshow(cm, cmap=plt.cm.Blues)
    indices = range(len(cm))
    plt.xticks(indices, ['1:non-seizure', '0:seizure'])
    plt.yticks(indices, ['1:non-seizure', '0:seizure'])
    plt.colorbar()
    plt.title('confusion matrix')
    plt.xlabel('True')
    plt.ylabel('Predict')
    # show
    for first_index in range(len(cm)): # row
        for second_index in range(len(cm[first_index])): # column
            plt.text(first_index, second_index, cm[second_index][first_index])
    plt.show()




def report_and_confusion_matrix(best_model_list, X_train, y_train, X_val, y_val, X_test, y_test, confusion_matrix, labels):
    for best_model in best_model_list:
        print('==================== {} ===================='.format(best_model.__class__.__name__))
        # train
        y_train_pred = best_model.predict(X_train)
        print('========== train ==========')
        print(classification_report(y_train, y_train_pred, target_names=labels))
        confusion_matrix(y_train, y_train_pred)
        # val
        y_val_pred = best_model.predict(X_val)
        print('========== val ==========')
        print(classification_report(y_val, y_val_pred, target_names=labels))
        confusion_matrix(y_val, y_val_pred)
        # test
        y_test_pred = best_model.predict(X_test)
        print('========== test ==========')
        print(classification_report(y_test, y_test_pred, target_names=labels))
        confusion_matrix(y_test, y_test_pred)




def roc_and_auc(best_model_list, X_test, y_test, roc_curve):
    plt.figure(figsize=(10, 10))
    for best_model in best_model_list:
        # calculate fpr & tpr
        if hasattr(best_model, 'predict_proba'):
            pred_prob = best_model.predict_proba(X_test)[:, 1] # [:, 1]: prob of label 1
        else:
            pred_prob = best_model.decision_function(X_test)
        FPR, TPR, thresholds = roc_curve(y_test, pred_prob)
        # calculate AUC
        AUC = auc(FPR, TPR)
        print('{}:{:.2f}%'.format(best_model.__class__.__name__, AUC*100))
        # plot ROC curve
        plt.plot(FPR, TPR, label='{}: AUC on test = {:.2f}%'.format(best_model.__class__.__name__, AUC*100)) # lw=1, 'r'
    # plot random line
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random')

    plt.title('ROC Curve & AUC')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.legend(loc='lower right')
    plt.xlabel('FPR (1-Specificity)')
    plt.ylabel('TPR (Recall / Sensitivity)')
    plt.show()