from sklearn.model_selection import StratifiedKFold


read_all_classes = ['rhythmic', 'seiz', 'single', 'other'] 

seizure_classes = ['rhythmic', 'seiz', 'single'] # class 0: seizure
# seizure_classes = ['rhythmic', 'seiz', 'single', 'other'] # class 0: seizure
# not need to define # class 1: non-seizure
other_classes = ['other'] # class 2: other

sampling_rate = 500 # Data-Section [Hz]
# sampling_rate = 256 # CHB-MIT [Hz]

channel_num = 18 # Data-Section
# DataSection 18 channels
channel_indices = [
    (0, 10), (10, 2), (12, 14), (14, 8), (1, 11), 
    (11, 13), (13, 15), (15, 9), (0, 2), (2, 4), 
    (4, 6), (6, 8), (1, 3), (3, 5), (5, 7), 
    (7, 9), (16, 17), (17, 18)
]
# # Data-Section ? channels following to CHB-MIT
# channel_indices = [
#     (0, 10), (10, 2), (12, 14), (14, 8), (1, 11), 
#     (11, 13), (13, 15), (15, 9), (0, 2), (2, 4), 
#     (4, 6), (6, 8), (1, 3), (3, 5), (5, 7), 
#     (7, 9), (16, 17), (17, 18)
# ]
# channel_num = 23 # CHB-MIT

lowcut = 0.5 # low frequency threshold
highcut = 30 # high frequency threshold

scaling_factor = 1e6 # μV -> pV (1e-12 V)

# train & val
data_dir_1 = r'./DataSection-EEG/train_val/data/'
label_dir_1 = r'./DataSection-EEG/train_val/label/'
# test
data_dir_2 = r'./DataSection-EEG/test/data/'
label_dir_2 = r'./DataSection-EEG/test/label/'

wavelet = 'db4' # Daubechies 4
level = 6 # Data-Section (Decomposition level number)
# level = 5 # CHB-MIT (Decomposition level number)

window_size = 1 # [s]
moving_size = 0.3 # [s]

label_balance = 1 # e.g. 1(seiz:non-seiz=1:1), 2(seiz:non-seiz=1:2), 0.5(seiz:non-seiz=2:1), 0.333(seiz:non-seiz=3:1)

used_coeff_num = 4 # Data-Section (A6, D6, D5, D4)
# used_coeff_num = 4 # CHB-MIT (A5, D5, D3, D3)

num_folds = 5
skf = StratifiedKFold(n_splits=num_folds, shuffle=True)

hyper_search_metric = 'f1'

labels = ['0: seizure', '1: non-seizure']