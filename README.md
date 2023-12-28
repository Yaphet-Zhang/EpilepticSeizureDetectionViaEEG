# 1. Task
- Epileptic Seizure Detection




# 2. Method & Dataset
## 2.1 EEG-based
[5] [CHB-MIT Scalp EEG Database (Dataset)](https://physionet.org/content/chbmit/1.0.0/)

[6] [EEG Seizure Analysis Dataset (Kaggle)](https://www.kaggle.com/datasets/adibadea/chbmitseizuredataset)

[7] [Epileptic Seizure Recognition Data Set (Dataset)](https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition)

[Python深度学习：脑电图癫痫发作检测 (Blog)](https://www.toutiao.com/article/6687151354311344648/?wid=1684997097532)

[利用深度学习(Keras)进行癫痫分类-Python案例 (Blog)](https://zhuanlan.zhihu.com/p/104708730)

[8] [CHB-MIT波士顿儿童医院癫痫EEG脑电数据处理](https://blog.csdn.net/qq_42877824/category_10674404.html)

[9] [Pythonで脳波解析：Python MNEのチュートリアル](https://qiita.com/sentencebird/items/035ba0c48569f06e3a42)

[10] [脳波の手習い](https://naraamt.or.jp/Academic/kensyuukai/2005/kirei/nouha_mon/nouha_mon.html)

[11] [CNNs-on-CHB-MIT (Github)](https://github.com/SMorettini/CNNs-on-CHB-MIT)

[12] [Application of Machine Learning In Epileptic Seizure Detection machine (EEG) (Blog)](https://medium.com/@discoveryscientist/application-of-machine-learning-in-epileptic-seizure-detection-machine-eeg-dc095ce5de65)

[13] [Electroencephalography (EEG) | How EEG test works? | What conditions can an EEG diagnose? | Animated (YouTube)](https://www.youtube.com/watch?v=T7MKlPYiL48&t=495s)

[14] [EEG ML/DL (YouTube)](https://www.youtube.com/playlist?list=PLtGXgNsNHqPTgP9wyR8pmy2EuM2ZGHU5Z)

[Epilepsy classification using Welch power spectral density (Complete EEG tutorial) (YouTube)](https://www.youtube.com/watch?v=cZf5rE5tUr0&list=PLtGXgNsNHqPTgP9wyR8pmy2EuM2ZGHU5Z&index=16)

[EEG ML/DL (Github)](https://github.com/talhaanwarch/youtube-tutorials)

- Dataset
    - CHB-MIT
        - scalp EEG signals: 844 h, 24 children, 163 seizures, 13 patients, 10-20 electrode positions, 1/256 Hz
        - inter-ictal period: 4h before start ~ 4h after end
        - category type: seizure & non-seizure
        - seizure type: combined seizure & main seizures
    - Kaggle
    - Freiburg
    - Bonn
    - Flint-Hills
    - Bern-Barcelona
    - Hauz Khas
    -  Zenodo
- Method
    - Traditional
        - Time-Frequence
            - Wavelet
        - Time
            - PCA
        - Frequence
            - Spectrum
        - Non-linear
            - Entropy
    - Deep Learning
        - 2D-CNN
        - 1D-CNN
        - RNN
        - AE




# 3. Install
```
conda create -n EEG python=3.8
conda activate EEG
pip install -r requirements.txt
```




