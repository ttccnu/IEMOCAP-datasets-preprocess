import os
import glob
import python_speech_features as ps
import wave
import numpy as np
import re
import jieba


def read_file(filename):
    file = wave.open(filename, 'r')
    params = file.getparams()
    nchannels, sampwidth, framerate, wav_length = params[:4]
    str_data = file.readframes(wav_length)
    wavedata = np.fromstring(str_data, dtype=np.short)
    time = np.arange(0, wav_length) * (1.0 / framerate)
    file.close()
    return wavedata, time, framerate


def read_IEMOCAP():
    rootdir = r"D:\Datasets\IEMOCAP corpus\IEMOCAP"
    train_emt = {'hap': 0, 'ang': 0, 'neu': 0, 'sad': 0}
    train_filename_save = []
    train_data = []
    train_label = []

    for speaker in os.listdir(rootdir):
        if (speaker[0] == 'S'):
            sub_dir = os.path.join(rootdir, speaker, speaker, 'sentences/wav')
            emoevl = os.path.join(rootdir, speaker, speaker, 'dialog/EmoEvaluation')
            for sess in os.listdir(sub_dir):
                emotdir = emoevl + '/' + sess + '.txt'
                emot_map = {}
                with open(emotdir, 'r') as emot_to_read:
                    while True:
                        line = emot_to_read.readline()
                        if not line:
                            break
                        if (line[0] == '['):
                            t = line.split()
                            emot_map[t[3]] = t[4]

                file_dir = os.path.join(sub_dir, sess, '*.wav')
                files = glob.glob(file_dir)

                for filename in files:
                    wavname = filename.split("\\")[-1][:-4]
                    emotion = emot_map[wavname]

                    if emotion in ['hap', 'ang', 'neu', 'sad']:
                        data, time, rate = read_file(filename)
                        mel_spec = ps.logfbank(data, rate, nfilt=40)
                        time = mel_spec.shape[0]
                        if time <= 300:
                            part = mel_spec
                            part = np.pad(part, ((0, 300 - part.shape[0]), (0, 0)), 'constant',
                                          constant_values=0)
                            train_data.append(part)
                            train_label.append(emotion)
                            train_emt[emotion] = train_emt[emotion] + 1
                            train_filename_save.append(filename)
                        else:
                            begin = 0
                            end = 300
                            part = mel_spec[begin:end, :]
                            train_data.append(part)
                            train_label.append(emotion)
                            train_emt[emotion] = train_emt[emotion] + 1
                            train_filename_save.append(filename)
                    elif emotion in ['exc']:  # 将激动也归为happy类。
                        data, time, rate = read_file(filename)
                        mel_spec = ps.logfbank(data, rate, nfilt=40)
                        time = mel_spec.shape[0]
                        if time <= 300:
                            part = mel_spec
                            part = np.pad(part, ((0, 300 - part.shape[0]), (0, 0)), 'constant',
                                          constant_values=0)
                            train_data.append(part)
                            train_label.append('hap')
                            train_emt['hap'] = train_emt['hap'] + 1
                            train_filename_save.append(filename)
                        else:
                            begin = 0
                            end = 300
                            part = mel_spec[begin:end, :]
                            train_data.append(part)
                            train_label.append('hap')
                            train_emt['hap'] = train_emt['hap'] + 1
                            train_filename_save.append(filename)

    # 提取出transcription
    
    train_text = []
    i = 0
    for path in train_filename_save:
        temp_1 = path.split("\\")
        sub_path = temp_1[0] + "\\" + temp_1[1] + "\\" + temp_1[2] + "\\" + temp_1[3] \
                   + "\\" + temp_1[4] + "\\" + \
                   temp_1[5] + "\\" + 'dialog\\transcriptions\\' + temp_1[7] + ".txt"
        with open(sub_path, 'r') as f:
            d = f.read()
            txt = d.split("\n")
            for line in txt:
                if len(line) > 30 and "]:" in line:
                    filename = line.split()[0]
                    content = line.split("]: ")[1]
                    if filename == temp_1[8].split('.')[0]:
                        train_text.append(content)
                        break
            f.close()
        i += 1
        if len(train_text) != i:
            train_text.append(" ")
        pass

    texts = [s.lower() for s in train_text]
    re_han = re.compile("([a-zA-Z]+)")

    train_text = []
    for line in texts:
        word = []
        blocks = re_han.split(line)
        for blk in blocks:
            if re_han.match(blk):
                word.extend(jieba.lcut(blk))
        train_text.append(word)

    return train_data, train_label, train_filename_save, train_text, train_emt


train_data, train_label, train_filename_save, train_text, train_emt = read_IEMOCAP()

