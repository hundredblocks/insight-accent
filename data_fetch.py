import os
from os import listdir
from os.path import isfile, join
from shutil import copyfile

import librosa
import itertools
import matplotlib

from bs4 import BeautifulSoup
import urllib
import tarfile
import numpy as np
import utils
import wave

# url = 'http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Original/48kHz_16bit/'
url = 'http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/'


def fetch_data():
    page = urllib.urlopen(url)
    soup = BeautifulSoup(page)

    for link in soup.find_all('a'):
        link_url = link.get('href')
        extension = link_url.split('.')[-1]
        if extension == 'tgz':
            name = link_url.split('.')[-2]
            download_url = url + link.get('href')
            print(download_url)
            file_name = 'raw_sound/' + name + '.' + extension
            urllib.urlretrieve(download_url, file_name)
            tar = tarfile.open(file_name)
            tar.extractall('raw_sound/')


def inspect():
    dialect_dict = {}
    i = 0
    j = 0
    k = 0
    for root, dirs, files in os.walk('raw_sound/'):
        i += 1
        if os.path.basename(root) != 'etc':
            continue
        for f in files:
            if not f.startswith('README'):
                continue
            with open(os.path.join(root, f)) as fp:
                k += 1
                data = fp.read()
                try:
                    idx = data.index('Pronunciation dialect')
                except ValueError:
                    break
                try:
                    idx2 = data.index('Gender')
                except ValueError:
                    break
                j += 1
                fp.seek(idx)
                dialect_line = fp.readline()
                dialect_field = dialect_line.split(':')[1] if len(dialect_line.split(':')) > 0 else ''
                dialect_field = dialect_field.strip('\n')

                fp.seek(idx2)
                gender_line = fp.readline()
                gender_field = gender_line.split(':')[1] if len(gender_line.split(':')) > 0 else ''
                gender_field = gender_field.strip('\n')
                # print(gender_field)
                to_dict = dialect_field
                # to_dict = dialect_field + ' / ' + gender_field
                print(dialect_field)
                print(root, dirs)

                dialect_dict[to_dict] = dialect_dict.get(to_dict, 0) + 1
                print('parsed %s out of %s readmes out of %s files' % (j, k, i))
    dialect_tup = [(k, v) for k, v in dialect_dict.items()]
    dialect_tup = sorted(dialect_tup, key=lambda x: x[1], reverse=True)
    print(dialect_tup)


def organize():
    for root, dirs, files in os.walk('raw_sound/'):
        dirnames = [os.path.basename(d) for d in dirs]
        if len(dirnames) == 2:
            if 'wav' in dirnames and 'etc' in dirnames:
                move_subfolder(root, dirs)
            else:
                continue


def move_subfolder(root, dirs):
    readme_files, wav_files = get_dir_files(root, dirs)
    if readme_files == "" or wav_files == "":
        return

    readme = [f for f in readme_files if f.startswith('README')]
    if len(readme) != 1:
        return

    readme = readme[0]
    dialect, gender = get_dialect_gender(root, readme)

    if dialect == "" or gender == "":
        return
    target_path = get_target_folder(dialect, gender)
    if target_path == "":
        return
    wavs = [f for f in wav_files if f.endswith('.wav')]

    for wav in wavs:
        copyfile(os.path.join(root, 'wav', wav), os.path.join(target_path, wav))


def get_dir_files(root, dirs):
    readme = [d for d in dirs if d.startswith('etc')]
    wav = [d for d in dirs if d.startswith('wav')]
    if len(readme) != 1 or len(wav) != 1:
        return "", ""

    readme_dir = readme[0]
    readme_dir = os.path.join(root, readme_dir)
    readme_files = [f for f in listdir(readme_dir) if isfile(join(readme_dir, f))]

    wav_dir = wav[0]
    wav_dir = os.path.join(root, wav_dir)
    wav_files = [f for f in listdir(wav_dir) if isfile(join(wav_dir, f))]
    return readme_files, wav_files


def get_target_folder(dialect, gender, base_output_path="sorted_sound"):
    dialect_path = ""
    gender_path = ""
    if dialect.find("american") != -1:
        dialect_path = "american"
    elif dialect.find("british") != -1:
        dialect_path = "british"

    if gender.find("female") != -1:
        gender_path = "female"
    elif gender.find("male") != -1:
        gender_path = "male"

    if dialect_path != "" and gender_path != "":
        return base_output_path + "/" + dialect_path + "/" + gender_path
    return ""


def get_dialect_gender(root, f):
    with open(os.path.join(root, 'etc', f)) as fp:
        data = fp.read()
        try:
            idx = data.index('Pronunciation dialect')
        except ValueError:
            return "", ""
        try:
            idx2 = data.index('Gender')
        except ValueError:
            return "", ""
        fp.seek(idx)
        dialect_line = fp.readline()
        dialect_field = dialect_line.split(':')[1] if len(dialect_line.split(':')) > 0 else ''
        dialect_field = dialect_field.strip('\n')
        dialect_field = dialect_field.strip(' ')
        dialect_field = dialect_field.lower()

        fp.seek(idx2)
        gender_line = fp.readline()
        gender_field = gender_line.split(':')[1] if len(gender_line.split(':')) > 0 else ''
        gender_field = gender_field.strip('\n')
        gender_field = gender_field.strip(' ')
        gender_field = gender_field.lower()

        return dialect_field, gender_field


def cut():
    # utils.slice('organized_sound/wav/british/bo156.wav', 'test.wav', 0, 3000)
    base_dir = 'organized_sound/wav/'
    language_dirs = ['american', 'british']
    for language in language_dirs:
        lang_dir_path = base_dir + language + '/'
        for filename in os.listdir(base_dir + language):
            if filename.endswith('.wav'):
                filepath = lang_dir_path + filename
                outpath = lang_dir_path + 's_' + filename
                infile = wave.open(filepath)
                utils.slice(infile, outpath, 0, 3000)


def cut_all(base_dir):
    # base_dir = 'sorted_sound'
    for root, dirs, files in os.walk(base_dir):
        print(root, dirs, files)
        if 'wav' not in dirs:
            continue
        out_path = os.path.join(root, 'cut')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        wav_path = os.path.join(root, 'wav')
        for filename in os.listdir(wav_path):
            filepath = os.path.join(wav_path, filename)
            infile = wave.open(filepath)
            utils.multislice(infile, out_path, filename,
                             ms_cut_size=3000, ms_step_size=1000)


def cut_daps(path, start_pad=0, end_pad=None, ms_cut_size=3000, ms_step_size=1000, out='cut'):
    to_cut = []
    for root, dirs, files in os.walk(path):
        print(root, dirs, files)
        for filename in files:
            is_wav = filename.endswith('wav')
            if not is_wav:
                continue
            filepath = os.path.join(root, filename)
            print(filepath)
            infile = wave.open(filepath)
            out_path = os.path.join(root, out)
            to_cut.append([infile, out_path, filename])
    for infile, out_path, filename in to_cut:
        print([infile, out_path, filename])
        utils.multislice(infile, out_path, filename, start_pad=start_pad, end_pad=end_pad,
                         ms_cut_size=ms_cut_size, ms_step_size=ms_step_size)


# used_genders is an array
def preprocess_and_load(path, data_limit=None, used_genders=None):
    path_dict = get_path_dict(path, used_genders=used_genders)

    truncated_path_dict = get_equal_classes(path_dict, data_limit)
    new_classes_length = [[k, len(v)] for k, v in truncated_path_dict.items()]
    print("classes and number of examples %s" % new_classes_length)
    return get_examples_from_paths(truncated_path_dict)


# Data format: [[input, target, paths],...]
def get_all_audio_in_folder(path, subsample=-1):
    source_list = []
    target_list = []
    path_list = []
    onlyfiles = [f for f in listdir(path) if
                 isfile(join(path, f)) and not f.startswith('.DS')]
    onlyfiles = sorted(onlyfiles)
    for i, f in enumerate(onlyfiles):
        if subsample != -1 and i > subsample:
            continue
        file = os.path.join(path, f)
        x, fs = librosa.load(file)
        S = utils.read_audio_spectrum(x, fs)
        formatted_vec = np.ascontiguousarray(S.T[None, None, :, :])
        source_list.append(formatted_vec)
        target_list.append(formatted_vec)
        path_list.append(f)
    # assume source and target have same dims
    t_dim = source_list[0].shape[2]
    f_dim = source_list[0].shape[3]

    formatted_source = np.zeros([len(source_list), t_dim, f_dim])
    for i, x in enumerate(source_list):
        formatted_source[i] = x

    formatted_target = np.zeros([len(source_list), t_dim, f_dim])
    for i, x in enumerate(target_list):
        formatted_target[i] = x
    data_and_label = [[formatted_source[i, :, :], formatted_target[i, :, :], path_list[i]] for i in
                      range(len(source_list))]
    return data_and_label, fs


# Data format: [[input, paths],...]
def get_all_autoencoder_audio_in_folder(path, subsample=-1, class_label=None, random=False):
    source_list = []
    path_list = []
    if class_label is None:
        class_label = [1, 0]
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f)) and not f.startswith('.DS')]
    if random is True:
        onlyfiles = np.random.permutation(onlyfiles)
    for i, f in enumerate(onlyfiles):
        if subsample != -1 and i >= subsample:
            continue
        file = os.path.join(path, f)
        x, fs = librosa.load(file)
        S = utils.read_audio_spectrum(x, fs)
        formatted_vec = np.ascontiguousarray(S.T[None, None, :, :])
        source_list.append(formatted_vec)
        path_list.append(f)

    t_dim = source_list[0].shape[2]
    f_dim = source_list[0].shape[3]

    formatted_source = np.zeros([len(source_list), t_dim, f_dim])
    for i, x in enumerate(source_list):
        formatted_source[i] = x

    data_and_label = [np.array([formatted_source[i, :, :], class_label, path_list[i]]) for i in
                      range(len(source_list))]
    return data_and_label, fs


def save_spectrogram_array(out_path, spectrogram):
    with open(out_path, 'w+') as outfile:
        np.save(outfile, spectrogram)


def load_spectrogram_array(in_path):
    with open(in_path, 'r') as outfile:
        return np.load(outfile)


# def sound_to_spectrogram(sound, sampling, n_fft=2048):
#     y = utils.read_audio_spectrum(sound, sampling, n_fft=n_fft, reduce_factor=1)


def get_male_female_pairs(path, product=True, subsample=-1):
    folders = [f for f in listdir(path) if
               not isfile(join(path, f)) and f.endswith('male')]
    # data = {}
    label_map = {}
    for fol in folders:
        data_and_label, fs = get_all_audio_in_folder(os.path.join(path, fol), subsample=subsample)
        label_map[fol] = {'_'.join(d[-1].split('_')[1:]): d[0] for d in data_and_label}

    male_audio = []
    female_audio = []
    female_map = label_map['female']
    for key, value in label_map['male'].items():
        male_audio.append([value, key])
        female_audio.append([female_map[key], key])
    final_pairs = []
    if product:
        to_iterate = itertools.product(male_audio, female_audio)
    else:
        to_iterate = [[male_audio[i], female_audio[i]] for i in range(len(male_audio))]
    for pair in to_iterate:
        name = '%s_%s' % (pair[0][-1], pair[-1][-1])
        formatted_pair = [np.array(pair[0][0]), np.array(pair[1][0]), name]
        final_pairs.append(formatted_pair)
    return final_pairs, fs


def get_examples_from_paths(path_dict):
    data_list = []
    label_list = []
    classes = path_dict.keys()
    num_classes = len(classes)
    for cls, paths in path_dict.items():
        class_int = classes.index(cls)
        for audio_file in paths:
            x, fs = librosa.load(audio_file)
            S = utils.read_audio_spectrum(x, fs)
            formatted_vec = np.ascontiguousarray(S.T[None, None, :, :])
            data_list.append(formatted_vec)

        label_list.append(np.ones([len(paths), 1]) * class_int)

    Ys = np.concatenate(label_list)

    specX = np.zeros([len(data_list), 130, 1025])
    for i, x in enumerate(data_list):
        specX[i] = x

    data_and_label = [[specX[i, :, :], Ys[i]] for i in range(len(data_list))]
    split1 = specX.shape[0] - specX.shape[0] / 5
    split2 = (specX.shape[0] - split1) / 2

    shuffled_data = np.random.permutation(data_and_label)
    shuffled_x = [a[0] for a in shuffled_data]
    shuffled_y = [a[1] for a in shuffled_data]
    trainX, otherX = np.split(shuffled_x, [split1])
    trainYa, otherY = np.split(shuffled_y, [split1])
    valX, testX = np.split(otherX, [split2])
    valYa, testYa = np.split(otherY, [split2])

    trainY = to_one_hot(trainYa)
    testY = to_one_hot(testYa)
    valY = to_one_hot(valYa)
    return num_classes, trainX, trainY, valX, valY, testX, testY


def get_equal_classes(path_dict, data_limit=None):
    classes_length = [[k, len(v)] for k, v in path_dict.items()]
    # here we want the same out of each class
    to_use = min([c[1] for c in classes_length])
    if data_limit is not None:
        to_use = min(to_use, data_limit)
    truncated_path_dict = {k: np.random.choice(v, to_use, replace=False) for k, v in path_dict.items()}
    new_classes_length = [[k, len(v)] for k, v in truncated_path_dict.items()]
    return truncated_path_dict


def get_path_dict(path, used_genders):
    accent = [f for f in listdir(path) if not isfile(join(path, f))]
    path_dict = {}
    if used_genders is None:
        used_genders = ['male', 'female']
    for acc in accent:
        full_path = os.path.join(path, acc)
        gender = [f for f in listdir(full_path) if not isfile(join(full_path, f))]
        gender = [g for g in gender if g in used_genders]
        for gen in gender:
            full_gender_path = os.path.join(full_path, gen)
            cut_subfolders = [f for f in listdir(full_gender_path) if f.endswith('cut')]
            if len(cut_subfolders) != 1:
                raise ValueError("One cut folder required: %s" % cut_subfolders)
            cut_subfolder = cut_subfolders[0]
            cut_path = os.path.join(full_gender_path, cut_subfolder)
            onlyfiles = [os.path.join(cut_path, f) for f in listdir(cut_path) if
                         isfile(join(cut_path, f)) and not f.startswith('.DS')]
            path_dict["%s_%s" % (acc, gen)] = onlyfiles

    return path_dict


def preprocess(sound_path):
    folders = [f for f in listdir(sound_path) if not isfile(join(sound_path, f))]
    num_classes = len(folders)
    data_list = []
    label_list = []
    for folder_ix, folder in enumerate(folders):
        sub_path = join(sound_path, folder)
        onlyfiles = [f for f in listdir(sub_path) if isfile(join(sub_path, f)) and not f.startswith('.DS')]

        for ix, audio_file in enumerate(onlyfiles):
            x, fs = librosa.load(join(sub_path, audio_file))
            S, fs = utils.read_audio_spectrum(x, fs)
            formatted_vec = np.ascontiguousarray(S.T[None, None, :, :])
            data_list.append(formatted_vec)
        label_list.append(np.ones([len(onlyfiles), 1]) * folder_ix)

    Ys = np.concatenate(label_list)

    specX = np.zeros([len(data_list), 130, 1025])
    for i, x in enumerate(data_list):
        specX[i] = x

    data_and_label = [[specX[i, :, :], Ys[i]] for i in range(len(data_list))]
    split1 = specX.shape[0] - specX.shape[0] / 5
    split2 = (specX.shape[0] - split1) / 2

    shuffled_data = np.random.permutation(data_and_label)
    shuffled_x = [a[0] for a in shuffled_data]
    shuffled_y = [a[1] for a in shuffled_data]
    trainX, otherX = np.split(shuffled_x, [split1])
    trainYa, otherY = np.split(shuffled_y, [split1])
    valX, testX = np.split(otherX, [split2])
    valYa, testYa = np.split(otherY, [split2])

    trainY = to_one_hot(trainYa)
    testY = to_one_hot(testYa)
    valY = to_one_hot(valYa)
    return num_classes, trainX, trainY, valX, valY, testX, testY


def to_one_hot(Y):
    res = []
    for y in Y:
        if y == 0:
            res.append([1, 0])
        if y == 1:
            res.append([0, 1])
    return res


if __name__ == '__main__':
    pass
    # fetch_data()
    # inspect()
    # organize()
    # cut_all()
    # preprocess('organized_sound/wav/')
