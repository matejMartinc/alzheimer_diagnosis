import pandas as pd
from collections import defaultdict
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


import numpy as np
import os
from minisom import MiniSom


def create_adr_features(features, durations, id2idx, k, feature_grid, kmeans=True):
    pca = PCA(n_components=1)
    print("Num features: ", len(features))
    if durations is not None:
        durations = np.array(durations)

    #normalize
    np_features = np.array(features)
    print("Feature shape", np_features.shape)

    if kmeans:
        clustering = KMeans(n_clusters=min(k, len(np_features)), random_state=0).fit(np_features)
        labels = clustering.labels_
        centroids = clustering.cluster_centers_
    else:
        clustering = MiniSom(k, k, np_features.shape[1], sigma=0.5, learning_rate=0.2, neighborhood_function='gaussian', random_seed=10)
        clustering.train_batch(features, 500, verbose=True)
        labels = np.array([clustering.winner(x) for x in features]).T
        labels = np.ravel_multi_index(labels, (k,k))

    print("Labels shape", labels.shape)
    print("Labels count:", len(Counter(labels)))

    data = []
    for id, idxs in id2idx.items():
        doc_distrib = labels[idxs]
        doc_embeds = np_features[idxs]

        if kmeans:
            centroid_velocities = np.zeros((len(doc_distrib)-1))
            embed_velocities = np.zeros((len(doc_distrib)-1))
            for i in range(1, len(doc_distrib)):
                len(doc_distrib)
                label1 = doc_distrib[i]
                label2 = doc_distrib[i-1]
                cs_centroids = cosine_similarity([centroids[label1]], [centroids[label2]])[0][0]
                cs_embeds = cosine_similarity([doc_embeds[i]], [doc_embeds[i-1]])[0][0]
                embed_velocities[i - 1] = cs_embeds
                centroid_velocities[i - 1] = cs_centroids

            if len(centroid_velocities) > 128:
                centroid_velocities = centroid_velocities[:128]
                embed_velocities = embed_velocities[:128]

            else:
                centroid_velocities = np.pad(centroid_velocities, (0, 128 - len(centroid_velocities)), 'constant', constant_values=(0, 0))
                embed_velocities = np.pad(embed_velocities, (0, 128 - len(embed_velocities)), 'constant', constant_values=(0, 0))

            centroid_embeds = np.zeros((len(doc_distrib), doc_embeds.shape[1]))
            for i in range(len(doc_distrib)):
                label = doc_distrib[i]

                centroid_embed = centroids[label]
                centroid_embeds[i,:] = centroid_embed

            centroid_embeds = pca.fit_transform(centroid_embeds).squeeze()
            if len(centroid_embeds) > 128:
                centroid_embeds = centroid_embeds[:128]
            else:
                centroid_embeds = np.pad(centroid_embeds, (0, 128 - len(centroid_embeds)), 'constant', constant_values=(0, 0))

        label_velocities = []
        for i in range(1, len(doc_distrib)):
            diff_labels = doc_distrib[i] - doc_distrib[i-1]
            label_velocities.append(diff_labels)

        label_acceleration = []
        for i in range(1, len(label_velocities)):
            diff_labels = label_velocities[i] - label_velocities[i-1]
            label_acceleration.append(diff_labels)

        #get count features
        c = Counter(doc_distrib)
        num_all = len(doc_distrib)
        counts = []
        for i in range(k):
            if i in c:
                counts.append(c[i])
            else:
                counts.append(0)
        counts = [x / num_all for x in counts]

        #get embedding features
        embeds = pca.fit_transform(doc_embeds).squeeze()
        if len(embeds) > 128:
            embeds = embeds[:128]
        else:
            embeds = np.pad(embeds, (0, 128 - len(embeds)), 'constant', constant_values=(0, 0))

        #duration
        if durations is not None:
            doc_dur = durations[idxs]
            dur_dict = defaultdict(int)
            all_dur = sum(doc_dur)
            for l, dur in zip(doc_distrib, doc_dur):
                dur_dict[l] += dur
            doc_durations = []
            for i in range(k):
                if i in dur_dict:
                    doc_durations.append(dur_dict[i]/all_dur)
                else:
                    doc_durations.append(0)
            #print(id, doc_durations)

        features = id.split('-')
        if 'duration' in feature_grid and durations is not None:
            features = features + doc_durations
        if 'counts' in feature_grid:
            features = features + counts
        if 'embeds' in feature_grid:
            features = features + list(embeds)
        if 'centroid_embeds' in feature_grid:
            features = features + list(centroid_embeds)
        if 'embed_velocity' in feature_grid:
            features = features + list(embed_velocities)
        if 'centroid_velocity' in feature_grid and kmeans:
            features = features + list(centroid_velocities)
        data.append(features)
    return data


def get_duration(id):
    id = id.split('.')[0]
    duration = int(id.split('-')[-1]) - int(id.split('-')[-2])
    return duration


def read_audio_features(input_folder, id_dict):
    files = os.listdir(input_folder)
    fname = 'eGeMAPs'

    df = None
    for file in files:
        path = os.path.join(input_folder,file)
        df_file = pd.read_csv(path, encoding='utf8', sep=',',header=None)
        if df is None:
            df = df_file
        else:
            df = pd.concat([df, df_file])
    df = df.fillna(0)
    df['duration'] = df[0].apply(lambda x: get_duration(x))
    df = df.rename(columns={0: "id"})
    filtered_columns = ['id', 'duration']

    dur = df['duration'].tolist()
    for col in df.columns:
        if col != 'id':
            col_data = df[col].tolist()
            pearson, _ = pearsonr(dur, col_data)
            if abs(pearson) < 0.2:
                filtered_columns.append(col)

    df = df[filtered_columns]

    for idx, row in df.iterrows():
        id = row['id']

        duration = row['duration']

        file_id = id.split('-')[0]
        chunk_start = id.split('-')[1]
        num_zeros = 10 - len(chunk_start)
        s = ''
        for _ in range(num_zeros):
            s += '0'
        chunk_start = s + chunk_start

        word = id.split('-')[4]
        position = id.split('-')[3]
        num_zeros = 3 - len(position)
        s = ''
        for _ in range(num_zeros):
            s += '0'
        position = s + position
        chunk_pos_id = chunk_start + position

        row = row.drop(['id', 'duration'])
        features = np.array(row)
        features = (features - np.min(features)) / np.ptp(features)
        id_dict[file_id].append((chunk_pos_id + '_' + fname, features, duration))

    return id_dict

def read_text_features(input_folders, embeddings_path, id_dict):
    embeddings_dict = {}
    with open(embeddings_path, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector

    for input_folder, dirnames, filenames in os.walk(input_folders):
        for file in [f for f in filenames]:
            if file.endswith('.txt'):
                file_id = file.split('-')[0]
                chunk_start = file.split('-')[1]
                num_zeros = 10 - len(chunk_start)
                s = ''
                for _ in range(num_zeros):
                    s += '0'
                chunk_start = s + chunk_start
                path = os.path.join(input_folder, file)
                with open(path, 'r', encoding='utf8') as f:
                    text = []
                    for line in f:
                        text.append(line.split(',')[0])
                    for word_idx, word in enumerate(text):
                        emb_word = word.replace("'S", '').replace("'RE", '').replace("'M", '').replace("'LL", '').replace("'T", '')
                        emb_word = emb_word.lower()
                        if emb_word in embeddings_dict and emb_word != 'sp':
                            word_embed = np.array(embeddings_dict[emb_word])
                            word_embed = (word_embed - np.min(word_embed)) / np.ptp(word_embed)
                        else:
                            word_embed = np.zeros(50)

                        position = str(word_idx + 1)
                        num_zeros = 3 - len(position)
                        s = ''
                        for _ in range(num_zeros):
                            s += '0'
                        position = s + position
                        chunk_pos_id = chunk_start + position
                        id_dict[file_id].append((chunk_pos_id + '_text', word_embed))
    return id_dict


def combine_text_and_audio(id_dict, audio):
    new_id_dict = {}
    if audio:
        duration_id_dict = {}
    num_all = 0
    for id, features in id_dict.items():

        seq_id_dict = defaultdict(list)
        if audio:
            seq_duration_dict = defaultdict(float)

        features = sorted(features, key=lambda x:x[0])
        for feature in features:
            seq_id = feature[0][:13]

            if isinstance(feature[1], np.ndarray):
                #print(seq_id, feature[1].tolist())
                seq_id_dict[seq_id].extend(feature[1].tolist())
            if feature[0].endswith('eGeMAPs'):
                num_all += 1
                duration = feature[2]
                seq_duration_dict[seq_id] = duration

        seq_list = sorted(list(seq_id_dict.items()), key=lambda x: x[0])
        seq_list = [x[1] for x in seq_list]
        new_id_dict[id] = seq_list

        if audio:
            seq_dur_list = sorted(list(seq_duration_dict.items()), key=lambda x: x[0])
            seq_dur_list = [x[1] for x in seq_dur_list]
            duration_id_dict[id] = seq_dur_list

    print('All combined features: ', num_all)
    if audio:
        return new_id_dict, duration_id_dict

    return new_id_dict


def get_adr_input(word_text_features, word_audio_features, embeddings_path, audio=True, text=True):
    if audio and text:
        num_features = 122
    elif audio:
        num_features = 72
    elif text:
        num_features = 50
    id_dict = defaultdict(list)
    if audio:
        id_dict = read_audio_features(word_audio_features, id_dict)
    if text:
        id_dict = read_text_features(word_text_features, embeddings_path, id_dict)
    if audio:
        feature_dict, dur_dict = combine_text_and_audio(id_dict, audio)
    else:
        feature_dict = combine_text_and_audio(id_dict, audio)
    id2idx = defaultdict(list)
    all_features = []
    for k, v in feature_dict.items():
        for seg in v:
            id2idx[k].append(len(all_features))
            if len(seg) == num_features:
                all_features.append(np.array(seg).squeeze())
    all_features = np.array(all_features)
    print("ADR feature shape: ", all_features.shape)

    if audio:
        all_dur = []
        for k, v in dur_dict.items():
            for dur in v:
                all_dur.append(dur)
        return all_features, all_dur, id2idx
    return all_features, None, id2idx













