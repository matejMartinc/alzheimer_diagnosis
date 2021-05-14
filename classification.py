import pandas as pd
import pickle
import argparse
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn import pipeline
from sklearn.preprocessing import Normalizer
from adr_features import get_adr_input, create_adr_features
import os
import joblib
import plotly.figure_factory as ff
from sklearn.base import BaseEstimator, TransformerMixin

class text_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key]

class digit_features(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        columns = [col for col in data_dict.keys() if col.startswith(self.key)]
        filtered_dict = {col: data_dict[col] for col in columns}
        features = pd.DataFrame(filtered_dict).to_numpy()
        return features
        scaler = preprocessing.MinMaxScaler().fit(features)
        normalized_features = scaler.transform(features)
        return normalized_features

def visualize(combo, audio_acc, text_acc, audio_text_acc, adr_char_acc, char_acc):

    # Group data together
    hist_data = [audio_acc, text_acc, audio_text_acc, adr_char_acc, char_acc]

    group_labels = ['Audio', 'Text', 'Text+audio', 'Text+audio+char', 'Char']

    fig = ff.create_distplot(hist_data, group_labels, bin_size=0.007, show_rug=False, histnorm='probability', )
    fig.update_layout(xaxis_tickangle=0,
                      font=dict(family='Rockwell', size=18),
                      xaxis_title="Accuracy",
                      yaxis_title="Probability",
                      xaxis_tickfont=dict(family='Rockwell', size=24),
                      yaxis_tickfont=dict(family='Rockwell', size=24),
                      legend=dict(font=dict(family="Rockwell", size=24)),
                      )
    fig.write_image(f"{image_folder}/{combo}.png")



if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description='Dementia Pit')
    argparser.add_argument('--char_input_paths', type=str, default='data/train_text.tsv,data/test_text.tsv', help='Path to train and test TSV files generated in step 1 divided by comma.')
    argparser.add_argument('--results_folder', type=str, default='results', help='Path to folder containing all results')
    argparser.add_argument('--model_folder', type=str, default='trained_classification_models', help='Path to folder containing all trained models')
    argparser.add_argument('--audio_features_folder', type=str, default='data/word_audio_features', help='Path to folder containing generated audio features')
    argparser.add_argument('--text_features_folder', type=str, default='data/word_timestamps_1', help='Path to folder containing generated word timestamps features')
    argparser.add_argument('--embeddings_path', type=str, default='data/glove.6B.50d.txt', help='Path to embeddings')
    args = argparser.parse_args()

    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)
    if not os.path.exists(args.model_folder):
        os.makedirs(args.model_folder)

    results_folder = args.results_folder
    image_folder = os.path.join(results_folder, 'images')

    feature_combos = {
        'all': ['counts', 'duration', 'centroid_velocity', 'embed_velocity', 'centroid_embeds', 'embeds'],
        'temporal': ['counts', 'duration', 'centroid_velocity', 'embed_velocity'],
        'no_temporal': ['counts', 'duration', 'centroid_embeds', 'embeds'],
        'embed_PCA': ['counts', 'duration', 'embed_velocity', 'embeds'],
        'centroids_PCA': ['counts', 'duration', 'centroid_velocity', 'centroid_embeds'],
        'no_count+duration': ['centroid_velocity', 'centroid_embeds', 'embed_velocity', 'embeds']
    }

    folder = args.model_folder
    num_clusters = 30

    adr_configs = ['audio', 'text', 'audio+text', 'audio+text+char', 'char']
    char_done = False
    char_acc = []
    char_acc_cv = []

    output_accuracies_cv = open(os.path.join(results_folder, 'accuracies_cv.csv'), 'w', encoding='utf8')
    output_accuracies = open(os.path.join(results_folder, 'accuracies.csv'), 'w', encoding='utf8')

    counter = 0

    for combo_name, combo in feature_combos.items():
        audio_acc = []
        text_acc = []
        audio_text_acc = []
        audio_text_char_acc = []

        audio_acc_cv = []
        text_acc_cv = []
        audio_text_acc_cv = []
        audio_text_char_acc_cv = []

        for config in adr_configs:
            counter += 1
            num_combs = len(feature_combos) * len(adr_configs)
            print('\n\n')
            print('------------------------------------')
            print(f'{counter}/{num_combs} Training : ', combo_name, config)
            print('-----------------------------------')
            if config == 'char' and char_done:
                continue
            if config == 'char':
                char_done = True

            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                os.remove(file_path)

            train_path = args.char_input_paths.split(',')[0]
            df_text = pd.read_csv(train_path, encoding="utf-8", delimiter="\t")
            df_text['target'] = df_text['target'].map(lambda x: 0 if x == '!CC' else 1)


            # get adr train test labels
            test_path = args.char_input_paths.split(',')[1]
            df_test = pd.read_csv(test_path, encoding='utf8', sep='\t')
            df_train = df_text.drop(['target'], axis=1)
            train_ids = df_train['id'].tolist()
            test_ids = df_test['id'].tolist()

            # get adr_input
            if config == 'audio':
                adr_input, dur_input, id2idx = get_adr_input(args.text_features_folder, args.audio_features_folder, args.embeddings_path, audio=True, text=False)
            elif config == 'text':
                adr_input, dur_input, id2idx = get_adr_input(args.text_features_folder, args.audio_features_folder, args.embeddings_path, audio=False, text=True)
            else:
                adr_input, dur_input, id2idx = get_adr_input(args.text_features_folder, args.audio_features_folder, args.embeddings_path, audio=True, text=True)

            features = create_adr_features(adr_input, dur_input, id2idx, num_clusters, combo)
            feature_names = ['id'] + ['embeds_' + str(i) for i in range(len(features[0]) - 1)]
            df_adr = pd.DataFrame(features, columns=feature_names)
            df_train_adr = df_adr[df_adr.id.isin(train_ids)]
            df_test_adr = df_adr[df_adr.id.isin(test_ids)]
            df_data = df_text.merge(df_train_adr, on='id')
            df_data = df_data.sample(frac=1, random_state=1234)
            train_ids = df_data['id'].tolist()
            y = df_data['target'].values
            X = df_data.drop(['id', 'target'], axis=1)

            all_preds = []

            for rs in range(50):
                rfc = RandomForestClassifier(random_state=rs, n_estimators=50, max_depth=5)
                name, learner = ('rfc', rfc)
                character_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(4, 4), lowercase=False, min_df=1, max_df=0.8)
                tfidf_transformer = TfidfTransformer(sublinear_tf=True)


                features = [
                    ('embeds ADR', digit_features(key='embeds'))
                ]

                if config == 'char':
                    features = [('Character 4-grams', pipeline.Pipeline([('s5', text_col(key='!PAR')),
                                                                         ('character_vectorizer', character_vectorizer),
                                                                         ('tfidf_character', tfidf_transformer)]))]
                elif config == 'audio+text+char':
                    features.append(('Character 4-grams', pipeline.Pipeline([('s5', text_col(key='!PAR')),
                                                                             ('character_vectorizer', character_vectorizer),
                                                                             ('tfidf_character', tfidf_transformer)])))
                clf = pipeline.Pipeline([
                    ('union', FeatureUnion(
                        transformer_list=features,
                        n_jobs=1
                    )),
                    ('scale', Normalizer()),
                    ('lr', learner)])

                kfold = model_selection.KFold(n_splits=len(df_data))

                predicted_all_folds = []

                for train_index, test_index in kfold.split(df_data, y):

                    df_train, df_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    clf.fit(df_train, y_train)
                    preds = clf.predict(df_test)
                    predicted_all_folds.extend(preds)

                all_preds.append(predicted_all_folds)
                acc = accuracy_score(y, predicted_all_folds)

                if config == 'audio':
                    audio_acc_cv.append(acc)
                elif config == 'text':
                    text_acc_cv.append(acc)
                elif config == 'audio+text':
                    audio_text_acc_cv.append(acc)
                elif config == 'audio+text+char':
                    audio_text_char_acc_cv.append(acc)
                elif config == 'char':
                    char_acc_cv.append(acc)

                clf.fit(X, y)
                pickle.dump(clf, open(os.path.join(args.model_folder,'model_alg:' + name + '_acc:' + str(acc)[:5] + '_feat:' + combo_name + '_' + str(rs) + '.pkl'), 'wb'))
                print('Random seed: ', rs, 'Accuracy: ', acc)

            majority_preds = []
            for i in range(len(all_preds[0])):
                example_preds = []
                for j in range(len(all_preds)):
                    example_preds.append(all_preds[j][i])
                num_ones = example_preds.count(1)
                num_zeros = example_preds.count(0)
                if num_ones > num_zeros:
                    majority_preds.append(1)
                else:
                    majority_preds.append(0)

            maj_acc = accuracy_score(y, majority_preds)
            train_ids = [x + ' ' for x in train_ids]
            df_results = pd.DataFrame({"ID": train_ids, "Prediction": majority_preds})
            df_results.to_csv(f'{results_folder}/{combo_name}_{config}_results_cv.csv', index=False, header=True, sep=';')
            print('Majority vote accuracy CV: ', maj_acc)
            output_accuracies_cv.write(f'{combo_name}_{config};{maj_acc}\n')

            print('------------------------------------')
            print('Testing: ', combo_name, config)
            print('-----------------------------------')

            test_path = args.char_input_paths.split(',')[1]
            df_text = pd.read_csv(test_path, encoding="utf-8", delimiter="\t")
            df_data = df_text.merge(df_test_adr, on='id')
            df_data = df_data.sort_values(by=['id'])

            ids = df_data['id']
            true = df_data['target'].values
            X = df_data.drop(['id', 'target'], axis=1)

            trained_classification_models = os.listdir(args.model_folder)
            trained_classification_models = [os.path.join(args.model_folder, name) for name in trained_classification_models]

            trained_models = trained_classification_models

            all_preds = []

            for tm in trained_models:
                clf = joblib.load(tm)
                preds = clf.predict(X)
                all_preds.append(preds)
                s_ids = [x + ' ' for x in ids]
                s_preds = [" " + str(x) for x in preds]
                acc = accuracy_score(true, preds)
                print('Accuracy: ', acc)
                if config == 'audio':
                    audio_acc.append(acc)
                elif config == 'text':
                    text_acc.append(acc)
                elif config == 'audio+text':
                    audio_text_acc.append(acc)
                elif config == 'audio+text+char':
                    audio_text_char_acc.append(acc)
                elif config == 'char':
                    char_acc.append(acc)

            majority_preds = []
            for i in range(len(all_preds[0])):
                example_preds = []
                for j in range(len(all_preds)):
                    example_preds.append(all_preds[j][i])
                num_ones = example_preds.count(1)
                num_zeros = example_preds.count(0)
                if num_ones > num_zeros:
                    majority_preds.append(1)
                else:
                    majority_preds.append(0)

            df_results = pd.DataFrame({"ID": s_ids, "Prediction": majority_preds})
            df_results.to_csv(f'{results_folder}/{combo_name}_{config}_results.csv', index=False, header=True, sep=';')

            maj_acc = accuracy_score(true, majority_preds)
            print('Majority vote accuracy CV: ', maj_acc)
            output_accuracies.write(f'{combo_name}_{config};{maj_acc}\n')

        visualize(combo_name, audio_acc, text_acc, audio_text_acc, audio_text_char_acc, char_acc)
        visualize(combo_name + '_cv', audio_acc_cv, text_acc_cv, audio_text_acc_cv, audio_text_char_acc_cv, char_acc_cv)

    output_accuracies_cv.close()
    output_accuracies.close()




