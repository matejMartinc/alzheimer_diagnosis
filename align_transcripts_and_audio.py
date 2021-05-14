from p2fa import align
import os
import argparse


def get_timestamps(train_files, test_files, output_folder):
    for idx, files in enumerate([train_files, test_files]):
        pairs = []
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                file1 = files[i].split('/')[-1].split('.')[0]
                file2 = files[j].split('/')[-1].split('.')[0]

                if file1 == file2:
                    sorter = file1.split('-')
                    s = "-"
                    num_zeros = 10 - len(sorter[1])
                    for _ in range(num_zeros):
                        s += '0'
                    sorter = sorter[0] + s + sorter[1]
                    if files[i].endswith('.txt'):
                        pairs.append((files[j], files[i], sorter))
                    else:
                        pairs.append((files[i], files[j], sorter))

        pairs = sorted(pairs, key=lambda x: x[-1])
        for wav, txt, _ in pairs:
            try:
                original = []
                with open(txt, 'r', encoding='utf8') as f:
                    for line in f:
                        original.append(line.strip())
                phoneme_alignments, word_alignments = align.align(wav, txt)
                output_file = txt.split('/')[-1]
                if idx == 0:
                    output_path = os.path.join(output_folder, 'train')
                if idx == 1:
                    output_path = os.path.join(output_folder, 'test')
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                with open(os.path.join(output_path, output_file), 'w', encoding='utf8') as f:
                    for word, start, stop in word_alignments:
                        f.write(word + ',' + str(start) + ',' + str(stop) + '\n')
                print('Alignment successful', wav, txt)
            except:
                print('Alignment failed: ', wav, txt)
            print('-----------------------------')
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_paths_train_text', type=str, default='sentence_aligned/cc,sentence_aligned/cd',
                        help='path to all folders containing train transcript files divided by comma')
    parser.add_argument('--input_paths_train_audio', type=str, default='sentence_aligned/cc,sentence_aligned/cd',
                        help='path to all folders containing train audio files divided by comma')
    parser.add_argument('--input_paths_test_text', type=str, default='sentence_aligned/test',
                        help='path to all folders containing test transcript files divided by comma')
    parser.add_argument('--input_paths_test_audio', type=str, default='sentence_aligned/test',
                        help='path to all folders containing test audio files divided by comma')
    parser.add_argument('--output_folder', type=str, default='data/word_timestamps', help='path to output folder')
    args = parser.parse_args()

    text_train = args.input_paths_train_text.split(',')
    audio_train = args.input_paths_train_audio.split(',')

    text_test = args.input_paths_test_text.split(',')
    audio_test = args.input_paths_test_audio.split(',')

    all_train_files = []
    for folder in text_train:
        files = os.listdir(folder)
        for f in files:
            if f.endswith('.txt'):
                all_train_files.append(os.path.join(folder, f))
    for folder in audio_train:
        files = os.listdir(folder)
        for f in files:
            if f.endswith('.wav'):
                all_train_files.append(os.path.join(folder, f))
    all_test_files = []
    for folder in text_test:
        files = os.listdir(folder)
        for f in files:
            if f.endswith('.txt'):
                all_test_files.append(os.path.join(folder, f))
    for folder in audio_test:
        files = os.listdir(folder)
        for f in files:
            if f.endswith('.wav'):
                all_test_files.append(os.path.join(folder, f))

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    get_timestamps(all_train_files, all_test_files, args.output_folder)




