import os
from collections import defaultdict
import re
import argparse

def make_file(text, id, output):
    pattern = r'\[[:*+<>=].*?\]'
    text = re.sub(pattern, '', text)
    text = text.replace('[/]','').replace('[//]','').replace('[+ gram]', '').replace('&=clears throat', '').replace('=sings', '')\
        .replace('=laughs', '').replace('=clears:throat', '').replace('=sighs', '').replace('=hums', '').replace('=chuckles', '')\
        .replace('=grunt', '').replace('=finger:tap', '').replace('=claps', '').replace('=snif', '').replace('=coughs', '').replace('=tapping', '')
    text = text.replace('(.)', '').replace('(..)', '').replace('(..)', '').replace('<', '').replace('>', '').replace('/', '').replace('xxx', '').replace('+', '')
    text = text.replace('(', '').replace(')', '').replace('&', '')
    text = " ".join(text.split())
    while '[x' in text:
        text = text.split()
        for idx, token in enumerate(text):
            if token == '[x':
                count = int(text[idx + 1][:-1])
                word = text[idx - 1]
                before_text = text[:idx]
                after_text = text[idx + 2:]
                words = [word for _ in range(count - 1)]
                new_text = before_text + words + after_text
                break
        text = " ".join(new_text)
    p = re.compile(r'.*?')
    start_idx = 0
    for m in p.finditer(text):
        name = m.group()
        end_idx = m.start()
        chunk = text[start_idx: end_idx].strip()
        start_idx = end_idx + len(name)
        name = name.replace('', '').replace('_', '-')
        with open(output + '/' + id + '-' + name + '.txt', 'w', encoding='utf8') as f:
            for word in chunk.split():
                f.write(word.upper() + '\n')
        f.close()



def parse_transcripts(input_folder, output_path):
    groups = ['cc', 'cd']
    for group in groups:
        path = os.path.join(input_folder, group)
        filenames = os.listdir(path)
        for filename in filenames:
            file_path =  os.path.join(path, filename)
            representations = defaultdict(list)

            with open(file_path, 'r', encoding='utf8') as f:
                inv_bool = True
                last_rep = ""

                for line in f:
                    if line.startswith('@End') or line.startswith('@Begin') or line.startswith('@Languages') or line.startswith('@Participants') or line.startswith('@Media') or line.startswith('@Comment'):
                        pass
                    elif line.startswith('@UTF8') or line.startswith('%com'):
                        pass
                    elif line.startswith('@ID'):
                        if 'Participant' in line:
                            meta = line.split('Participant')[0].split('PAR')[-1]
                    elif line.startswith('@PID'):
                        num = line.split('\t')[-1]
                    elif line.startswith('*PAR'):
                        last_rep = 'PAR'
                        inv_bool = False
                        representations[last_rep].append(line.split('\t')[-1].strip())
                    elif line.startswith('*INV'):
                        last_rep = 'INV'
                        inv_bool = True
                        representations[last_rep].append(line.split('\t')[-1].strip())
                    elif line.startswith('%mor'):
                        line = line.split('\t')[-1].strip()
                        if inv_bool:
                            last_rep = 'INV_MORPH'
                            representations[last_rep].append(line)
                        else:
                            last_rep = 'PAR_MORPH'
                            representations[last_rep].append(line)
                    elif line.startswith('%gra'):
                        line = line.split('\t')[-1].strip()
                        if inv_bool:
                            last_rep = 'INV_GRA'
                            representations[last_rep].append(line)
                        else:
                            last_rep = 'PAR_GRA'
                            representations[last_rep].append(line)
                    else:
                        line = line.strip()
                        if inv_bool:
                            representations[last_rep][-1] = representations[last_rep][-1] + ' ' + line
                        else:
                            representations[last_rep][-1] = representations[last_rep][-1] + ' ' + line

                id = filename.split('.')[0]
                if group == 'cd':
                    c = '!CD'
                if group == 'cc':
                    c = '!CC'

                meta = '!PARMETA' + meta.strip() + num.strip()

                if representations['PAR']:
                    if not os.path.exists(os.path.join(output_path, group)):
                        os.makedirs(os.path.join(output_path, group))
                    make_file(" ".join(representations['PAR']), id.strip(), os.path.join(output_path, group))



def parse_test_transcripts(input_folder, output_path):
    path = input_folder
    filenames = os.listdir(path)
    for filename in filenames:
        file_path =  os.path.join(path, filename)
        representations = defaultdict(list)

        with open(file_path, 'r', encoding='utf8') as f:
            inv_bool = True
            last_rep = ""

            for line in f:
                if line.startswith('@End') or line.startswith('@Begin') or line.startswith('@Languages') or line.startswith('@Participants') or line.startswith('@Media') or line.startswith('@Comment'):
                    pass
                elif line.startswith('@UTF8') or line.startswith('%com'):
                    pass
                elif line.startswith('@ID'):
                    if 'Participant' in line:
                        meta = line.split('Participant')[0].split('PAR')[-1]
                elif line.startswith('@PID'):
                    num = line.split('\t')[-1]
                elif line.startswith('*PAR'):
                    last_rep = 'PAR'
                    inv_bool = False
                    representations[last_rep].append(line.split('\t')[-1].strip())
                elif line.startswith('*INV'):
                    last_rep = 'INV'
                    inv_bool = True
                    representations[last_rep].append(line.split('\t')[-1].strip())
                elif line.startswith('%mor'):
                    line = line.split('\t')[-1].strip()
                    if inv_bool:
                        last_rep = 'INV_MORPH'
                        representations[last_rep].append(line)
                    else:
                        last_rep = 'PAR_MORPH'
                        representations[last_rep].append(line)
                elif line.startswith('%gra'):
                    line = line.split('\t')[-1].strip()
                    if inv_bool:
                        last_rep = 'INV_GRA'
                        representations[last_rep].append(line)
                    else:
                        last_rep = 'PAR_GRA'
                        representations[last_rep].append(line)
                else:
                    line = line.strip()
                    if inv_bool:
                        representations[last_rep][-1] = representations[last_rep][-1] + ' ' + line
                    else:
                        representations[last_rep][-1] = representations[last_rep][-1] + ' ' + line

            id = filename.split('.')[0]

            meta = '!PARMETA' + meta.strip() + num.strip()

            if representations['PAR']:
                make_file(" ".join(representations['PAR']), id.strip(), output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path_train', type=str, default='data/ADReSS-IS2020-data/train/transcription', help='path to train transcription folder')
    parser.add_argument('--output_path_train', type=str, default='data/sentence_aligned', help='output folder for parsed train transcriptions')
    parser.add_argument('--input_path_test', type=str, default='data/ADReSS-IS2020-data/test/transcription',
                        help='path to test transcription folder')
    parser.add_argument('--output_path_test', type=str, default='data/sentence_aligned/test',
                        help='output folder for parsed test transcriptions')
    args = parser.parse_args()

    if not os.path.exists(args.output_path_train):
        os.makedirs(args.output_path_train)
    if not os.path.exists(args.output_path_test):
        os.makedirs(args.output_path_test)

    parse_transcripts(args.input_path_train, args.output_path_train)
    parse_test_transcripts(args.input_path_test, args.output_path_test)








