import os
from collections import defaultdict
import re
import pandas as pd
import argparse

def build_csv(input_path, output_path):
    d = defaultdict(dict)
    with open(input_path, 'r', encoding="utf8") as f:
        for line in f:
            id, speaker, group, text = line.split('\t')
            text = " ".join(text.split()).strip()
            if speaker.startswith('!MORPH'):
                text = " ".join([word.split('|')[0] if '|' in word else word for word in text.split()])
            d[id][speaker] = text
            d[id]['target'] = group

    data = []
    for id, values in d.items():
        line = [id]
        line.append(values['target'])
        line.append(values['!PAR'])
        if '!INV' in values:
            line.append(values['!INV'])
        else:
            line.append('')
        line.append(values['!MORPH_PAR'])
        if '!MORPH_INV' in values:
            line.append(values['!MORPH_INV'])
        else:
            line.append('')
        line.append(values['!GRA_PAR'])
        if '!GRA_INV' in values:
            line.append(values['!GRA_INV'])
        else:
            line.append('')
        data.append(line)
    df = pd.DataFrame(data, columns=['id', 'target', '!PAR', '!INV', '!MORPH_PAR', '!MORPH_INV', '!GRA_PAR', '!GRA_INV'])
    df.to_csv(output_path, encoding="utf8", sep="\t", index=False)


def build_test_csv(input_path, output_path):
    d = defaultdict(dict)
    with open(input_path, 'r', encoding="utf8") as f:
        for line in f:
            id, speaker, text = line.split('\t')
            text = " ".join(text.split()).strip()
            if speaker.startswith('!MORPH'):
                text = " ".join([word.split('|')[0] if '|' in word else word for word in text.split()])
            d[id][speaker] = text

    data = []
    for id, values in d.items():
        line = [id]
        line.append(values['!PAR'])
        if '!INV' in values:
            line.append(values['!INV'])
        else:
            line.append('')
        line.append(values['!MORPH_PAR'])
        if '!MORPH_INV' in values:
            line.append(values['!MORPH_INV'])
        else:
            line.append('')
        line.append(values['!GRA_PAR'])
        if '!GRA_INV' in values:
            line.append(values['!GRA_INV'])
        else:
            line.append('')
        data.append(line)
    df = pd.DataFrame(data, columns=['id', '!PAR', '!INV', '!MORPH_PAR', '!MORPH_INV', '!GRA_PAR', '!GRA_INV'])
    print(df.shape)
    df.to_csv(output_path, encoding="utf8", sep="\t", index=False)



def parse_transcripts(input_folder, output_path):
    output = open(output_path, 'w', encoding='utf8')

    pattern = r'.*?'
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

                    line = re.sub(pattern, '', line)
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
                    instance = "\t".join([id.strip(), '!PAR', c.strip(), " ".join(representations['PAR'])])
                    output.write(instance + '\n')
                if representations['INV']:
                    instance = "\t".join([id.strip(), '!INV', c.strip(), " ".join(representations['INV'])])
                    output.write(instance + '\n')
                if representations['PAR_MORPH']:
                    instance = "\t".join([id.strip(), '!MORPH_PAR', c.strip(), " ".join(representations['PAR_MORPH'])])
                    output.write(instance + '\n')
                if representations['INV_MORPH']:
                    instance = "\t".join([id.strip(), '!MORPH_INV', c.strip(), " ".join(representations['INV_MORPH'])])
                    output.write(instance + '\n')
                if representations['PAR_GRA']:
                    instance = "\t".join([id.strip(), '!GRA_PAR', c.strip(), " ".join(representations['PAR_GRA'])])
                    output.write(instance + '\n')
                if representations['INV_GRA']:
                    instance = "\t".join([id.strip(), '!GRA_INV', c.strip()," ".join(representations['INV_GRA'])])
                    output.write(instance + '\n')
    output.close()


def parse_test_transcripts(input_folder, output_path):
    output = open(output_path, 'w', encoding='utf8')

    pattern = r'.*?'


    path = input_folder
    filenames = os.listdir(path)
    for filename in filenames:
        file_path =  os.path.join(path, filename)
        representations = defaultdict(list)

        with open(file_path, 'r', encoding='utf8') as f:
            inv_bool = True
            last_rep = ""

            for line in f:

                line = re.sub(pattern, '', line)
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
                instance = "\t".join([id.strip(), '!PAR', " ".join(representations['PAR'])])
                output.write(instance + '\n')
            if representations['INV']:
                instance = "\t".join([id.strip(), '!INV', " ".join(representations['INV'])])
                output.write(instance + '\n')
            if representations['PAR_MORPH']:
                instance = "\t".join([id.strip(), '!MORPH_PAR', " ".join(representations['PAR_MORPH'])])
                output.write(instance + '\n')
            if representations['INV_MORPH']:
                instance = "\t".join([id.strip(), '!MORPH_INV', " ".join(representations['INV_MORPH'])])
                output.write(instance + '\n')
            if representations['PAR_GRA']:
                instance = "\t".join([id.strip(), '!GRA_PAR', " ".join(representations['PAR_GRA'])])
                output.write(instance + '\n')
            if representations['INV_GRA']:
                instance = "\t".join([id.strip(), '!GRA_INV'," ".join(representations['INV_GRA'])])
                output.write(instance + '\n')
    output.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path_train', type=str, default='data/ADReSS-IS2020-data/train/transcription', help='path to train transcription folder')
    parser.add_argument('--output_path_train', type=str, default='data/train_text', help='output file path without file suffix')
    parser.add_argument('--input_path_test', type=str, default='data/ADReSS-IS2020-data/test/transcription',
                        help='path to test transcription folder')
    parser.add_argument('--output_path_test', type=str, default='data/test_text',
                        help='output file path without file suffix')
    args = parser.parse_args()


    parse_test_transcripts(args.input_path_test, args.output_path_test + '.txt')
    build_test_csv(args.output_path_test + '.txt', args.output_path_test + '.tsv')

    parse_transcripts(args.input_path_train, args.output_path_train + '.txt')
    build_csv(args.output_path_train + '.txt', args.output_path_train + '.tsv')








