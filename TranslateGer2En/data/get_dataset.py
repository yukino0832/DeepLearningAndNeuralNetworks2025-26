from download import download
import re

url = "https://modelscope.cn/api/v1/datasets/SelinaRR/Multi30K/repo?Revision=master&FilePath=Multi30K.zip"

download(url, './', kind='zip', replace=True)

datasets_path = './datasets/'
train_path = datasets_path + 'train/'
valid_path = datasets_path + 'valid/'
test_path = datasets_path + 'test/'

def print_data(data_file_path, print_n=5):
    print("=" * 40 + "datasets in {}".format(data_file_path) + "=" * 40)
    with open(data_file_path, 'r', encoding='utf-8') as en_file:
        en = en_file.readlines()[:print_n]
        for index, seq in enumerate(en):
            print(index, seq.replace('\n', ''))


print_data(train_path + 'train.de')
print_data(train_path + 'train.en')