import pandas as pd
import jieba
import re

# 加载停用词库
def load_stopwords(path):
    with open(path, 'r', encoding='utf-8') as file:
        stopwords = set([line.strip() for line in file.readlines()])
    return stopwords

# 清洗文本，去除停用词和标点符号
def clean_text(text, stopwords):
    # 确保text是字符串类型
    if isinstance(text, str):
        # 去除标点符号
        text = re.sub(r'[^\w\s]', '', text)
        # 分词
        words = jieba.cut(text)
        # 去除停用词
        filtered_words = [word for word in words if word not in stopwords and word != ' ']
        return ''.join(filtered_words)
    else:
        return ''

# 加载CSV文件
df = pd.read_csv('emo_less200.csv')

# 加载停用词库
stopwords = load_stopwords('stopwords_full.txt')

# 清洗文本
df['review'] = df['review'].apply(lambda x: clean_text(str(x), stopwords))

# 保存处理后的数据到新的CSV文件
df.to_csv('cleaned_emo.csv', index=False)