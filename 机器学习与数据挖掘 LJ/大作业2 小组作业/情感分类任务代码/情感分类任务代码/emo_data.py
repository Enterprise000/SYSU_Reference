import pandas as pd

# 加载文件
df = pd.read_csv('emo.csv')

# 定义一个函数来检查句子长度是否超过200
def is_too_long(text):
    # 确保text是字符串类型
    if isinstance(text, str):
        return len(text) > 200
    else:
        return False  # 如果不是字符串，就认为它不是过长的文本

# 应用函数，过滤掉长度超过300的句子
df_filtered = df[df['review'].apply(is_too_long) == False]
df_shuffled = df_filtered.sample(frac=1, random_state=42).reset_index(drop=True)

# 保存过滤后的数据到新的CSV文件
df_shuffled.to_csv('emo_less200.csv', index=False)