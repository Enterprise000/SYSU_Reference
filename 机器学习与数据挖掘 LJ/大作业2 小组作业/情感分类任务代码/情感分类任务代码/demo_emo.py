import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from gpt_model import *


class TestDataset(Dataset):
    """
    测试数据集类。
    """

    def __init__(self, reviews, labels, word2id):
        self.reviews = reviews
        self.labels = labels
        self.word2id = word2id

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):
        review = self.reviews[index]
        label = self.labels[index]

        # 转换文本为 ID
        input_ids = [self.word2id.get(word, self.word2id['<unk>']) if word != ' ' else self.word2id['<sep>']
                     for word in review]
        input_ids += [self.word2id['<sep>']]

        return input_ids, label

    @staticmethod
    def collate_fn(batch):
        inputs, labels = zip(*batch)
        max_len = max(len(seq) for seq in inputs)

        # padding
        padded_inputs = [seq + [word2id['<pad>']] * (max_len - len(seq)) for seq in inputs]
        inputs_tensor = torch.tensor(padded_inputs, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return inputs_tensor, labels_tensor


def predict_with_batches(model, data_loader, device):
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs, _ = model(inputs, classify=True)
            predictions = torch.argmax(outputs, dim=1).tolist()
            all_predictions.extend(predictions)

    return all_predictions


def calculate_accuracy(predictions, labels):
    # 计算预测结果的准确率
    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(labels)


if __name__ == "__main__":
    # 加载微调后的模型
    device = torch.device('cuda')
    model = GPT().to(device)
    model.load_state_dict(torch.load('GPT2_emo_last_banmask.pt', weights_only=True))

    # 加载测试数据
    test_file = "emo_less300.csv"
    dataframe = pd.read_csv(test_file)
    df = dataframe.sample(frac=1, random_state=18).reset_index(drop=True)
    reviews = df['review'].astype(str).tolist()
    labels = df['label'].astype(int).tolist()

    # 创建数据集和 DataLoader
    batch_size = 8
    dataset = TestDataset(reviews, labels, word2id)
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=TestDataset.collate_fn)

    # 分批预测
    predictions = predict_with_batches(model, data_loader, device)

    # 计算准确率
    accuracy = calculate_accuracy(predictions, labels)
    print(f"测试集准确率: {accuracy * 100:.2f}%")
