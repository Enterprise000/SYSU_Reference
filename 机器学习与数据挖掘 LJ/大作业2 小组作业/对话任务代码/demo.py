import torch

from gpt_model import GPT

if __name__ == '__main__':
    device = torch.device('cuda')
    model = GPT().to(device)
    model.load_state_dict(torch.load('GPT2.pt')['model_state_dict'], strict=False)
    print(model)




    model.eval()
    sentence = ''
    while True:
        temp_sentence = input("我:")
        sentence += (temp_sentence + '\t')
        if len(sentence) > 200:
            # 裁剪
            t_index = sentence.find('\t')
            sentence = sentence[t_index + 1:]
        print("GPT:", model.answer(sentence))
