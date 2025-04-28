import torch

import mobileclip
from constant import lpcv_class_names, PROMPTS


def extract_text_features(model, tokenizer, device):
    text_embedding = list()
    for class_name in lpcv_class_names:
        prompts = [p(class_name.lower()) for p in PROMPTS]
        prompts = tokenizer(prompts).to(device)
        with torch.no_grad():
            prompts = model.encode_text(prompts)
        text_embedding.append(prompts.mean(0))
    text_embedding = torch.stack(text_embedding, dim=0)
    text_embedding = torch.nn.functional.normalize(text_embedding, dim=-1)

    torch.save(text_embedding.detach().cpu(), 'weight/text_embedding_mobileclip_s1.pt')
    print("Text features are saved to weight/text_embedding_mobileclip_s1.pt")

    return text_embedding


if __name__ == '__main__':
    device = torch.device('cuda:0')
    model, _, _ = mobileclip.create_model_and_transforms('mobileclip_s1', f'weight/mobileclip_s1.pt',
                                                         reparameterize=True)
    model.to(device)
    model.eval()
    tokenizer = mobileclip.get_tokenizer('mobileclip_s0')

    text_f = extract_text_features(model, tokenizer, device)
