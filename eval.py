import copy

import torch
from lightning import Fabric
from torchmetrics import Accuracy
from tqdm import tqdm

import mobileclip
from dataset import LPCVDataset
from train import setup_environ, compress_model, create_test_transform, GROUP1, GROUP2

if __name__ == '__main__':
    setup_environ()

    fabric = Fabric(accelerator='cuda', devices=[0], precision='16-mixed')
    fabric.launch()

    model, _, _ = mobileclip.create_model_and_transforms(f'mobileclip_s1',
                                                         pretrained=f'weight/mobileclip_s1.pt',
                                                         reparameterize=False)
    logit_scale = model.logit_scale
    model = model.image_encoder.model
    original_model = copy.deepcopy(model)

    model = compress_model(model, 4, GROUP1, GROUP2)
    model.load_state_dict(torch.load('weight/1st_weight.pt', weights_only=True)['state_dict'])

    model_parameters = sum(p.numel() for p in model.parameters())
    original_model_parameters = sum(p.numel() for p in original_model.parameters())
    print(f"Original Model Parameters: {original_model_parameters / 1024 ** 2:.1f}M")
    print(f"Compressed Model Parameters: {model_parameters / 1024 ** 2:.1f}M")
    print(f"Compressing Ratio: {(original_model_parameters - model_parameters) / original_model_parameters * 100:.1f}%")

    test_ds = LPCVDataset('./coco', 'lpcv_data_test.csv', create_test_transform())
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=256, num_workers=8)

    model = fabric.setup(model)
    original_model = fabric.setup(original_model)
    test_dl = fabric.setup_dataloaders(test_dl)

    text_features = torch.load(f'weight/text_embedding_mobileclip_s1.pt', map_location=fabric.device, weights_only=True)

    accuracy = Accuracy('multiclass', num_classes=64).to(fabric.device)
    original_accuracy = Accuracy('multiclass', num_classes=64).to(fabric.device)

    for item in tqdm(test_dl, total=len(test_dl), desc='Inference'):
        img, target = item
        with fabric.autocast() and torch.no_grad():
            image_features = model(img)
            original_image_features = original_model(img)

        logit = torch.mm(image_features, text_features.T)
        accuracy.update(logit, target)

        original_logit = torch.mm(original_image_features, text_features.T)
        original_accuracy.update(original_logit, target)

    original_accuracy = original_accuracy.compute().item()
    accuracy = accuracy.compute().item()
    print(f'Original Model Accuracy: {original_accuracy * 100:.2f}')
    print(f'Compressed Model Accuracy: {accuracy * 100:.2f}')
    print(f'Accuracy Reduction Ratio: {(original_accuracy - accuracy) / original_accuracy * 100:.1f}%')
