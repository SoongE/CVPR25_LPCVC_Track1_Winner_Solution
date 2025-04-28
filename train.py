import argparse
import copy
import os

import torch
from lightning import Fabric
from torch.optim.lr_scheduler import StepLR
from torchmetrics import MeanMetric, Accuracy
from torchvision.transforms import v2

import mobileclip
from dataset import LPCVDataset


def forward_for_imitation(self, x: torch.Tensor, idx) -> torch.Tensor:
    x = self.forward_embeddings(x)
    x = self.network[0](x)
    x = self.network[1](x)
    x = self.network[2](x)
    x = self.network[3](x)
    # x = self.network[4](x)
    for i, block in enumerate(self.network[4]):
        x = block(x)
        if i == idx: break

    return x


def average_layer_weight(layer1, layer2):
    def _loop(module1, module2):
        for (name1, child1), (name2, child2) in zip(module1.named_children(), module2.named_children()):
            if isinstance(child1, torch.nn.Conv2d):
                copy_child = copy.deepcopy(child1)
                copy_child.weight.data = (child1.weight.data + child2.weight.data) / 2
                if copy_child.bias is not None:
                    copy_child.bias.data = (child1.bias.data + child2.bias.data) / 2
                setattr(module1, name1, copy_child)
            elif isinstance(child1, torch.nn.BatchNorm2d):
                copy_child = copy.deepcopy(child1)
                copy_child.weight.data = (child1.weight.data + child2.weight.data) / 2
                copy_child.bias.data = (child1.bias.data + child2.bias.data) / 2
                copy_child.running_mean = (child1.running_mean + child2.running_mean) / 2
                copy_child.running_var = (child1.running_var + child2.running_var) / 2
                setattr(module1, name1, copy_child)
            elif isinstance(child1, torch.nn.Linear):
                copy_child = copy.deepcopy(child1)
                copy_child.weight.data = (child1.weight.data + child2.weight.data) / 2
                if copy_child.bias is not None:
                    copy_child.bias.data = (child1.bias.data + child2.bias.data) / 2
                setattr(module1, name1, copy_child)
            elif isinstance(child1, torch.nn.Parameter):
                copy_child = (child1 + child2) / 2
                setattr(module1, name1, copy_child)
            else:
                _loop(child1, child2)

    _loop(layer1, layer2)
    return layer1


def compress_model(model, target_idx, group1, group2):
    print(f'Before Combine: {len(model.network[target_idx])} layers')

    def _resolve(node):
        if isinstance(node, int):
            return model.network[target_idx][node]
        elif isinstance(node, list) and len(node) == 2:
            left = _resolve(node[0])
            right = _resolve(node[1])
            return average_layer_weight(left, right)
        else:
            raise ValueError(f"Invalid tree node structure: {node}")

    combined_layer_1 = _resolve(group1)
    combined_layer_2 = _resolve(group2)

    min_idx = group1[0]
    max_idx = group2[-1]

    model.network[target_idx] = model.network[target_idx][:min_idx + 1] + model.network[target_idx][max_idx:]
    model.network[target_idx][min_idx] = combined_layer_1
    model.network[target_idx][min_idx + 1] = combined_layer_2

    print(f'After Combine: {len(model.network[target_idx])} layers')
    return model


def setup_environ():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def seed_everything(seed, local_rank=0, workers=0):
    import random
    import numpy

    each_seed = seed + local_rank

    os.environ['PL_GLOBAL_SEED'] = str(seed)
    os.environ['PL_SEED_WORKERS'] = str(workers)

    random.seed(each_seed)
    numpy.random.seed(each_seed)
    torch.manual_seed(each_seed)
    torch.cuda.manual_seed_all(each_seed)


def create_train_transform():
    return v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(0.4, 0.4, 0.4),
        v2.RandAugment(num_ops=2, magnitude=10),
        v2.ToDtype(torch.float32, scale=True),
    ])


def create_test_transform():
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(224),
        v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale=True),
    ])


GROUP1 = [5, [[6, 7], [8, 9]]]
GROUP2 = [[[10, 11], [12, 13]], 14]
parser = argparse.ArgumentParser()
parser.add_argument('-lr', '--lr', type=float, default=1.25e-3)
parser.add_argument('-wd', '--weight-decay', type=float, default=1e-2)
parser.add_argument('-bs', '--batch-size', type=int, default=128)
parser.add_argument('-epoch', '--epoch', type=int, default=100)
parser.add_argument('--gpu', nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7])

if __name__ == '__main__':
    setup_environ()

    cfg = parser.parse_args()
    fabric = Fabric(accelerator='cuda', devices=cfg.gpu, precision='16-mixed',
                    strategy='ddp_find_unused_parameters_true')
    fabric.launch()

    seed_everything(seed=42, local_rank=fabric.local_rank)

    model, _, _ = mobileclip.create_model_and_transforms(f'mobileclip_s1',
                                                         pretrained=f'weight/mobileclip_s1.pt',
                                                         reparameterize=False)
    logit_scale = model.logit_scale
    model = model.image_encoder.model

    bound_method = forward_for_imitation.__get__(model, model.__class__)
    setattr(model, 'forward_for_imitation', bound_method)

    original_model = copy.deepcopy(model)
    model = compress_model(model, 4, GROUP1, GROUP2)

    parameters = list()
    parameters.append({'params': model.network[4][5].parameters()})
    parameters.append({'params': model.network[4][6].parameters()})
    optimizer = torch.optim.AdamW(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = torch.nn.MSELoss()

    train_ds = LPCVDataset('./coco', 'lpcv_data_train.csv', create_train_transform())
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=8,
                                           pin_memory=True)

    test_ds = LPCVDataset('./coco', 'lpcv_data_test.csv', create_test_transform())
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=cfg.batch_size, num_workers=8)
    fabric.print(f'Train/Test Dataset Size: {len(train_ds)}/{len(test_ds)}')

    model, optimizer = fabric.setup(model, optimizer)
    original_model = fabric.setup_module(original_model)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

    model.mark_forward_method('forward_for_imitation')
    original_model.mark_forward_method('forward_for_imitation')

    train_dl, test_dl = fabric.setup_dataloaders(train_dl, test_dl)
    total_len = len(train_dl)
    text_features = torch.load(f'weight/text_embedding_mobileclip_s1.pt', map_location=fabric.device, weights_only=True)

    losses = MeanMetric().to(fabric.device)
    accuracy = Accuracy('multiclass', num_classes=64).to(fabric.device)

    best_acc = 0.
    for epoch in range(cfg.epoch):
        # LAYER IMITATION TRAIN
        scheduler.step()
        losses.reset()
        accuracy.reset()
        for i, item in enumerate(train_dl):
            img, target = item
            optimizer.zero_grad()
            with fabric.autocast():
                image_features = model.forward_for_imitation(img, 6)
                with torch.no_grad():
                    origin_image_features = original_model.forward_for_imitation(img, 14)
                loss = criterion(image_features, origin_image_features)

            fabric.backward(loss)
            optimizer.step()

            losses.update(loss)
            if i % 50 == 0:
                computed_loss = losses.compute().item() * 100000
                fabric.print(f'LAYER IMITATION TRAIN: [{i}/{total_len}] Loss: {computed_loss:.3f}')

        # EVAL
        losses.reset()
        accuracy.reset()
        for i, item in enumerate(test_dl):
            img, target = item
            with fabric.autocast() and torch.no_grad():
                image_features = model(img)
            logit = torch.mm(image_features, text_features.T)
            accuracy.update(logit, target)

        computed_acc = accuracy.compute().item()
        if epoch != cfg.epoch - 1:
            fabric.print(f'EVAL: [{epoch}/{cfg.epoch}] Accuracy: {computed_acc:.3f}')
        else:
            fabric.print(f'FINAL Accuracy: {computed_acc:.3f}')

        if computed_acc > best_acc:
            fabric.save(f'output/{epoch}_{int(computed_acc * 1000)}.pt', {'state_dict': model.state_dict()})
            best_acc = computed_acc
