from models import *
from datasets import *
from escnn import gspaces
import random
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

def create_dataloaders(data_dir: str, batch_size : int, resolution: int = 8, num_workers : int = 2, interpolation = "nearest", samples=7000, data_augmentation_group=None, seed=0):
    generator = torch.Generator()
    generator.manual_seed(seed)
    dict_dataloaders = dict()
    for kind in ["train", "valid", "test"]:
        dataset = VelocityDataset(
            os.path.join(data_dir, kind),
            resolution,
            interpolation=interpolation,
            data_augmentation_group=data_augmentation_group,
            generator=generator
        )
        dict_dataloaders[kind] = DataLoader(
            dataset,
            batch_size,
            drop_last = True if kind == "train" else False,
            sampler = SubsetRandomSampler(range(samples), generator) if kind in ["train"] else None,
            pin_memory=True,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=generator
        )
    return dict_dataloaders

def create_model(name):
    if name == "DSCMS":
        model = DSCMS(2, 2, 0.5)
    elif name == "DSCMS_C4":
        r2_act = gspaces.rot2dOnR2(N=4)
        model = EqDSCMS(
            enn.FieldType(r2_act, [r2_act.irrep(1)]),
            enn.FieldType(r2_act, [r2_act.irrep(1)]),
            enn.FieldType(r2_act, 12*[r2_act.regular_repr]),
            enn.FieldType(r2_act,  2*[r2_act.regular_repr]),
            enn.FieldType(r2_act,  2*[r2_act.regular_repr]),
            enn.FieldType(r2_act,  2*[r2_act.regular_repr]),
            enn.FieldType(r2_act,  2*[r2_act.regular_repr]),
        )
    elif name == "DSCMS_C8":
        r2_act = gspaces.rot2dOnR2(N=8)
        model = EqDSCMS(
            enn.FieldType(r2_act, [r2_act.irrep(1)]),
            enn.FieldType(r2_act, [r2_act.irrep(1)]),
            enn.FieldType(r2_act, 8*[r2_act.regular_repr]),
            enn.FieldType(r2_act,  2*[r2_act.regular_repr]),
            enn.FieldType(r2_act,  2*[r2_act.regular_repr]),
            enn.FieldType(r2_act,  2*[r2_act.regular_repr]),
            enn.FieldType(r2_act,  2*[r2_act.regular_repr]),
        )
    elif name == "DSCMS_D4":
        r2_act = gspaces.flipRot2dOnR2(N=4)
        model = EqDSCMS(
            enn.FieldType(r2_act, [r2_act.irrep(1,1)]),
            enn.FieldType(r2_act, [r2_act.irrep(1,1)]),
            enn.FieldType(r2_act, 8*[r2_act.regular_repr]),
            enn.FieldType(r2_act,  2*[r2_act.regular_repr]),
            enn.FieldType(r2_act,  2*[r2_act.regular_repr]),
            enn.FieldType(r2_act,  2*[r2_act.regular_repr]),
            enn.FieldType(r2_act,  2*[r2_act.regular_repr]),
        )
    elif name == "DSCMS_SO2":
        model = FullDSCMS()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Loaded {name} with {trainable_params} trainable parameters.")
    return model

def get_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device.")
    return device

def save_checkpoint(path, weights, optimizer_state, early_stop_count):
    torch.save({
        'model_state_dict': weights,
        'optimizer_state_dict': optimizer_state,
        'early_stop_count': early_stop_count
    }, path)

def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)