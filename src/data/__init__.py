from .dataset import InpaintingData

from torch.utils.data import DataLoader, Subset
from icecream import ic

def sample_data(loader): 
    while True:
        for batch in loader:
            yield batch


def create_loader(args): 
    dataset = InpaintingData(args)
    dataset = Subset(dataset, list((range(0,int(len(dataset)*args.subset_factor)))))
    if args.global_rank==0:
        ic(len(dataset))
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size//args.world_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    return sample_data(data_loader)