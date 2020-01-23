from torch.utils.data import Dataset


################################################
# TODO: Construct custom dataset
class custom1(Dataset):
    def __init__(self, root_dir, dataset_type):
        print('Custom dataset')

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return 0