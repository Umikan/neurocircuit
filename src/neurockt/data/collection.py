import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer


class Category(Dataset):
    def __init__(self, df, encoder):
        self.raw_labels = list(df)
        self.encoder = encoder
        self.labels = self.encoder.fit_transform(self.raw_labels)

    def get_vocab(self):
        return list(self.encoder.classes_)
    
    def num_classes(self):
        return len(list(self.encoder.classes_))
    
    def __len__(self):
        return len(self.labels)
    

class Multiclass(Category):
    def __init__(self, df):
        super().__init__(df, LabelEncoder())

    def __getitem__(self, idx):
        return torch.tensor(self.labels[idx])


class Multilabel(Category):
    def __init__(self, df):
        super().__init__(df, MultiLabelBinarizer())
        
    def __getitem__(self, idx):
        return torch.LongTensor(self.labels[idx])
    

# with albumentation library
class Image(Dataset):
    def __init__(self, df, transform=None):
        self.path = list(df)
        self.transform = transform
    
    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        img_path = self.path[idx] 
        image = read_image(img_path)
        return self.transform(image).float() if self.transform else image.float()
