from itertools import count
import torch.nn.functional as F
import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from utils.transform import *

class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, root, trainsize, transform=True):
        self.trainsize = trainsize
        self.image_root = os.path.join(root, 'images/')
        self.gt_root = os.path.join(root, 'masks/')
        self.images = [self.image_root + f for f in os.listdir(self.image_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tif')]
        self.gts = [self.gt_root + f for f in os.listdir(self.gt_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tif')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.size = len(self.images)
        if transform==True:
            self.transform = transforms.Compose([
                   Resize((self.trainsize, self.trainsize)),
                   RandomHorizontalFlip(),
                   RandomVerticalFlip(),
                   RandomRotation(90),
                   RandomZoom((0.9, 1.1)),
                   #RandomCrop((self.trainsize, self.trainsize)),
                   ToTensor(),

               ])
        else:
            self.transform = transforms.Compose([
                    Resize((self.trainsize, self.trainsize)),
                    ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        name = self.images[index].split('/')[-1]
        data = {'image': image, 'label': gt}
        data = self.transform(data)
        return data

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

class PolypParisDataset(data.Dataset):
    """
    dataloader for paris-catogorical polyp segmentation tasks
    """
    def __init__(self, root, type_file, mapping_func, trainsize):
        self.image_root = os.path.join(root, 'images')
        self.gt_root = os.path.join(root, 'masks')
        self.type_path = os.path.join(root, type_file)
        self.trainsize = trainsize
        self.type_mapping = mapping_func
        with open(self.type_path, 'r') as f:
            #label_file的格式， （label_file image_label)
            self.names = list(map(lambda line: line.strip().split(' '), f))

        self.transform = transforms.Compose([
                   Resize((self.trainsize, self.trainsize)),
                   RandomHorizontalFlip(),
                   RandomVerticalFlip(),
                   RandomRotation(90),
                   RandomZoom((0.9, 1.1)),
                   #RandomCrop((self.trainsize, self.trainsize)),
                   ToTensor(),

               ])
    def get_types_for_all_imgs(self):
        types_for_all_imgs = list()
        for name, type in self.names:
            types_for_all_imgs.append(self.type_mapping[type])
        return types_for_all_imgs

    def get_all_imgs(self):
        all_imgs = list()
        for name, type in self.names:
            all_imgs.append(name)
        return all_imgs

    def __getitem__(self, index):
        name, type = self.names[index]
        type = self.type_mapping[type]

        image_path = os.path.join(self.image_root, name)
        gt_path = os.path.join(self.gt_root, name)
        image = self.rgb_loader(image_path)
        gt = self.binary_loader(gt_path)
        data = {'image': image, 'label': gt}
        data = self.transform(data)
        return data, type

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    
    def get_num_class(self):
        return len(self.type_mapping)

    def __len__(self):
        return len(self.names)
        
class PolypImbalanceDataset(data.Dataset):
    def __init__(self, root, type_file, mapping_func, trainsize):
        self.image_root = os.path.join(root, 'images')
        self.gt_root = os.path.join(root, 'masks')
        self.type_path = os.path.join(root, type_file)
        self.trainsize = trainsize
        self.type_mapping = mapping_func
        self.imgs = list()
        self.types = list()
        with open(self.type_path, 'r') as f:
            data = f.readlines()
        for d in data:
            #label_file的格式， （label_file image_label)
            img, type = d.strip().split()
            self.imgs.append(img)   
            self.types.append(self.type_mapping[type])
        assert len(self.imgs) == len(self.types)

        self.transform = transforms.Compose([
                   Resize((self.trainsize, self.trainsize)),
                   RandomHorizontalFlip(),
                   RandomVerticalFlip(),
                   RandomRotation(90),
                   RandomZoom((0.9, 1.1)),
                   #RandomCrop((self.trainsize, self.trainsize)),
                   ToTensor(),

               ])
    
    def __getitem__(self, index):
        name = self.imgs[index]
        type = self.types[index]
        image_path = os.path.join(self.image_root, name)
        gt_path = os.path.join(self.gt_root, name)
        image = self.rgb_loader(image_path)
        gt = self.binary_loader(gt_path)
        data = {'image': image, 'label': gt}
        data = self.transform(data)
        return data, type

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    
    def __num_class__(self):
        return len(self.type_mapping)

    def __len__(self):
        return len(self.imgs)

def get_naive_loader(root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True, transform=True):
    dataset = PolypDataset(root, trainsize=trainsize, transform=transform)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

def get_loader(root, batchsize, dataset_name, trainsize, shuffle=True, num_workers=4, pin_memory=True):
    if dataset_name == 'EndoScene':
        mapping = {'Is': 0, 'Ip': 1, 'Isp': 2, 'LST': 3}
    elif dataset_name == 'PICCOLO':
        mapping = {'IIa': 0, 'IIac': 1, 'Ip': 2, 'Is': 3, 'Isp': 4, 'unknown': 5}
    else:
        raise Exception("Invalid dataset name!")
    dataset = PolypParisDataset(root, type_file='types.txt', mapping_func=mapping, trainsize=trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

def get_balance_loader(root, batchsize, dataset_name, trainsize, shuffle=False, num_workers=4, pin_memory=True):
    if dataset_name == 'EndoScene':
        mapping = {'LST': 0, 'Ip': 1, 'Isp': 2, 'Is': 3}
        counts = [25, 78, 152, 292]
    elif dataset_name == 'PICCOLO':
        mapping = {'IIac': 0, 'unknown': 1, 'Isp': 2, 'Ip': 3, 'Is': 4, 'IIa': 5}
        counts = [27, 172, 245, 274, 433, 1052]
    else:
        raise Exception("Invalid dataset name!")
    
    dataset = PolypParisDataset(root, type_file='types.txt', mapping_func=mapping, trainsize=trainsize)
    weights = 1./ torch.tensor(counts, dtype=torch.float)
    train_targets = dataset.get_types_for_all_imgs()
    samples_weights = weights[train_targets]
    sampler = data.WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  sampler=sampler,
                                  pin_memory=pin_memory)
    return data_loader

def get_imbalance_loader(root, batchsize, dataset_name, trainsize, imb_factor, shuffle=False, num_workers=4, pin_memory=True):
    def get_img_num_per_cls(cls_num, total_num, imb_type, imb_factor):
        # This function is excerpted from a publicly available code [commit 6feb304, MIT License]:
        # https://github.com/kaidic/LDAM-DRW/blob/master/imbalance_cifar.py
        img_min = 1000 / imb_factor
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_min * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_min))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_min * imb_factor))
        else:
            img_num_per_cls.extend([int(img_min)] * cls_num)
        return img_num_per_cls
    def gen_imbalanced_data(img_num_per_cls, imgList, labelList):
        # This function is excerpted from a publicly available code [commit 6feb304, MIT License]:
        # https://github.com/kaidic/LDAM-DRW/blob/master/imbalance_cifar.py
        new_data = []
        new_targets = []
        targets_np = np.array(labelList, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)  # remove shuffle in the demo fair comparision
        num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            #np.random.shuffle(idx) # remove shuffle in the demo fair comparision
            selec_idx = idx[:the_img_num]
            for idx in selec_idx:
                new_data.append(imgList[idx])
            #new_data.append(imgList[selec_idx, ...])
            new_targets.extend([the_class, ] * len(selec_idx))
        return (new_data, new_targets)
        
    if dataset_name == 'EndoScene':
        mapping = {'LST': 0, 'Ip': 1, 'Isp': 2, 'Is': 3}
    elif dataset_name == 'PICCOLO':
        mapping = {'IIac': 0, 'unknown': 1, 'Isp': 2, 'Ip': 3, 'Is': 4, 'IIa': 5}
    else:
        raise Exception("Invalid dataset name!")
    
    dataset = PolypImbalanceDataset(root, type_file='types.txt', mapping_func=mapping, trainsize=trainsize)
    img_num_per_cls = get_img_num_per_cls(cls_num=dataset.__num_class__(), total_num=dataset.__len__(), imb_type='exp', imb_factor=imb_factor)
    new_imgList, new_labelList = gen_imbalanced_data(img_num_per_cls, imgList=dataset.imgs, labelList=dataset.types)
    dataset.imgs = new_imgList
    dataset.types = new_labelList
    assert len(dataset.imgs) == len(dataset.types)
    print(len(dataset.imgs))
    print(len(dataset.types))
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, root, testsize=None):
        self.testsize = testsize
        image_root = os.path.join(root, 'images/')
        gt_root = os.path.join(root, 'masks/')
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.tif') or f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png') or f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        if self.testsize != None:
            self.transform = transforms.Compose([
                transforms.Resize((self.testsize, self.testsize)),
                transforms.ToTensor()])
            self.gt_transform =  transforms.Compose([
                #transforms.Resize((self.testsize, self.testsize)),
                transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])
            self.gt_transform =  transforms.Compose([transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        gt = self.gt_transform(gt).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
            
class paris_test_dataset:
    def __init__(self, root, dataset_name, type_file='types.txt', testsize=None):
        self.image_root = os.path.join(root, 'images')
        self.gt_root = os.path.join(root, 'masks')
        self.type_path = os.path.join(root, type_file)
        self.testsize = testsize

        if dataset_name == 'EndoScene':
            self.type_mapping = {'Is': 0, 'Ip': 1, 'Isp': 2, 'LST': 3}
        elif dataset_name == 'PICCOLO':
            self.type_mapping = {'IIa': 0, 'IIac': 1, 'Ip': 2, 'Is': 3, 'Isp': 4, 'unknown': 5, 'IIb': 6}
        else:
            raise Exception("Invalid dataset name!")

        with open(self.type_path, 'r') as f:
            #label_file的格式， （label_file image_label)
            self.names = list(map(lambda line: line.strip().split(' '), f))

        if self.testsize != None:
            self.transform = transforms.Compose([
                transforms.Resize((self.testsize, self.testsize)),
                transforms.ToTensor()])
            self.gt_transform =  transforms.Compose([
                #transforms.Resize((self.testsize, self.testsize)),
                transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])
            self.gt_transform =  transforms.Compose([transforms.ToTensor()])

        self.index = 0

    def load_data(self):
        name, type = self.names[self.index]
        type = self.type_mapping[type]
        type = torch.tensor([type])
        image_path = os.path.join(self.image_root, name)
        gt_path = os.path.join(self.gt_root, name)

        image = self.rgb_loader(image_path)
        gt = self.binary_loader(gt_path)
        image = self.transform(image).unsqueeze(0)
        gt = self.gt_transform(gt).unsqueeze(0)
        data = {'image': image, 'label': gt}
        self.index += 1
        return data, type, name.split('/')[-1]

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def get_num_class(self):
        return len(self.type_mapping)




