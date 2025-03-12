import json
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import methods.util as util
import argparse
import os

UNKNOWN_LABEL = -1

my_mean = [0.5,0.5,0.5]
my_std = [0.25,0.25,0.25]

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

cifar_mean = (0.5,0.5,0.5)
cifar_std = (0.25,0.25,0.25)

tiny_mean = (0.5,0.5,0.5)
tiny_std = (0.25,0.25,0.25)

svhn_mean = (0.5,0.5,0.5)
svhn_std = (0.25,0.25,0.25)

workers = 6
test_workers = 6
use_droplast = True
require_org_image = True
no_test_transform = False

DATA_PATH = '/HOME/scz1838/run/data'
TINYIMAGENET_PATH = DATA_PATH + '/tiny-imagenet-200/'
LARGE_OOD_PATH = '/HOME/scz1838/run/largeoodds'
IMAGENET_PATH = '/data/public/imagenet2012'

class RandomNoise(object):
    def __init__(self,mean=0, std=1):
        self.mean = mean
        self.std = std
    def __call__(self, image):
        noise =torch.randn_like(image)*self.std +self.mean
        noisy_image = image +noise
        return noisy_image

class tinyimagenet_data(Dataset):

    def __init__(self, _type, transform):
        if _type == 'train':
            self.ds = datasets.ImageFolder(f'{TINYIMAGENET_PATH}/train/', transform=transform)
            self.labels = [self.ds.samples[i][1] for i in range(len(self.ds))]
        elif _type == 'test':
            tmp_ds = datasets.ImageFolder(f'{TINYIMAGENET_PATH}/train/', transform=transform)
            cls2idx = tmp_ds.class_to_idx
            self.ds = datasets.ImageFolder(f'{TINYIMAGENET_PATH}/val/', transform=transform)
            with open(f'{TINYIMAGENET_PATH}/val/val_annotations.txt','r') as f:
                file2cls = {}
                for line in f.readlines():
                    line = line.strip().split('\t')
                    file2cls[line[0]] = line[1]
            self.labels = []
            for i in range(len(self.ds)):
                filename = self.ds.samples[i][0].split('/')[-1]
                self.labels.append(cls2idx[file2cls[filename]])
            # print("test labels",self.labels)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self,idx):
        return self.ds[idx][0],self.labels[idx]

class Imagenet1000(Dataset):

    lab_cvt = None

    def __init__(self,istrain, transform):

        set = "train" if istrain else "val"
        self.ds = datasets.ImageFolder(f'{IMAGENET_PATH}/{set}/', transform=transform)
        self.labels = [self.ds.samples[i][1] for i in range(len(self.ds))]
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self,idx):
        return self.ds[idx]

class LargeOODDataset(Dataset):

    def __init__(self,ds_name,transform) -> None:
        super().__init__()
        data_path = f'{LARGE_OOD_PATH}/{ds_name}/'
        self.ds = datasets.ImageFolder(data_path, transform=transform)
        self.labels = [-1] * len(self.ds)
    
    def __len__(self,):
        return len(self.ds)
    
    def __getitem__(self, index):
        return self.ds[index]


class PartialDataset(Dataset):

    def __init__(self,knwon_ds,lab_keep = None,lab_cvt = None) -> None:
        super().__init__()
        self.known_ds = knwon_ds
        labels = knwon_ds.labels
        if lab_cvt is None:  # by default, identity mapping
            lab_cvt = [i for i in range(1999)]
        if lab_keep is None:  # by default, keep positive labels
            lab_keep = [x for x in lab_cvt if x > -1]
        keep = {x for x in lab_keep}
        self.sample_indexes = [i for i in range(len(knwon_ds)) if lab_cvt[labels[i]] in keep]
        self.labels = [lab_cvt[labels[i]] for i in range(len(knwon_ds)) if lab_cvt[labels[i]] in keep]
        self.labrefl = lab_cvt

    def __len__(self) -> int:
        return len(self.sample_indexes) 

    def __getitem__(self, index: int):
        inp,lb = self.known_ds[self.sample_indexes[index]]
        return inp,self.labrefl[lb],index

class UnionDataset(Dataset):

    def __init__(self,ds_list) -> None:
        super().__init__()
        self.dslist = ds_list
        self.totallen = sum([len(ds) for ds in ds_list])
        self.labels = []
        for x in ds_list:
            self.labels += x.labels
    
    def __len__(self) -> int:
        return self.totallen
    
    def __getitem__(self, index: int):
        orgindex = index
        for ds in self.dslist:
            if index < len(ds):
                a,b,c = ds[index]
                return a,b,orgindex
            index -= len(ds)
        return None


def gen_transform(mean,std,crop = False,toPIL = False,imgsize = 32,testmode = False):
    t = []
    if toPIL:
        t.append(transforms.ToPILImage())
    if not testmode:
        return transforms.Compose(t)
    if crop:
        if imgsize > 200:
            t += [transforms.Resize(256),transforms.CenterCrop(imgsize)]
        else:
            t.append(transforms.CenterCrop(imgsize))
    # print(t)
    return transforms.Compose(t + [transforms.ToTensor(), transforms.Normalize(mean, std)])


def gen_cifar_transform(crop = False, toPIL = False,testmode = False):
    return gen_transform(cifar_mean,cifar_std,crop,toPIL=toPIL,imgsize=32,testmode = testmode)

def gen_tinyimagenet_transform(crop = False,testmode = False):
    return gen_transform(tiny_mean,tiny_std,crop,False,imgsize=64,testmode = testmode)

def gen_imagenet_transform(crop = False, testmode = False):
    return gen_transform(imagenet_mean,imagenet_std,crop,False,imgsize=224,testmode = testmode)

def gen_svhn_transform(crop = False,toPIL = False,testmode = False):
    return gen_transform(svhn_mean,svhn_std,crop,toPIL=toPIL,imgsize=32,testmode = testmode)

def my_transform(crop = False, testmode = False):
    return gen_transform(my_mean,my_std,crop,False,imgsize=224,testmode = testmode)

def get_cifar10(settype):
    if settype == 'train':
        trans = gen_cifar_transform()
        ds = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=trans)
    else:
        ds = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=gen_cifar_transform(testmode=True))
    ds.labels = ds.targets
    # print(dir(ds))
    print(ds)

    return ds

def get_cifar10_new(settype):
    transform_new1 = transforms.Compose([
        # transforms.Resize(224),
        # transforms.RandomVerticalFlip(p=1),   ##竖直翻转
        # transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    transform_new2 = transforms.Compose([
        # transforms.Resize(224),
        # transforms.RandomVerticalFlip(p=1),   ##竖直翻转
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if settype == 'train':
        # ds = datasets.ImageFolder('D:\\work\\data\\public\\image\\S1_256\\train', transform_new1)  ##改动
        ds = datasets.ImageFolder('D:\\work\\data\\public\\3img\\S2\\S2train')  ##改动
    else:
        # ds = datasets.ImageFolder('D:\\work\\data\\public\\image\\S1_256\\val', transform_new2)  ##改动
        ds = datasets.ImageFolder('D:\\work\\data\\public\\3img\\S2\\S2test', transform_new2)  ##改动

    ds.labels = ds.targets
    # print(ds)
    return ds

def get_cifar10_cross(settype):
    transform_new1 = transforms.Compose([
        # transforms.Resize(224),
        # transforms.RandomVerticalFlip(p=1),   ##竖直翻转
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        # RandomNoise(mean=0, std=0.5),
        transforms.ToPILImage()
    ])

    transform_new2 = transforms.Compose([
        # transforms.Resize(224),
        # transforms.RandomVerticalFlip(p=1),   ##竖直翻转
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if settype == 'train':
        ds = datasets.ImageFolder('D:\\work\\code\\2024\\3img\\3\\D3train', my_transform())  ##改动
    else:
        ds = datasets.ImageFolder('D:\\work\\code\\2024\\3img\\1\\D1test', my_transform(testmode=True))  ##改动测试集

    ds.labels = ds.targets
    return ds

def get_cifar100(settype):
    if settype == 'train':
        trans = gen_cifar_transform()
        ds = torchvision.datasets.CIFAR100(root=DATA_PATH, train=True, download=True, transform=trans)
    else:
        ds =  torchvision.datasets.CIFAR100(root=DATA_PATH, train=False, download=True, transform=gen_cifar_transform(testmode=True))
    ds.labels = ds.targets
    return ds

def get_svhn(settype):
    if settype == 'train':
        trans = gen_svhn_transform()
        ds = torchvision.datasets.SVHN(root=DATA_PATH, split='train', download=True, transform=trans)
    else :
        ds = torchvision.datasets.SVHN(root=DATA_PATH, split='test', download=True, transform=gen_svhn_transform(testmode=True))
    return ds

def get_tinyimagenet(settype):
    if settype == 'train':
        trans = gen_tinyimagenet_transform()
        ds = tinyimagenet_data('train',trans)
    else:
        ds = tinyimagenet_data('test',gen_tinyimagenet_transform(testmode=True))
    return ds

def get_imagenet1000(settype):
    if settype == 'train':
        trans = gen_imagenet_transform()
        ds = Imagenet1000(True,trans)
    else:
        ds = Imagenet1000(False,gen_imagenet_transform(crop = True, testmode=True))
    return ds

def get_ood_inaturalist(settype):
    if settype == 'train':
        raise Exception("OOD iNaturalist cannot be used as train set.")
    else:
        return LargeOODDataset('iNaturalist',gen_imagenet_transform(crop = True, testmode=True))

ds_dict = {
    "cifarova" : get_cifar10,
    "cifar10" : get_cifar10,
    "cifar10_new" : get_cifar10_new,
    "cifar10_cross" : get_cifar10_cross,
    "cifar100" : get_cifar100,
    "svhn" : get_svhn,
    "tinyimagenet" : get_tinyimagenet,
    "imagenet" : get_imagenet1000,
    'oodinaturalist' : get_ood_inaturalist,
}

cache_base_ds = {

}

def get_ds_with_name(settype,ds_name):
    global cache_base_ds
    key = str(settype) + ds_name      ##train+cifar10
    if key not in cache_base_ds.keys():
        cache_base_ds[key] = ds_dict[ds_name](settype)
    # print(cache_base_ds[key])
    return cache_base_ds[key]

def get_partialds_with_name(settype,ds_name,label_cvt,label_keep):
    ds = get_ds_with_name(settype,ds_name)
    # print(ds)   ##打印print
    return PartialDataset(ds,label_keep,label_cvt)
    
# setting list [[ds_name, sample partition list, label convertion list],...]
def get_combined_dataset(settype,setting_list):
    ds_list = []
    for setting in setting_list:
        ds = get_partialds_with_name(settype,setting['dataset'],setting['convert_class'],setting['keep_class'])
        if ds.__len__() > 0:
            ds_list.append(ds)
    return UnionDataset(ds_list) if len(ds_list) > 0 else None

def get_combined_dataloaders(args,settings):
    istrain_mode = True
    print("Load with train mode :",istrain_mode)
    train_labeled = get_combined_dataset('train',settings['train'])
    test = get_combined_dataset('test',settings['test'])
    return torch.utils.data.DataLoader(train_labeled, batch_size=args.bs, shuffle=istrain_mode, num_workers=workers,pin_memory=True,drop_last = use_droplast) if train_labeled is not None else None,\
            torch.utils.data.DataLoader(test, batch_size=args.bs, shuffle=False, num_workers=test_workers,pin_memory=args.gpu != 'cpu') if test is not None else None

ds_classnum_dict = {
    'cifar10' : 6,
    'cifar10_new' : 13,
    'cifar10_cross' : 7,
    'svhn' : 6,
    'tinyimagenet' : 20,
    "imagenet" : 1000,
}

imgsize_dict = {
    'cifar10' : 32,
    'cifar10_new' : 224,
    'cifar10_cross' : 224,
    'svhn' : 32,
    'tinyimagenet' : 64,
    "imagenet" : 224,
}

def load_partitioned_dataset(args,ds):
    with open(ds,'r') as f:
        settings = json.load(f)
    util.img_size = imgsize_dict[settings['name']]
    a,b = get_combined_dataloaders(args,settings)
    return a,b,ds_classnum_dict[settings['name']]


def update_config_keyvalues(config, update):
    if update == "":
        return config
    spls = update.split(",")
    for spl in spls:
        key, val = spl.split(':')
        key_parts = key.split('.')
        sconfig = config
        for i in range(len(key_parts) - 1):
            sconfig = sconfig[key_parts[i]]
        org = sconfig[key_parts[-1]]
        if isinstance(org, bool):
            sconfig[key_parts[-1]] = val == 'True'
        elif isinstance(org, int):
            sconfig[key_parts[-1]] = int(val)
        elif isinstance(org, float):
            sconfig[key_parts[-1]] = float(val)
        else:
            sconfig[key_parts[-1]] = val
        print("Updating", key, "with", val, "results in", sconfig[key_parts[-1]])
    return config


def update_subconfig(cfg, u):
    for k in u.keys():
        if not k in cfg.keys() or not isinstance(cfg[k], dict):
            cfg[k] = u[k]
        else:
            update_subconfig(cfg[k], u[k])


def load_config(file):
    with open(file, "r") as f:
        config = json.load(f)
    if 'inherit' in config.keys():
        inheritfile = config['inherit']
        if inheritfile != 'None':
            parent = load_config(inheritfile)
            update_subconfig(parent, config)
            config = parent
    return config

if __name__ == '__main__' :
    # print(len([]))
    for i in range(0):
        print(i)
    os._exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, required=False, default="0", help='GPU number')
    parser.add_argument('--ds', type=str, required=False, default="./exps/cifar10/spl_a.json",help='dataset setting, choose file from ./exps')
    parser.add_argument('--config', type=str, required=False, default="./configs/pcssr/cifar10.json",help='model configuration, choose from ./configs')
    parser.add_argument('--save', type=str, required=False, default="", help='Saving folder name')
    parser.add_argument('--method', type=str, required=False, default="cssr",help='Methods : ' + ",".join(util.method_list.keys()))
    parser.add_argument('--test', action="store_true", help='Evaluation mode')
    parser.add_argument('--configupdate', type=str, required=False, default="",help='Update several key values in config')
    parser.add_argument('--test_interval', type=int, required=False, default=1,help='The frequency of model evaluation')

    args = parser.parse_args()

    if args.config != "None" :
        config = load_config(args.config)
    else:
        config = {}
    config = update_config_keyvalues(config,args.configupdate)
    args.bs = config['batch_size']
    load_partitioned_dataset(args, args.ds)