import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import os
import time
import torch.nn.functional as F
import numpy as np
import sys
from torch.utils.data import Dataset, DataLoader
import pickle
import plda
# from sklearn.metrics import f1_score,precision_score,recall_score




def readpkl(path):
    with open(path,'rb') as f:
        data,label = pickle.load(f)
        return data,label

class getdata(Dataset):
    # dirname 为训练/测试数据地址，使得训练/测试分开
    def __init__(self, dirname, train=True):
        super(getdata, self).__init__()
        self.images, self.labels = readpkl(dirname)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        image = image.astype(np.float32)
        # image = torch.from_numpy(image).div(255.0)
        label = self.labels[index]
        label = int(label)
        return image, label

def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(DEVICE)

    BATCH_SIZE = 16

    scenario = 'D3D1'
    train_path1 = scenario +'_512_train.pkl'
    val_path = scenario + '_512_test.pkl'

    with open(train_path1,'rb') as f1:
        train_data, train_label = pickle.load(f1)
    # print(train_data[0])
    # os._exit(0)
    # with open(train_path2,'rb') as f2:
    #     train_data_2, train_label_2 = pickle.load(f2)
    with open(val_path, 'rb') as f3:
        test_data, test_label = pickle.load(f3)

    # print(train_data_1.shape)
    # print(train_data_2.shape)
    # train_data = np.append(train_data_1,train_data_2,axis=0)
    # train_data = train_data.reshape(train_data.shape[0], -1)  # flatten

    # train_label = np.append(train_label_1,train_label_2,axis=0)

    # print(train_data.shape)
    # print(train_label.shape)

    # os._exit(0)
    since = time.time()

    train_X, train_y = train_data, train_label
    test_X, test_y = test_data, test_label

    print('Start Training')
    overfit_classifier = plda.Classifier()
    overfit_classifier.fit_model(train_X, train_y)  # 训练plda

    plda_file = scenario + '.txt'
    f = open(plda_file, 'wb')
    pickle.dump(overfit_classifier, f)
    f.close()
    g = open(plda_file, 'rb')
    bb = pickle.load(g)
    g.close()
    predictions, log_p_predictions = bb.predict(test_X)  # 预测，使用plda
    # print(predictions)
    # print(log_p_predictions)
    # predictions, log_p_predictions = overfit_classifier.predict(test_X)
    acc = (test_y == predictions).mean()
    print('Accuracy: {:.4f}'.format(acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # f1score = f1_score(test_y,predictions,average='macro')
    # print('f1score:{:.4f}'.format(f1score))
    # acc_str = 'Acc: ' + str(acc)
    # f1_str = 'F1: ' + str(f1score)
    # with open(file_name + '_result.txt', 'w') as f:
    #     f.write(acc_str + '\n')
    #     f.write(f1_str)


def test_plda(net,  testloader, device,path):
    net.eval()
    testdata = np.array([[]])
    testlabel = np.array([])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            _,outputs = net(inputs)
            for emb in outputs:
                emb = torch.unsqueeze(emb,0)
                l2_norms = np.linalg.norm(emb.cpu(), axis=1, keepdims=True)
                emb = emb.cpu() / l2_norms
                try:
                    testdata = np.append(testdata, emb, axis=0)
                except:
                    testdata = emb
            for label in targets:
                testlabel = np.append(testlabel, label.cpu())

    print('testdata:',testdata.shape)
    print('testlabel',testlabel.shape)
    # for key, value in testdata
    # s1:1457
    # s2:2668
    # known_num = 1946
    # test_X, test_y = testdata[0:known_num], testlabel[0:known_num]
    test_X, test_y = testdata, testlabel

    plda_file = path +'.txt'
    g = open(plda_file, 'rb')
    # g = open('crossS1_BN.txt', 'rb')
    bb = pickle.load(g)
    g.close()
    predictions, log_p_predictions = bb.predict(test_X)  # 预测，使用plda

    # print(predictions)
    # print(log_p_predictions)
    # predictions, log_p_predictions = overfit_classifier.predict(test_X)
    acc = (test_y == predictions).mean()
    print('Accuracy: {:.4f}'.format(acc))

    macro_f1score = f1_score(test_y, predictions, average='macro')
    # w_f1score = f1_score(test_y, predictions, average='weighted')
    print('macro-f1score:{:.4f}'.format(macro_f1score))

    # print('w-f1score:{:.4f}'.format(w_f1score))
    prec = precision_score(test_y, predictions, average='macro')
    print('精确率:{:.4f}'.format(prec))

    recall = recall_score(test_y, predictions, average='macro')
    print('召回:{:.4f}'.format(recall))

def test_tdnn(net,testloader,device):
    net.eval()

    testdata = np.array([])
    testlabel = np.array([])
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs,_ = net(inputs)
            # print(outputs)
            for out in outputs:
                # print(out)
                pre = np.argmax(out.cpu())
                # print(pre)
                testdata = np.append(testdata, pre.cpu())
            for label in targets:
                testlabel = np.append(testlabel, label.cpu())

    print('testdata:',testdata.shape)
    print('testlabel',testlabel.shape)
    # print(testdata)
    # print(testlabel)
    # s1:1457
    # s2:2668
    # print('jiance')
    # known_num = 2342
    # test_X, test_y = testdata[0:known_num], testlabel[0:known_num]
    test_X, test_y = testdata, testlabel

    print('Accuracy: {}'.format((test_X == test_y).mean()))
    f1score = f1_score(test_X, test_y, average='macro')
    print('f1score=', f1score)

if __name__ == '__main__':
    main()
