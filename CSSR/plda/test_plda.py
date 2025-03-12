import os
import numpy as np
import pickle
import plda

def main():
    scenario = 'D3D1'
    sunshi_path1 = scenario +'_DAA.pkl'
    with open(sunshi_path1,'rb') as f1:
        train_data, train_label = pickle.load(f1)


    val_path = scenario +'_512_test.pkl'
    with open(val_path, 'rb') as f3:
        test_data, test_label = pickle.load(f3)

    plda_thr = -10

    print('testdata:',test_data.shape)
    print('testlabel',test_label.shape)

    open_num = len(test_label)
    test_X, test_y = test_data, test_label
    # print(test_y)
    # with open('linshi.txt','w') as f:
    #     for i in test_y:
    #         f.write(str(i)+'\n')
    # os._exit(0)


    g = open(scenario +'.txt', 'rb')

    bb = pickle.load(g)
    g.close()
    predictions, log_p_predictions = bb.predict(test_X)  # 预测，使用plda
    # print(predictions)
    # print(log_p_predictions[0])
    # print('----------')
    log_p_predictions = log_p_predictions + train_data*0.5
    # print(train_data[0]*10)
    # print('----------')
    # print(log_p_predictions[0])


    ###分割线——————————————————————————————————————————————————————————————————————————————————————————
    # out_t=[]
    # for i in range(len(log_p_predictions)):
    #     out = log_p_predictions[i].argmax()  # 取出预测的最大值的索引
    #     out_t.append(out)
    #     i+=1
    # # print(out_t)
    # for i in range(open_num):
    #     score = log_p_predictions[i]
    #     # print(score)
    #     # score2 = plda_w[i]
    #     # score = score/score2
    #     # print('score:',score)
    #     if max(score) < 1.89:  #与acc_known正比
    #         out_t[i] = -1.0  #Subscript of the unknown class
    # known_num = 1946  # num of known samples from S1,1457####S2,2668
    # print('known_Accuracy: {}'.format((test_y[0:known_num] == out_t[0:known_num]).mean()))
    # print('unknown_Accuracy: {}'.format((test_y[known_num:] == out_t[known_num:]).mean()))
    # acc_k = (test_y[0:known_num] == out_t[0:known_num]).mean()
    # acc_u = (test_y[known_num:] == out_t[known_num:]).mean()
    # ACC = (acc_u + acc_k) / 2
    # print('ACC:', ACC)
    # os._exit(0)
    ###分割线——————————————————————————————————————————————————————————————————————————————————————————

    for i in range(open_num):
        score = log_p_predictions[i]
        # score2 = plda_w[i]
        # score = score/score2
        # print('score:',score)
        if max(score) < plda_thr:  #与acc_known正比
            predictions[i] = -1.0  #Subscript of the unknown class

    # print(predictions)
    # with open('linshi2.txt','w') as f:
    #     for i in predictions:
    #         f.write(str(i)+'\n')


    known_num = 436  #num of known samples from S1,1457####S2,2668
    print('known_Accuracy: {}'.format((test_y[0:known_num] == predictions[0:known_num]).mean()))
    print('unknown_Accuracy: {}'.format((test_y[known_num:] == predictions[known_num:]).mean()))
    acc_k = (test_y[0:known_num] == predictions[0:known_num]).mean()
    acc_u = (test_y[known_num:] == predictions[known_num:]).mean()
    ACC = (acc_u+acc_k)/2
    print('ACC:',ACC)


if __name__ == '__main__':
    main()
