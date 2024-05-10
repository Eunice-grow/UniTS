from metrics.combine_all_scores import main
from metrics.combine_all_scores import combine_all_evaluation_scores

def test():
    import pandas as pd
    df = pd.read_csv("output.csv")
    a, b = df["ground truth"], df["predict value"]
    y_test = [ 0.0 if sum(a[i:i+96])/96.0 <0.5 else 1.0 for i in range(0, len(a), 96)]
    pred_score = [sum(b[i:i+96])/96.0 for i in range(0, len(b), 96)]
    pred_labels = [0.0 if i<0.5 else 1.0 for i in range(len(pred_score))] 
    print("len(y_test) = ",len(y_test))
    print("len(pred_labels) = ",len(pred_labels))
    df1 = pd.DataFrame(list(zip(y_test, pred_labels,pred_score)), columns=['ground truth', 'predict label','predict score'])
    df1.to_csv('output_change.csv', index=False)

def print_ucr_yaml():
    import os
    root_path  = os.path.join(
                ".",
                "dataset",
                "ucr",
                "UCR_TimeSeriesAnomalyDatasets2021",
                "FilesAreInHere",
                "UCR_Anomaly_FullData"
            )
    fname_list = os.listdir(root_path)
    with open('./data_provider/ad_exp_ucr.yaml', 'w') as f:
        f.write('task_dataset:\n')
        for idx,fname in enumerate(fname_list):
            dataset = '  UCR_' + "{}".format(idx)
            str = dataset+':\n' \
            '    task_name: anomaly_detection\n' \
            '    dataset_name: UCR\n' \
            '    dataset: UCR\n'\
            '    data: UCR\n' \
            '    root_path: ./dataset/ucr/UCR_TimeSeriesAnomalyDatasets2021/FilesAreInHere/UCR_Anomaly_FullData\n' +\
            '    file_name: '+fname + '\n' \
            '    seq_len: 96\n' \
            '    label_len: 0\n' \
            '    pred_len: 0\n' \
            '    features: M\n' \
            '    embed: timeF\n' \
            '    enc_in: 1\n' \
            '    dec_in: 1\n' \
            '    c_out: 1\n\n' \

            f.write(str)
    print(fname_list)
    print(len(fname_list))

def test_ocsvm():
    import pandas as pd
    import time
    import numpy as np
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler
    origin_train = pd.read_csv('./dataset/SWaT/SWaT_train.csv',index_col=[0])
    origin_test = pd.read_csv('./dataset/SWaT/SWaT_test.csv',index_col=[0])
    train_data = origin_train.values[:, :-1]
    train_labels = origin_train['label']
    test_data = origin_test.values[:,:-1]
    test_labels = origin_test['label']
    print('train_data',train_data.shape)
    print('test_data',test_data.shape)


    scaler = StandardScaler()
    np_scaled = scaler.fit_transform(train_data)
    data = pd.DataFrame(np_scaled)
    # train oneclassSVM 
    model = OneClassSVM(nu=0.01, kernel="rbf", gamma=0.01)
    print("---------------------start training-------------------")
    start_time = time.time()
    model.fit(data)
    print('the training costs {}s'.format(time.time() - start_time))
    print("---------------------end training-------------------")

    predict = model.predict(test_data)
    predict_test_labels = np.where(predict == 1, 0, 1) # 正常 0 异常1
    
    combine_all_evaluation_scores(test_labels , predict_test_labels, 0)



if __name__ == '__main__':
    # main()
    # test()
    # print_ucr_yaml()
    test_ocsvm()