if '__main__' == __name__:
    from PyCmpltrtok.data.mnist.load_mnist import load, shape_
    from PyCmpltrtok.common import sep, divide_int
    from PyCmpltrtok.common_np import uint8_to_flt_by_lut
    import matplotlib.pyplot as plt
    import os
    import sys
    import io
    import numpy as np
    from PIL import Image
    from NumpyNet.network import NumpyNet
    from NumpyNet.train import NumpyNetTrain

    sep('Load data')
    x_train, y_train, x_test, y_test = load()
    y_train = (y_train == 5).astype(np.uint8)
    y_test = (y_test == 5).astype(np.uint8)
    print('y_train', np.unique(y_train))
    print('y_test', np.unique(y_test))
    WIDTH = shape_[0] * shape_[1]
    x_train = x_train.reshape(-1, WIDTH)
    x_test = x_test.reshape(-1, WIDTH)
    x_train = uint8_to_flt_by_lut(x_train)
    x_test = uint8_to_flt_by_lut(x_test)
    x_val, y_val = x_test, y_test
    print('x_train', x_train.shape)
    print('y_train', y_train.shape)
    print('x_val', x_val.shape)
    print('y_val', y_val.shape)
    print('x_test', x_test.shape)
    print('y_test', y_test.shape)

    # Hyper params
    sep('Start')
    NAME = 'mnist_bin_clf_by_NumpyNet_MLP_256x2'
    """
    python tvts.py -m 'cost|cost_val,acc|acc_val|recall|recall_val|precision|precision_val|f1|f1_val' --batch_metrics 'cost,acc|recall|precision|f1' -k 'acc_val' --hyper duration_in_sec mnist_bin_clf_by_NumpyNet_MLP_256x2
    """
    IS_TRAIN = 0
    PARENT_ID = 10
    PARENT_EPOCH = 0
    RSEED = 1
    if RSEED is not None:
        np.random.seed(RSEED)
    VER = 'v1.0'
    N_BATCH_SIZE = 512
    LR = 0.1  # learning rate (It should be higher for all sigmoid layers than all relu ones)
    N_EPOCH = 8  # Please to be greater than 650 for a good model; to be 300 for a normal P-R curve
    SAVE_FREQ = 2

    BASE_DIR, FILE_NAME = os.path.split(__file__)
    SAVE_DIR = os.path.join(BASE_DIR, '_save', FILE_NAME, VER)
    os.makedirs(SAVE_DIR, exist_ok=True)

    conf = dict(
        network=dict(
            lr=LR,
            n_inputs=x_test.shape[1],
        ),
        layers=[
            dict(type='conn', n_outputs=256, act='sigmoid'),
            dict(type='conn', n_outputs=256, act='sigmoid'),
            dict(type='conn', n_outputs=1, act='sigmoid'),
        ],
    )
    model = NumpyNet(conf)

    def get_save_path(trainObj, xepoch):
        return os.path.join(SAVE_DIR, trainObj.ts.get_save_name(xepoch)) + '.npy'

    trainObj = NumpyNetTrain(
        NAME,
        model, N_BATCH_SIZE,
        save_freq=SAVE_FREQ, ver=VER, epoch_ckpt_map=get_save_path,
        tvts_host='192.168.31.20',
        save_dir=SAVE_DIR,
    )
    trainObj.resume(PARENT_ID, PARENT_EPOCH)

    if IS_TRAIN:
        sep('Train')
        lr_his, cost_his, cost_his_val, metrics_his, metrics_his_val, *_ = \
            trainObj.train(N_EPOCH, x_train, y_train, x_val, y_val, verbose=1)

    sep('Test')
    trainObj.test(x_test, y_test, verbose=1)

    sep('Calc Demo')
    # Draw testing demo in memory buf
    plt.figure(figsize=[6, 8])
    plt.subplots_adjust(hspace=0.4)
    spr = 8
    spc = 8
    spn = 0
    m = spr * spc
    m1, m2 = divide_int(m)
    idx = y_test.ravel() == 1
    x5 = x_test[idx][:m2]
    y5 = y_test[idx][:m2]
    xn5 = x_test[~idx][:m1]
    yn5 = y_test[~idx][:m1]
    x_demo = np.concatenate([x5, xn5], axis=0)
    y_demo = np.concatenate([y5, yn5], axis=0)
    pred = trainObj.predict(x_demo)
    for i in range(m):
        spn += 1
        plt.subplot(spr, spc, spn)
        y = y_demo[i, 0]
        img = x_demo[i].reshape(*shape_)
        pr = int(pred[i, 0] > 0.5)
        plt.axis('off')
        plt.title(f'{y}=>{pr}', color='black' if pr == y else 'red')
        plt.imshow(img)
    buf = io.BytesIO()
    buf.seek(0)
    plt.savefig(buf, format='png')
    img_demo = Image.open(buf)
    plt.close()

    # Show curve of cost and metrics in training if it exits and show the demo of test
    if not IS_TRAIN:
        plt.figure(figsize=[6, 6])
        plt.axis('off')
        plt.imshow(img_demo)
        buf.close()
        plt.show()
    elif IS_TRAIN:
        plt.figure(figsize=[15, 5])
        plt.axis('off')
        spr = 1
        spc = 3
        spn = 0

        spn += 1
        plt.subplot(spr, spc, spn)
        plt.title(f'Cost')
        plt.plot(cost_his, label='train')
        plt.plot(cost_his_val, label='val')
        plt.legend()
        plt.grid()

        spn += 1
        plt.subplot(spr, spc, spn)
        plt.title('Metrics')
        print(metrics_his)
        print(metrics_his_val)
        xdict = {**metrics_his, **metrics_his_val}
        print(xdict)
        for k in sorted(xdict.keys()):
            plt.plot(xdict[k], label=k)

        plt.legend()
        plt.grid()

        spn += 1
        plt.subplot(spr, spc, spn)
        plt.axis('off')
        plt.imshow(img_demo)
        buf.close()

        plt.show()
