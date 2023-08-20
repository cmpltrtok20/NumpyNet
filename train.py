import os
import sys
import numpy as np
from tvts.tvts import Tvts
from PyCmpltrtok.common_np import onehot_by_lut


class NumpyNetTrain(object):

    @staticmethod
    def get_save_path(self, xepoch):
        return os.path.join(self.save_dir, self.ts.get_save_prefix(xepoch)) + '.npy'

    def __init__(
        self,
        name,
        model, batch_size,
        tvts_host=None, tvts_port=None, tvts_db=None, tvts_table_prefix=None, tvts_init_params=None,
        epoch_ckpt_map=None, save_dir=None, save_freq=1,
        metrics=None,
        ver='v1.0'
    ):
        self.name = name
        self.model = model
        self.batch_size = batch_size
        if epoch_ckpt_map is None:
            epoch_ckpt_map = self.get_save_path
        self.epoch_ckpt_map = epoch_ckpt_map
        self.ver = ver
        self.save_freq = save_freq
        if save_dir is None:
            save_dir = os.path.split(sys.argv[0])[0]
        self.save_dir = save_dir
        self.n_epoch = 0
        self.epoch = 0
        self.batch = 0

        params = tvts_init_params
        if params is None:
            params = {}
        params['ver'] = ver
        params['batch_size'] = batch_size
        params['lr'] = self.model.lr

        tvts_args = dict(
            tvts_host=tvts_host,
            tvts_port=tvts_port,
            tvts_db=tvts_db,
            tvts_table_prefix=tvts_table_prefix,
        )
        tvts_opt = {}
        n_prefix_chars = len('tvts_')
        for k in tvts_args.keys():
            v = tvts_args[k]
            if v is not None:
                tvts_opt[k[n_prefix_chars:]] = v

        self.ts = Tvts(name, params=params, save_freq=save_freq, save_dir=save_dir, **tvts_opt)

        if model.bin:
            self.std_metrics = set(['acc', 'recall', 'precision', 'f1'])
        else:
            self.std_metrics = set(['acc'])
        self.metrics = self.check_metrics(metrics)

    def check_metrics(self, metrics):
        if metrics is None:
            return self.std_metrics.copy()
        unsupported = metrics.difference(self.std_metrics)
        if len(unsupported):
            raise Exception(f'{unsupported} is not supported!')
        metrics = sorted(list(metrics))
        return metrics

    def resume(self, parent_id=0, parent_epoch=0):
        if not parent_id:
            return None
        ckpt, save_dir = self.ts.resume(parent_id, parent_epoch)
        if self.save_dir is not None:
            save_dir = self.save_dir
        ckpt = os.path.join(save_dir, ckpt)
        if not os.path.exists(ckpt):
            raise Exception(f'Weight saved path {ckpt} does not exist!')
        self.model.load_weights(ckpt)
        return ckpt

    def dlx(self, x_data, batch_size=None):
        """data loader"""
        if batch_size is None:
            batch_size = self.batch_size
        n = len(x_data)
        n_batch = int(np.ceil(n / batch_size))

        for i in range(n_batch):
            yield x_data[i * batch_size:(i + 1) * batch_size]

    def dl(self, x_data, y_data, is_train=False, batch_size=None):
        """data loader"""
        if batch_size is None:
            batch_size = self.batch_size
        n = len(x_data)
        n_batch = int(np.ceil(n / batch_size))

        if is_train:
            rand_idx = np.random.permutation(n)
            x_data = x_data[rand_idx]
            y_data = y_data[rand_idx]

        for i in range(n_batch):
            yield x_data[i * batch_size:(i + 1) * batch_size], y_data[i * batch_size:(i + 1) * batch_size]

    def run_on_data(self,
        label, x_data,
        epoch=0, y_data=None, is_train=False,
        is_return_pred=False, verbose=True,
        batch_size=None,
        metrics=None, thresh=None
    ):
        if batch_size is None:
            batch_size = self.batch_size
        model = self.model
        ts = self.ts
        if metrics is None:
            metrics = self.metrics
        else:
            metrics = self.check_metrics(metrics)
        n = len(x_data)
        n_batch = int(np.ceil(n / batch_size))

        opt_args = {}
        if thresh is not None:
            opt_args['thresh'] = thresh

        cost_avg = 0.
        lr_avg = 0.
        metrics_avg = None
        if is_return_pred:
            pred = np.array([])

        if y_data is None:
            for i, x_batch in enumerate(self.dlx(x_data, batch_size)):
                batch = i + 1
                self.batch = batch
                h = model(x_batch)
                if is_return_pred:
                    pred = np.append(pred, h)
        else:
            for i, (x_batch, y_batch) in enumerate(self.dl(x_data, y_data, is_train, batch_size)):
                batch = i + 1
                self.batch = batch
                h = model(x_batch)
                if is_return_pred:
                    pred = np.append(pred, h)
                cost = model.cost(h, y_batch)
                all_metrics_dict = model.metrics(y_batch, h, **opt_args)
                if verbose:
                    print(
                        label,
                        'epoch:', epoch, 'batch:', batch,
                        'cost:', cost,
                        'metrics:', [f'{k}:{all_metrics_dict[k]}' for k in metrics]
                    )
                if is_train:
                    lr_avg += self.model.lr / n_batch
                    params = {
                        'lr': self.model.lr,
                        'cost': cost,
                    }
                    for k in metrics:
                        params[k] = all_metrics_dict[k]
                    ts.save_batch(epoch, batch, params)

                cost_avg += cost / n_batch
                if metrics_avg is None:
                    metrics_avg = all_metrics_dict
                    for k in metrics_avg.keys():
                        metrics_avg[k] /= n_batch
                else:
                    for k in all_metrics_dict.keys():
                        metrics_avg[k] += all_metrics_dict[k] / n_batch

                if is_train:
                    model.backward(y_batch)
                    model.step()
            if verbose:
                print(
                    label,
                    'epoch:', epoch,
                    'lr_avg:', lr_avg,
                    'cost_avg:', cost_avg,
                    'metrics:', [f'{k}:{metrics_avg[k]}' for k in metrics])

        if is_return_pred:
            res_pred = pred.reshape(-1, *h.shape[1:])
        else:
            res_pred = None
        if not is_train:
            lr_avg = None

        return lr_avg, cost_avg, metrics_avg, res_pred

    def train(
            self,
            n_epoch, x_train, y_train,
            x_val=None, y_val=None,
            verbose=True, batch_size=None,
            metrics=None, thresh=None,
            is_return_pred_train=False, is_return_pred_val=False):
        if batch_size is None:
            batch_size = self.batch_size
        self.n_epoch = n_epoch
        if metrics is None:
            metrics = self.metrics
        else:
            metrics = self.check_metrics(metrics)

        if not self.model.bin:
            y_train = onehot_by_lut(y_train, self.model.n_cls)
            if x_val is not None:
                if y_val is None:
                    raise Exception(f'y_val must be not None when you specified the x_val!')
                y_val = onehot_by_lut(y_val, self.model.n_cls)

        if is_return_pred_train:
            pred_train = np.array([])
        if is_return_pred_val:
            pred_val = np.array([])

        lr_his, cost_his, cost_his_val = [], [], []
        metrics_his, metrics_his_val = {}, {}
        for k in metrics:
            metrics_his[k], metrics_his_val[f'{k}_val'] = [], []

        for i in range(self.n_epoch):
            epoch = i + 1
            self.epoch = epoch

            lr_avg, cost_avg, metrics_avg, pred = self.run_on_data(
                'train', x_train,
                epoch, y_data=y_train, is_train=True,
                is_return_pred=is_return_pred_train, verbose=verbose,
                batch_size=batch_size,
                metrics=metrics, thresh=thresh,
            )
            lr_his.append(lr_avg)
            cost_his.append(cost_avg)
            for k in metrics:
                metrics_his[k].append(metrics_avg[k])

            if is_return_pred_train:
                pred_train = np.append(pred_train, pred)

            params = {
                'lr': self.model.lr,
                'cost': cost_avg,
            }
            for k in self.metrics:
                params[k] = metrics_avg[k]

            print(self.name, self.epoch, params)

            if x_val is not None:
                _, cost_avg_val, metrics_avg_val, pred = self.run_on_data(
                    'val', x_val,
                    epoch, y_data=y_val, is_train=False,
                    is_return_pred=is_return_pred_val, verbose=verbose,
                    batch_size=batch_size,
                    metrics=metrics, thresh=thresh,
                )

                cost_his_val.append(cost_avg_val)
                for k in metrics:
                    metrics_his_val[f'{k}_val'].append(metrics_avg_val[k])

                if is_return_pred_val:
                    pred_val = np.append(pred_val, pred)

                params['cost_val'] = cost_avg_val
                for k in self.metrics:
                    params[f'{k}_val'] = metrics_avg_val[k]

                print(self.name, self.epoch, params)

            save_rel_path = None
            if epoch % self.save_freq == 0:
                save_path = self.epoch_ckpt_map(self, epoch)
                self.model.save_weights(save_path)
                save_rel_path = os.path.relpath(save_path, self.save_dir)

            self.ts.save_epoch(epoch, params, save_rel_path=save_rel_path)

        if is_return_pred_train:
            res_pred_train = pred_train.reshape(*y_train.shape)
        else:
            res_pred_train = None
        if x_val is not None:
            if is_return_pred_val:
                res_pred_val = pred_val.reshape(*y_val.shape)
            else:
                res_pred_val = None
        else:
            res_pred_val = None

        return lr_his, cost_his, cost_his_val, metrics_his, metrics_his_val, res_pred_train, res_pred_val

    def test(self,
             x_test, y_test,
             verbose=True, batch_size=None,
             metrics=None, thresh=None,
             is_return_pred=False):
        if batch_size is None:
            batch_size = self.batch_size
        if metrics is None:
            metrics = self.metrics
        else:
            metrics = self.check_metrics(metrics)

        if not self.model.bin:
            y_test = onehot_by_lut(y_test, self.model.n_cls)

        _, cost_avg, metrics_avg, pred = self.run_on_data(
            'test', x_test,
            y_data=y_test, is_train=False,
            is_return_pred=is_return_pred, verbose=verbose,
            batch_size=batch_size,
            metrics=metrics, thresh=thresh,
        )
        print('Tested', 'cost', cost_avg, [f'{k}:{metrics_avg[k]}' for k in metrics])
        return cost_avg, metrics_avg, pred

    def predict(self, x, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        pred = self.run_on_data(
            'pred', x,
            is_train=False,
            is_return_pred=True, verbose=False,
            batch_size=batch_size,
        )[3]
        return pred
