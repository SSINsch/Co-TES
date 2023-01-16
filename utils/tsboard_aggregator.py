import os
import pandas as pd
import logging
import json
import matplotlib
import matplotlib.pyplot as plt
from typing import Dict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from utils import get_subdirs

logger = logging.getLogger(__name__)


class TensorBoardPrinter:
    def __init__(self, root_dir: str, output_path: str = None):
        self.root_dir = root_dir
        self.output_path = output_path

    @staticmethod
    def run(event_item_path: str) -> Dict[str, pd.DataFrame]:
        runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
        try:
            event_acm = EventAccumulator(event_item_path)
            event_acm.Reload()
            tags_scalar = event_acm.Tags()["scalars"]
            for tag in tags_scalar:
                event_list = event_acm.Scalars(tag)
                values = list(map(lambda x: x.value, event_list))
                step = list(map(lambda x: x.step, event_list))
                r = {"metric": [tag] * len(step), "value": values, "step": step}
                r = pd.DataFrame(r)
                runlog_data = pd.concat([runlog_data, r])

        except Exception as e:
            m = f'Event file possibly corrupt: {event_item_path}'
            logger.error(m)
            logger.error(e)

        return runlog_data

    def recursive_run(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        # root_dir = 실험 모아둔 폴더
        ex_folders = get_subdirs(self.root_dir)

        dict_ex_result = {}
        for ex in ex_folders:
            # load run parameters
            f_json = os.path.join(ex, 'info.json')
            with open(f_json) as f:
                run_params = json.load(f)

            opt1 = '_'.join([run_params['model1'], 'opt1'])
            opt2 = '_'.join([run_params['model2'], 'opt2'])

            if 'cnn' in opt1:
                run_params[opt1] = list(map(str, run_params[opt1]))
                run_params[opt1] = ''.join(run_params[opt1])
            if 'cnn' in opt2:
                run_params[opt2] = list(map(str, run_params[opt2]))
                run_params[opt2] = ''.join(run_params[opt2])

            # dictionary key name 생성
            name = [run_params['seed'], run_params['noise_type'], run_params['noise_rate'], run_params['model1'],
                    run_params[opt1], run_params['model2'], run_params[opt2]]
            name = '_'.join(map(str, name))

            # metric load, get df
            metrics = get_subdirs(ex)
            dict_item_result = {}
            for mt in metrics:
                if 'train_loss' in mt:
                    df_key = 'train_loss'
                elif 'val_loss' in mt:
                    df_key = 'val_loss'
                elif 'train_acc' in mt:
                    df_key = 'train_acc'
                elif 'test_acc' in mt:
                    df_key = 'test_acc'
                else:
                    logger.error(f'Metric not specified!')
                    raise KeyError(f'Metric not specified!')
                df = self.run(event_item_path=mt)
                dict_item_result[df_key] = df

            dict_ex_result[name] = dict_item_result

        return dict_ex_result

    def save_fig_acc(self,
                     target: Dict[str, Dict[str, pd.DataFrame]],
                     filter: Dict = None,
                     output_path: str = None) -> None:
        raise NotImplementedError


if __name__ == '__main__':
    p = '../logs/news/coteaching_plus/test'
    tb_printer = TensorBoardPrinter(root_dir=p)
    res = tb_printer.recursive_run()
    print('Done')
