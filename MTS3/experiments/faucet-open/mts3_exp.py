import sys
sys.path.append('.')
from omegaconf import DictConfig, OmegaConf
import hydra
import os

import torch
import pickle
import gdown

from experiments.exp_prediction_mts3 import Experiment
from hydra.utils import get_original_cwd
from agent.worldModels.MTS3 import MTS3


nn = torch.nn

@hydra.main(config_path='conf',config_name="config")
def my_app(cfg)->OmegaConf:
    global config
    model_cfg = cfg
    exp = Experiment(model_cfg)

    train_obs, train_act, train_targets, test_obs, test_act, test_targets, normalizer = exp._get_data_set()
    if 'diff' in normalizer:
        normalizer['act_diff'] = normalizer['diff']
    
    ### train the model
    mts3_model, wandb_run, save_path = exp._train_world_model(train_obs, train_act, train_targets, test_obs, test_act, test_targets)
    # save_path = get_original_cwd() + '/experiments/saved_models/mts3-faucet-open.ckpt'
    # mts3_model = MTS3(input_shape=[train_obs.shape[-1]], action_dim=train_act.shape[-1], config=model_cfg.model)
    # wandb_run = exp._wandb_init()
    
    ### test the model
    #TODO: normalizer format specify
    exp._test_world_model(test_obs, test_act, test_targets, normalizer, mts3_model, wandb_run, save_path)


class Experiment(Experiment):
    def __init__(self, cfg):
        super(Experiment, self).__init__(cfg)

    def _load_save_train_test_data(self):
        """
        write a function to load the data and return the train and test data
        :return: train_obs, train_act, train_targets, test_obs, test_act, test_targets, normalizer
        """
        ## load the data from pickle and if not present download from the url
        save_path = '/fs/nexus-scratch/anhu/mts3-data-mw-faucet-open.pkl'
        if not os.path.exists(save_path):
            raise Exception(f'{save_path} not found!')
        else:
            print("..........Data Found...........Loading from local")
        with open(save_path, 'rb') as f:
            data_dict = pickle.load(f)
            print("Train Obs Shape", data_dict['train_obs'].shape)
            print("Train Act Shape", data_dict['train_act'].shape)
            print("Train Targets Shape", data_dict['train_targets'].shape)
            print("Test Obs Shape", data_dict['test_obs'].shape)
            print("Test Act Shape", data_dict['test_act'].shape)
            print("Test Targets Shape", data_dict['test_targets'].shape)
            # print("Normalizer", data_dict['normalizer'])
        return data_dict

    def _get_data_set(self):
        """
        write a function to load the data and return the train and test data
        :return: train_obs, train_act, train_targets, test_obs, test_act, test_targets, normalizer
        """
        ### load or generate data
        data_dict = self._load_save_train_test_data()

        return data_dict['train_obs'], data_dict['train_act'], data_dict['train_targets'], data_dict['test_obs'], \
            data_dict['test_act'], data_dict['test_targets'], data_dict['normalizer']


def main():
    my_app()



## https://stackoverflow.com/questions/32761999/how-to-pass-an-entire-list-as-command-line-argument-in-python/32763023
if __name__ == '__main__':
    main()