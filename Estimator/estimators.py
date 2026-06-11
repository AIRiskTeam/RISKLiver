import os
import numpy as np
import pandas as pd
from typing import Literal

from tabpfn import load_fitted_tabpfn_model

class BaseEstimator():
    def __init__(
            self, 
            working_dir: str,
            date: str,
            device: str,
            model: Literal['xgb','tabpfn'],
            model_type: Literal['classifier', 'regressor'],
            run_num: int,
            features: list,
            cat_dict: dict = {},
            cat_keys: list[str] = [],
    ):
        """
        Ensemble model

        :param working_dir: working dir
        :param device: refer to `torch.device`
        :param model: tabpfn/xgb
        :param model_type: classifier/regressor
        :param features: features
        :param date: `metadata['date']`
        :param run_num: `metadata['run_num']`
        :param cat_dict: `metadata['cat_dict']`
        :param cat_keys: `metadata['cat_keys']`
        """
        self.device = device
        self.model_type = model_type
        self.model = model
        self.run_num = run_num
        self.cat_dict = cat_dict
        self.cat_keys = cat_keys
        self.features = features
        self.date = date
        self.working_dir = working_dir

        # Load models
        self.model_list = []
        for idx in range(self.run_num):
            if self.model == "xgb":
                # XGBoost
                path = self.join("model", f"{self.date}_{idx}.xgb_fit")
                self.model_list.append(path)
            elif self.model == "tabpfn":
                # TabPFN
                path = self.join("model", f"{self.date}_{idx}.tabpfn_fit")
                self.model_list.append(path)
            else:
                raise ValueError("unknown model")

    def join(self, *path) -> str:
        return os.path.join(self.working_dir, *path)

    def _load_model(self, model_path:str):
        'load models'
        if self.model == "tabpfn":
            # TabPFN
            model = load_fitted_tabpfn_model(model_path, device=self.device)
        elif self.model == "xgb":
            pass

        return model


class ViabilityEstimator(BaseEstimator):
    'for Viability'

    def predict_proba(self, X:pd.DataFrame):
        'predict(for classification only)'
        assert self.model_type == 'classifier'
        input_X = X[self.features]

        # 3-d array(models*samples*categories)
        shape = (self.run_num, input_X.shape[0], len(self.cat_keys))
        proba_array = np.zeros(shape, dtype=np.float64)

        # get outputs for each model
        for idx, model_path in enumerate(self.model_list):
            model = self._load_model(model_path)
            proba = model.predict_proba(input_X)
            proba_array[idx] = np.asarray(proba)

        # merge
        proba_mean = np.mean(proba_array, axis=0)
        return proba_mean

    def predict(self, X:pd.DataFrame):
        'predict'
        input_X = X[self.features]

        # 2-d array(models*samples*categories)
        shape = (self.run_num, input_X.shape[0])
        pred_array = np.zeros(shape, dtype=np.float64)

        # get outputs for each model
        for idx, model_path in enumerate(self.model_list):
            model = self._load_model(model_path)
            pred = model.predict(input_X)
            pred_array[idx] = pred

        # merge
        pred_all_models = np.vstack(pred_array)
        pred_mean = np.mean(pred_all_models, axis=0)

        if self.model_type == "classifier":
            return np.round(pred_mean).astype(np.int64)
        else:
            return pred_mean
        
    def predict_auto(self, X:pd.DataFrame):
        'auto choose methods to predict'
        def num_to_label(X):
            cat_keys_array = np.asarray(self.cat_keys)
            index_array = np.astype(X, np.int64)
            return cat_keys_array[index_array]

        if self.model_type == "classifier":
            # select the best probabiliy based on results
            probas = self.predict_proba(X)
            pred = np.argmax(probas, axis=1)
            pred = num_to_label(pred)
        else:
            # directly return pred
            pred = self.predict(X)
        
        return pred



class PotencyEstimator(BaseEstimator):
    'for potency (DEPRECATED)'
    def predict_proba(self, X:pd.DataFrame):
        'predict(for classification)'
        assert self.model_type == 'classifier'
        assert len(self.features) == self.run_num

        # 3-d array(models*samples*categories)
        shape = (self.run_num, X.shape[0], len(self.cat_keys))
        proba_array = np.zeros(shape, dtype=np.float64)

        for idx, model_path in enumerate(self.model_list):
            # for potency, every model uses different features
            features = self.features[idx]
            input_X = X[features]

            # load model and predict
            model = self._load_model(model_path)
            proba = model.predict_proba(input_X)
            proba_array[idx] = np.asarray(proba)

        # merge
        proba_mean = np.mean(proba_array, axis=0)
        return proba_mean
