from Descriptors.mordred import get_rdkit_desc
from Estimator.estimators import ViabilityEstimator
import pandas as pd
import numpy as np
import json, torch

def calculate_descriptors(input_str:str):
    '使用传入的str计算描述符'
    # 拆分smiles
    smiles_list = input_str.splitlines()

    # Core RDKit descriptors
    rdkit_desc = get_rdkit_desc(smiles_list)

    df_list = [
        pd.Series(smiles_list, name="smiles"),
        rdkit_desc,
    ]

    final_data_df = pd.concat(df_list, axis=1)
    return final_data_df

def _predict(final_data_df:pd.DataFrame, endpoints:list[str]):
    '预测入口'
    # 设置设备
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 载入config
    with open("config.json", "r") as f:
        config = json.load(f)

    # 建立estimator_list
    estimator_list = [
        ViabilityEstimator(device=DEVICE, **config[name]) for name in endpoints
    ]

    # input data
    input_data = final_data_df.copy()
    shape = (len(estimator_list), input_data.shape[0])
    result_array = np.zeros(shape, dtype=np.float64)

    # 预测
    for idx in range(len(estimator_list)):
        estimator = estimator_list[idx]
        result_array[idx] = estimator.predict_auto(input_data)
    
    # 整理结果
    result_df = pd.DataFrame(result_array.T, columns=[i.date for i in estimator_list])
    ## 重命名
    model_mappings = {config[name]['date']:name for name in config.keys()}
    result_df = result_df.rename(columns=model_mappings)
    ## 合并SMILES
    smiles_series = input_data['smiles']
    result_df = pd.concat(
        [smiles_series, result_df],
        axis=1,
    )
    return result_df

def hazard_identification(final_data_df:pd.DataFrame):
    '分类模型'
    result_df = _predict(
        final_data_df=final_data_df,
        endpoints=[
            'Hazard_viability',
            'Hazard_mitochondrial',
            'Hazard_apoptosis',
        ]
    )
    return result_df

def _call_opera(smiles_array:np.ndarray):
    '调用opera预测IC20'

def regression_modeling(final_data_df:pd.DataFrame, hi_result:pd.DataFrame):
    '回归模型'
    result_df = final_data_df[['smiles']]
    for endpoint in ['viability', 'mitochondrial', 'apoptosis']:
        # 建立遮罩
        active_mask = (hi_result[f'Hazard_{endpoint}'] == 'active').to_numpy()
        active_data = final_data_df.loc[active_mask].reset_index(drop=True)

        # active的部分用回归模型预测
        active_result = _predict(
            final_data_df=active_data,
            endpoints=[
                f'IC20_free_{endpoint}',
                f'IC20_cell_{endpoint}',
            ]
        )

        # inactive的部分用opera
        # TODO: OPERA
    # 通过hi_result筛选
    return result_df