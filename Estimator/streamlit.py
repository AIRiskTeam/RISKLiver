from Descriptors.mordred import get_rdkit_desc

from .estimators import ViabilityEstimator
from .opera import opera_predict
from .ivd import calculate_ivd

import pandas as pd
import numpy as np
import json, os

def calculate_descriptors(smiles_list:list[str], **kwargs) -> pd.DataFrame:
    '使用传入的str计算描述符 (kwargs是为了和其他方法兼容)'
    # 拆分smiles
    smiles_series = pd.Series(smiles_list, name="smiles")

    # Core RDKit descriptors
    rdkit_desc = get_rdkit_desc(smiles_series)
    return rdkit_desc

    # df_list = [
    #     smiles_series,
    #     rdkit_desc,
    # ]

    # final_data_df = pd.concat(df_list, axis=1)
    # return final_data_df

def _predict(final_data_df:pd.DataFrame, endpoints:list[str], device:str):
    '预测入口'
    # 载入config
    with open("config.json", "r") as f:
        config = json.load(f).get("ModelConfig")

    # 设置设备
    DEVICE = device # config.get("device", "cpu")

    # 建立estimator_list
    estimator_list = [
        ViabilityEstimator(
            device=DEVICE, working_dir=os.path.curdir, **config[name]
        ) for name in endpoints
    ]

    # input data
    input_data = final_data_df.reset_index(drop=True)
    shape = (len(estimator_list), input_data.shape[0])
    result_array = np.zeros(shape, dtype=object)
    # print(f"Shape of input data: {shape}")
    # print(f"SMILES: {input_data['smiles'].to_list()}")

    # 预测
    for idx in range(len(estimator_list)):
        estimator = estimator_list[idx]
        result_array[idx] = estimator.predict_auto(input_data)
    
    # 整理结果
    result_df = pd.DataFrame(result_array.T, columns=[i.date for i in estimator_list])
    # print(f"Shape of result_df: {result_df.shape}")
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

def hazard_identification(descriptors_data:dict, device:str):
    '分类模型'
    final_data_df = pd.DataFrame(descriptors_data)
    # print(f"Input shape: {final_data_df.shape}")
    result_df = _predict(
        final_data_df=final_data_df,
        endpoints=[
            'Hazard_viability',
            'Hazard_mitotox',
            'Hazard_apoptosis',
        ],
        device=device,
    )
    # print(f"Shape of model output: {result_df.shape}")

    # 总hazard等级
    hazard_result = []
    for line in result_df.itertuples():
        if (line.Hazard_viability == 'inactive') and (line.Hazard_mitotox == 'inactive') and (line.Hazard_apoptosis == 'inactive'):
            hazard_level = "inactive" 
        else:
            hazard_level = "active"
        hazard_result.append(hazard_level)
    hazard_series = pd.Series(hazard_result, name="Hazard")
    # print(f"Hazard shape: {hazard_series.shape}")

    # 合并
    result_df = pd.concat(
        [result_df, hazard_series],
        axis=1,
    )
    return result_df

def regression_modeling(descriptors_data:dict, hi_result_dict:dict, device:str):
    '回归模型'
    final_data_df = pd.DataFrame(descriptors_data)
    hi_result = pd.DataFrame(hi_result_dict)
    merged_df = pd.merge(final_data_df, hi_result, on="smiles")
    result_df = final_data_df[['smiles']].reset_index(drop=True)
    # step = 0
    # def update_step(step):
    #     step += 1
    #     st.session_state.reg_progress = f"Predicting: {step}/6"
    #     st.rerun()
    #     return step

    ENDPOINTS = ['viability', 'mitotox', 'apoptosis']
    FREE_NAMES = [f'IC20_free_{endpoint}' for endpoint in ENDPOINTS]
    CELL_NAMES = [f'IC20_cell_{endpoint}' for endpoint in ENDPOINTS]

    # 计算一下有哪些要跑
    reg_data = []
    opera_data = []
    for line in merged_df.itertuples():
        reg_free_endpoints = []
        reg_cell_endpoints = []
        for endpoint in ENDPOINTS:
            if getattr(line, f"Hazard_{endpoint}") == "active":
                reg_free_endpoints.append(f"IC20_free_{endpoint}")
                reg_cell_endpoints.append(f"IC20_cell_{endpoint}")

        if len(reg_free_endpoints) == 0:
            # 全inactive的话要调用OPERA
            opera_data.append(
                [
                    line.smiles,
                ]
            )
        else:
            # 有active的部分要跑回归
            reg_data.append(
                [
                    line.smiles,
                    reg_free_endpoints,
                    reg_cell_endpoints,
                ]
            )

    reg_data = pd.DataFrame(reg_data, columns=['smiles', 'reg_free_endpoints', 'reg_cell_endpoints'])
    opera_data = pd.DataFrame(opera_data, columns=['smiles'])

    if reg_data.shape[0] != 0:
        # 跑回归的部分直接合并信息中已有的smiles和原来的final_data_df, 并且把多余的column drop掉
        reg_input_data = pd.merge(final_data_df, reg_data, on="smiles").drop(columns=['reg_free_endpoints', 'reg_cell_endpoints'])
        regression_result = _predict(
            final_data_df=reg_input_data,
            endpoints=FREE_NAMES + CELL_NAMES,
            device=device,
        )
        # ic20取三个endpoints中的最小值
        min_ic20_data = []
        for line in pd.merge(regression_result, reg_data, on='smiles').itertuples():
            # 取出三个endpoint中最小的ic20_free
            ic20_free_list = [getattr(line, endpoint) for endpoint in line.reg_free_endpoints]
            logIC20_free = min(ic20_free_list)
            ic20_cell_list = [getattr(line, endpoint) for endpoint in line.reg_cell_endpoints]
            logIC20_cell = min(ic20_cell_list)
            min_ic20_data.append(
                dict(
                    logIC20_cell=logIC20_cell,
                    logIC20_free=logIC20_free,
                )
            )
        min_ic20_data = pd.DataFrame(min_ic20_data)
        regression_result = pd.concat(
            [regression_result, min_ic20_data],
            axis=1,
        )
    else:
        regression_result = pd.DataFrame()

    if opera_data.shape[0] != 0:
        # 跑opera的部分照常就行
        opera_result = opera_predict(opera_data['smiles'].to_numpy())
        ivd_result = calculate_ivd(opera_result, "mitotox")
        ivd_result = ivd_result.rename(columns={'SMILES': 'smiles'})
    else:
        ivd_result = pd.DataFrame()

    # 合并结果
    merged_result = pd.concat(
        [regression_result, ivd_result],
        axis=0,
    )
    result_df = pd.merge(result_df, merged_result, on='smiles')

    return result_df
