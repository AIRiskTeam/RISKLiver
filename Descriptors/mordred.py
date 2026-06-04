import numpy as np
import pandas as pd
import tqdm

from . import _Mordred as md
from .utils import generate_desc
from .neutralize import mol_from_smiles_robust, neutralize_mol, standardize_smiles_core_largest_fragment_neutral

### PROPERTIES

def get_col_mordred():
    'Mordred参数名称'
    # 需要先计算一遍
    md_cols = md.MORD_CALC.pandas([], quiet=True).columns
    md_cols = [f"md_{col}" for col in md_cols]
    return md_cols

def get_col_rdkit():
    'rdkit描述符参数列表'
    return [f"rd_{col}" for col in md.RD_NAMES]

def get_col_anchor():
    'anchor参数列表'
    # 这个也只能先计算一遍
    anchor_dict = md.anchor_features("CC", "CC")
    return list(anchor_dict.keys())


### Mordred

def __get_mordred(smiles:str):
    '为单个smiles计算mordred'
    sanitized_smi = md.standardize_smiles_keep_charge_remove_counterions(smiles)
    md_dict = md.mordred_features_full_core(sanitized_smi)
    md_list = [i[1] for i in md_dict.items()]
    return md_list

def get_mordred(smiles_series:pd.Series, njobs=4):
    'smi -> mordred描述符'
    def fill_error(x):
        '填充所有的报错'
        if np.isscalar(x):
            return x
        else:
            return 0

    # 清理并修改原始smiles
    # sanitized_smiles_series = smiles_series.apply(md.standardize_smiles_keep_charge_remove_counterions)
    sanitized_smiles_series = smiles_series.apply(standardize_smiles_core_largest_fragment_neutral) # 中和电荷
    valid_mask = sanitized_smiles_series.notna().to_numpy()
    sanitized_smiles_series = sanitized_smiles_series.loc[valid_mask]

    # 生成mol
    def _generate_mol(sanitized_smi):
        sanitized_mol = mol_from_smiles_robust(sanitized_smi)
        sanitized_mol = neutralize_mol(sanitized_mol)
        return sanitized_mol
    # sanitized_mol_series = sanitized_smiles_series.apply(md.mol_from_smiles_robust)
    sanitized_mol_series = sanitized_smiles_series.apply(_generate_mol) # 中和电荷
    # print(sanitized_mol_series.to_list())

    # 计算
    md_df = md.MORD_CALC.pandas(sanitized_mol_series, nproc=njobs)

    # 重设列名
    columns = [f"md_{i}" for i in md_df.columns]
    md_df.columns = columns

    # 加上原始smiles
    valid_smiles_series = smiles_series.loc[valid_mask].reset_index(drop=True)
    desc_df = pd.concat(
        [valid_smiles_series, md_df],
        axis=1,
    )

    # 清理非数值类型
    desc_df = desc_df.map(fill_error)

    # 去除N/A
    desc_df = desc_df.dropna(axis=0)

    return desc_df


### RDkit

def __smi_2_desc(smiles:str):
    '为单个smiles计算描述符'
    # sanitized_smi = md.standardize_smiles_keep_charge_remove_counterions(smiles)
    sanitized_smi = standardize_smiles_core_largest_fragment_neutral(smiles)
    sanitized_mol = mol_from_smiles_robust(sanitized_smi)
    sanitized_mol = neutralize_mol(sanitized_mol)
    if sanitized_mol is not None:
        desc = np.nan_to_num(md.RD_CALC.CalcDescriptors(sanitized_mol), posinf=1e+6, neginf=-1e+6)
    else:
        desc = None

    return desc

def get_rdkit_desc(smiles_series:pd.Series):
    'smi -> rdkit描述符'
    cols = get_col_rdkit()

    desc_df = generate_desc(smiles_series, cols, __smi_2_desc)

    return desc_df


### Anchor

def __smi_2_anchor(smiles:str):
    '为单个smiles计算描述符'
    # 处理smiles
    # sanitized_smi = md.standardize_smiles_keep_charge_remove_counterions(smiles)
    sanitized_smi = standardize_smiles_core_largest_fragment_neutral(smiles)

    # 计算
    if sanitized_smi is not None:
        desc_dict = md.anchor_features(smiles, sanitized_smi)
        desc = np.array([desc_dict[key] for key in desc_dict])
    else:
        desc = None

    return desc

def get_anchor_desc(smiles_series:pd.Series):
    'smi -> anchor标签'
    cols = get_col_anchor()

    desc_df = generate_desc(smiles_series, cols, __smi_2_anchor)

    return desc_df