import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm
from sklearn.impute import SimpleImputer

from rdkit import Chem

HEAVY_METALS = {
    'As','Hg','Cd','Pb','Sn','Sb','Bi',
    'Cu','Zn','Ag','Au',
    'Ni','Co','Cr','Mn','Fe','Pt','Pd'
}

ALKALI_METALS = {'Li','Na','K','Rb','Cs','Mg','Ca','Sr','Ba'}
HALIDE_IONS = {'Cl','Br','I','F'}

def is_inorganic(mol):
    return all(atom.GetSymbol() != 'C' for atom in mol.GetAtoms())

def contains_heavy_metal(mol):
    return any(atom.GetSymbol() in HEAVY_METALS for atom in mol.GetAtoms())

def is_small_polyhalogen(mol):
    if mol.GetNumAtoms() <= 5:
        n_halo = sum(a.GetSymbol() in HALIDE_IONS for a in mol.GetAtoms())
        return n_halo >= 2
    return False

def strip_counter_ions(mol):
    rw = Chem.RWMol(mol)
    remove = []
    for atom in mol.GetAtoms():
        if (
            atom.GetDegree() == 0 and
            atom.GetFormalCharge() != 0 and
            (atom.GetSymbol() in ALKALI_METALS or atom.GetSymbol() in HALIDE_IONS)
        ):
            remove.append(atom.GetIdx())
    for idx in sorted(remove, reverse=True):
        rw.RemoveAtom(idx)
    parent = rw.GetMol()
    Chem.SanitizeMol(parent)
    return parent

def clean_smiles(smiles, REMOVE_INORGANIC=True, REMOVE_HEAVY_METAL=True, REMOVE_POLYHALO_SMALL=True, DESALT_COUNTER_IONS=True):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    if REMOVE_INORGANIC and is_inorganic(mol):
        return None

    if REMOVE_HEAVY_METAL and contains_heavy_metal(mol):
        return None

    if REMOVE_POLYHALO_SMALL and is_small_polyhalogen(mol):
        return None

    if DESALT_COUNTER_IONS:
        mol = strip_counter_ions(mol)
        if is_inorganic(mol):
            return None

    return mol


def generate_desc_parallel(
        smiles_series:pd.Series, smi_2_desc_method:callable, 
        njobs:int|None=None, dropna=True, **kwargs
    ):
    'col, smiles_series -> desc dataframe (parallel)'
    smi_array = smiles_series.to_numpy()
    TOTAL = smi_array.shape[0]
    desc_list = np.zeros(TOTAL).tolist()

    # 多进程计算
    futures = []
    with tqdm.tqdm(total=TOTAL) as pbar:
        with ProcessPoolExecutor(max_workers=njobs) as executor:
            # 提交进程
            for idx in range(TOTAL):
                smi = smi_array[idx]
                futures.append(executor.submit(smi_2_desc_method, idx, smi, **kwargs))
            # 收集结果
            for run_res in as_completed(futures):
                pbar.update(1)
                idx, value = run_res.result()
                desc_list[idx] = value

    # 处理nan
    desc_df = pd.DataFrame(desc_list)
    if dropna:
        desc_df = desc_df.dropna(axis=0)
    desc_df = desc_df.replace([np.inf, -np.inf], np.nan)

    # 均值填充
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    # 拟合数据并进行转换
    imp_mean.fit(desc_df)
    desc_df = imp_mean.transform(desc_df)

    # 生成df
    desc_df = pd.concat(
        [smiles_series.reset_index(drop=True), desc_df],
        axis=1,
    )

    # 返回值
    return desc_df


def generate_desc(smiles_series:pd.Series, cols:list, smi_2_desc_method:callable, **kwargs):
    '根据col和smiles_series自动生成desc'
    # 初始化
    CLIP_ABS = 1e6
    # invalid_smiles_list = []
    invalid_id_list = []
    smi_array = smiles_series.to_numpy()
    desc_array = np.zeros_like(smi_array, dtype=object)
    empty_desc = np.zeros_like(cols, dtype=np.float64)

    # 逐行计算
    for idx in tqdm.trange(smi_array.shape[0]):
        def _set_none(idx):
            # 把指定的id设成none
            invalid_id_list.append(idx)
            # invalid_smiles_list.append(idx)
            desc_array[idx] = empty_desc

        # 提取smiles
        smi = smi_array[idx]
        desc = smi_2_desc_method(smi, **kwargs)
        if desc is None:
            _set_none(idx)
        else:
            desc = np.clip(desc, -CLIP_ABS, CLIP_ABS)
            desc_array[idx] = desc

    # 均值填充
    desc_array = np.vstack(desc_array).astype(np.float64)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    # return desc_array
    # 拟合数据并进行转换
    imp_mean.fit(desc_array)
    desc_array = imp_mean.transform(desc_array)

    # 拼接并裁剪
    desc_df = pd.DataFrame(desc_array)

    # 生成df
    data = np.hstack(
        # smi_array要重塑成二维
        [smi_array.reshape(-1,1), desc_array]
    )
    desc_df = pd.DataFrame(data, columns=['smiles'] + cols)

    # 去掉失败项
    mask = np.full(desc_df.shape[0], True)
    mask[invalid_id_list] = False
    desc_df = desc_df.loc[mask]

    return desc_df #, invalid_smiles_list, invalid_id_list
