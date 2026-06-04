
import numpy as np

from rdkit import Chem
try:
    # 新版 RDKit 通常可用
    from rdkit.Chem.MolStandardize import rdMolStandardize
    _UNCHARGER = rdMolStandardize.Uncharger()
    HAS_UNCHARGER = True
except Exception:
    _UNCHARGER = None
    HAS_UNCHARGER = False


def mol_from_smiles_robust(smi: str):
    """
    解析 SMILES：
    - 先 RDKit
    - 失败则（若开启且可用）OpenBabel 转 canonical 再用 RDKit
    """
    if smi is None or (isinstance(smi, float) and np.isnan(smi)):
        return None
    smi = str(smi).strip()
    if not smi:
        return None

    mol = None
    try:
        mol = Chem.MolFromSmiles(smi)
    except Exception:
        mol = None

    # if mol is None and USE_OPENBABEL_FALLBACK and OPENBABEL_AVAILABLE:
    #     try:
    #         m = pybel.readstring("smi", smi)
    #         can = m.write("can").strip()
    #         mol = Chem.MolFromSmiles(can)
    #     except Exception:
    #         mol = None

    return mol


def has_carbon(mol: Chem.Mol) -> bool:
    """是否含碳（用于过滤无机物/选择含碳fragment）"""
    return mol is not None and any(a.GetSymbol() == "C" for a in mol.GetAtoms())


def _split_fragments(mol: Chem.Mol):
    """
    稳健切分 fragment：
    - 优先 sanitizeFrags=True
    - 失败则 sanitizeFrags=False + 手动 sanitize
    """
    if mol is None:
        return []
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        return list(frags)
    except Exception:
        frags_raw = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        frags = []
        for f in frags_raw:
            try:
                Chem.SanitizeMol(f)
                frags.append(f)
            except Exception:
                pass
        return frags


def largest_carbon_fragment(mol: Chem.Mol):
    """选最大“含碳”fragment；若不存在含碳fragment -> None"""
    if mol is None:
        return None
    frags = _split_fragments(mol)
    carbon_frags = [f for f in frags if has_carbon(f)]
    if not carbon_frags:
        return None
    carbon_frags.sort(key=lambda m: (m.GetNumHeavyAtoms(), m.GetNumAtoms()), reverse=True)
    return carbon_frags[0]


def neutralize_mol(mol: Chem.Mol):
    """
    中和电荷（核心需求：core SMILES 需要是中性的）
    - 使用 RDKit MolStandardize.Uncharger（若可用）
    - 失败时返回原 mol（不会让流程崩）
    """
    if mol is None:
        return None

    if HAS_UNCHARGER and _UNCHARGER is not None:
        try:
            m2 = _UNCHARGER.uncharge(mol)
            # 有时 uncharge 后需要再 sanitize 一下
            Chem.SanitizeMol(m2)
            return m2
        except Exception:
            return mol

    # 如果你的 RDKit 版本没有 Uncharger，就只能保持原样
    return mol


def standardize_smiles_core_largest_fragment_neutral(raw_smi: str, FILTER_NO_CARBON_CORE=True):
    """
    生成 CORE（中性）：
    - RAW -> mol
    - 取最大含碳 fragment（通常会把 counter-ion / 盐拆掉）
    - 对该 fragment 做“中和电荷”
    - 输出 canonical core SMILES
    """
    mol = mol_from_smiles_robust(raw_smi)
    if mol is None:
        return None

    core = largest_carbon_fragment(mol)
    if core is None:
        if FILTER_NO_CARBON_CORE:
            return None
        frags = _split_fragments(mol)
        if not frags:
            return None
        frags.sort(key=lambda m: (m.GetNumHeavyAtoms(), m.GetNumAtoms()), reverse=True)
        core = frags[0]

    # ★关键：在计算描述符前，把 core 中和
    core_neu = neutralize_mol(core)

    try:
        return Chem.MolToSmiles(core_neu, canonical=True)
    except Exception:
        return None
