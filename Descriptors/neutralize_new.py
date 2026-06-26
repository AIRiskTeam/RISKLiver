import warnings
warnings.filterwarnings("ignore")

from rdkit import Chem

# 是否启用 OpenBabel fallback
USE_OPENBABEL_FALLBACK = True

try:
    from openbabel import pybel
    OPENBABEL_AVAILABLE = True
    print("[INFO] OpenBabel(pybel) available.")
except Exception as e:
    OPENBABEL_AVAILABLE = False
    pybel = None
    print(f"[WARNING] OpenBabel(pybel) NOT available. ({e})")

try:
    from rdkit.Chem.MolStandardize import rdMolStandardize
    _UNCHARGER = rdMolStandardize.Uncharger()
    UNCHARGER_AVAILABLE = True
    print("[INFO] RDKit Uncharger available.")
except Exception as e:
    _UNCHARGER = None
    UNCHARGER_AVAILABLE = False
    print(f"[WARNING] RDKit Uncharger NOT available. ({e})")

#SMILES 清理、解析、去盐函数
def clean_smiles_text(s: str) -> str:
    """Remove hidden characters that may break SMILES parsing."""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\u200b", "").replace("\ufeff", "").replace("\xa0", " ")
    return s.strip()


def mol_from_smiles_robust(smi: str):
    """
    Robust SMILES parser:
    1. RDKit
    2. OpenBabel fallback
    3. RDKit sanitize=False fallback
    """
    smi = clean_smiles_text(smi)

    if not smi:
        return None, "fail"

    # 1) RDKit first
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            return mol, "rdkit"
    except Exception:
        pass

    # 2) OpenBabel fallback
    if USE_OPENBABEL_FALLBACK and OPENBABEL_AVAILABLE:
        try:
            m = pybel.readstring("smi", smi)
            can = m.write("can").strip()
            mol2 = Chem.MolFromSmiles(can)
            if mol2 is not None:
                return mol2, "openbabel"
        except Exception:
            pass

    # 3) RDKit sanitize=False fallback
    try:
        mol3 = Chem.MolFromSmiles(smi, sanitize=False)
        if mol3 is not None:
            flags = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            Chem.SanitizeMol(mol3, sanitizeOps=flags)
            return mol3, "rdkit_nokek"
    except Exception:
        pass

    return None, "fail"


def has_carbon(mol: Chem.Mol) -> bool:
    return mol is not None and any(atom.GetSymbol() == "C" for atom in mol.GetAtoms())


def largest_carbon_fragment(mol: Chem.Mol):
    """
    Remove salts/counter-ions by keeping the largest carbon-containing fragment.
    """
    if mol is None:
        return None

    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    except Exception:
        frags_raw = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        frags = []
        for f in frags_raw:
            try:
                Chem.SanitizeMol(f)
                frags.append(f)
            except Exception:
                pass

    carbon_frags = [f for f in frags if has_carbon(f)]

    if not carbon_frags:
        return None

    carbon_frags.sort(
        key=lambda m: (m.GetNumHeavyAtoms(), m.GetNumAtoms()),
        reverse=True
    )

    return carbon_frags[0]


def mol_to_smi(mol: Chem.Mol):
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None
    
#封装成函数：输入 SMILES，返回 neutral SMILES
def neutralize_single_smiles(raw_smi: str, verbose: bool = True):
    """
    Convert one input SMILES to neutral SMILES.

    Logic:
    1. clean text
    2. robust parse
    3. keep largest carbon fragment
    4. neutralize using RDKit Uncharger
    5. return canonical isomeric neutral SMILES
    """
    raw_smi = clean_smiles_text(raw_smi)

    mol, src = mol_from_smiles_robust(raw_smi)

    if mol is None:
        if verbose:
            print("[ERROR] Failed to parse SMILES.")
        return None

    core = largest_carbon_fragment(mol)

    if core is None:
        if verbose:
            print("[ERROR] No carbon-containing fragment found.")
        return None

    if UNCHARGER_AVAILABLE and _UNCHARGER is not None:
        try:
            neutral_mol = _UNCHARGER.uncharge(Chem.Mol(core))
            try:
                Chem.SanitizeMol(neutral_mol)
            except Exception:
                pass
        except Exception:
            neutral_mol = core
    else:
        neutral_mol = core

    neutral_smi = mol_to_smi(neutral_mol)

    if verbose:
        print("[INPUT]  ", raw_smi)
        print("[PARSE]  ", src)
        print("[NEUTRAL]", neutral_smi)

    return neutral_smi
