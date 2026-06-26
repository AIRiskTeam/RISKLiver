import os, uuid, json
import pandas as pd
import numpy as np

from Descriptors.neutralize_new import neutralize_single_smiles

def opera_predict(smiles_array:np.ndarray):
    '调用opera预测IC20'
    # 读取配置
    with open("config.json", "r") as f:
        config = json.load(f)
    tmp_dir = config.get("tmpDir", "/dev/shm/riskliver")
    opera_executable = config.get("OPERAExecutable", "/usr/OPERA/application/run_OPERA.sh")
    matlab_runtime_dir = config.get("MATLABRuntimeDir", "/usr/local/MATLAB/MATLAB_Runtime/v912")
    
    # 创建临时文件
    os.makedirs(tmp_dir, exist_ok=True)
    u = uuid.uuid1()
    TMP_INPUT = os.path.join(tmp_dir, f"{u}.smi")
    TMP_OUTPUT = os.path.join(tmp_dir, f"{u}.csv")

    # 写入smiles
    smi_list = [f"{neutralize_single_smiles(i)}\tSMI-{idx}" for idx, i in enumerate(smiles_array)]
    with open(TMP_INPUT, "w", encoding='utf-8') as f:
        smi = "\n".join(smi_list)
        f.write(smi)
        # f.writelines(smi_list) # 这么写会导致出现无法解析的空行

    # 运行
    cmd = f"{opera_executable} {matlab_runtime_dir} -s {TMP_INPUT} -o {TMP_OUTPUT} -logP -logD -pKa -v 2 -c"
    if os.system(cmd) != 0:
        raise RuntimeError("OPERA raises an error.")
    
    # 读取
    return_csv = pd.read_csv(TMP_OUTPUT)
    os.remove(TMP_INPUT)
    os.remove(TMP_OUTPUT)
    return return_csv

