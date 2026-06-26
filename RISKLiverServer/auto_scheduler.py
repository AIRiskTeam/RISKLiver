import re, json, asyncio, traceback, sys
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from .db_models import Task
from . import get_config

from Estimator.streamlit import hazard_identification, regression_modeling, calculate_descriptors

async def get_gpu_info(gpu_type="nvidia") -> list[dict]:
    '获取GPU可用信息'
    if gpu_type == "nvidia":
        # 叫一个nvidia-smi
        p = await asyncio.subprocess.create_subprocess_shell("nvidia-smi", stdout=asyncio.subprocess.PIPE)
        await p.wait()
        nvidia_output = (await p.stdout.read()).decode(encoding='utf-8')

        # 匹配字符串
        cuda_id_lines = re.finditer(
            pattern=r"(?<=\|)\s+\d+(\s|\d|\w|\.)+(?=(Off|On)\s+\|)", # NVIDIA GeForce RTX 5090 ...
            string=nvidia_output
        )
        usage_lines = re.findall(
            pattern=r"(?<=\|)\s+\d+\%\s+\S+(?=\s+\|\n)",
            string=nvidia_output
        )

        # 提取信息
        device_list = []
        for idx, match in enumerate(cuda_id_lines):
            i = match.group().strip()
            d = dict(
                name=f"cuda:{i.split(' ')[0]}",
                model=" ".join(i.split(' ')[1:]).strip(),
                usage=int(usage_lines[idx].split('%')[0]),
            )
            device_list.append(d)
        return device_list
    else:
        raise RuntimeError(f"Unsupported GPU type {gpu_type}.")

async def get_most_usable_gpu() -> str:
    '获取最能用的GPU'
    # 允许使用的GPU列表
    config = get_config()
    allow_devices = config.get("AllowDevices", [])
    gpu_type = config.get("GPUType")

    # 查找可用的GPU
    for gpu_info in await get_gpu_info(gpu_type):
        # 可用时记录
        gpu_name = gpu_info['name']
        gpu_usage = gpu_info['usage']
        gpu_model = gpu_info['model']
        if gpu_name in allow_devices and gpu_usage == 0:
            print(f"Allocating GPU {gpu_name} - {gpu_model}")
            return gpu_name
    else:
        return ""
    
async def _run_task(task, **kwargs):
    '运行任务'
    return task(**kwargs)

async def scheduled_check():
    '定时检查'
    FUNC_DICT = dict(
        calculate_descriptors=calculate_descriptors,
        hazard_identification=hazard_identification,
        regression_modeling=regression_modeling,
    )
    FUNC_KEYS = list(FUNC_DICT.keys())

    # 查询等待中的任务
    job_list = await Task.filter(status=0, task_name__in=FUNC_KEYS).values(
        'status', 'hash', 'task_name', 'kwargs'
        )
    # print(f"job_list: {job_list}")
    for _, task_dict in enumerate(job_list):
        # 分配GPU
        device = await get_most_usable_gpu()
        if device != "":
            # 获取到可用GPU时继续运行
            task_name = task_dict['task_name']
            kwargs = json.loads(task_dict['kwargs'])
            func = FUNC_DICT[task_name]
            hash = task_dict['hash']
            # 打印一下
            print(f"Received task {hash} ({task_name}).")
            await Task.filter(hash=hash).update(status=1)
            # 创建任务
            # task = asyncio.create_task(_run_task(func, device=device, **kwargs))
            # result = await task
            try:
                # result = func(device=device, **kwargs)
                result = await asyncio.to_thread(func, device=device, **kwargs)
                # when complete
                result_str = json.dumps(result.to_dict())
                await Task.filter(hash=hash).update(status=2, result=result_str)
            except Exception:
                # when error
                result = traceback.format_exc()
                print(result, file=sys.stderr)
                await Task.filter(hash=hash).update(status=3, result=result)

async def init():
    '初始化'
    INTERVAL = 10
    scheduler = AsyncIOScheduler()
    scheduler.add_job(scheduled_check, trigger="interval", seconds=INTERVAL)
    scheduler.start()
    print(f"Autobackup scheduler started (Interval: {INTERVAL}s)")
    return scheduler

