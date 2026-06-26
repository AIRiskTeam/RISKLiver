from fastapi import FastAPI
import uvicorn, os
from contextlib import asynccontextmanager

from pydantic import BaseModel

from RISKLiverServer import get_config, db, auto_scheduler

from RISKLiverServer.db_models import Task
from tortoise.exceptions import DoesNotExist
from tortoise.context import TortoiseContext


class TaskData(BaseModel):
    'task的数据格式'
    hash: str
    status: int
    kwargs: str
    task_name: str

class ReqTaskData(BaseModel):
    '仅hash'
    hash: str


@asynccontextmanager
async def lifespan(_app):
    '生命周期管理'
    # config.load()
    await db.init_db()
    task_scheduler = await auto_scheduler.init()

    yield

    task_scheduler.shutdown()
    await db.close()

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def hello_world():
    return dict(code="200", message="hello world")

@app.post("/status")
async def get_status(data: ReqTaskData):
    try:
        res = await Task.get(hash=data.hash).values('status','hash')
    except DoesNotExist:
        return dict(status=-1)
    else:
        return dict(status=res['status'])
    
@app.post("/retry")
async def retry_task(data: ReqTaskData):
    await Task.filter(hash=data.hash).update(status=0)

@app.post("/submit")
async def submit_data(data: TaskData):
    if await Task.get_or_none(hash=data.hash) is None:
        await Task.create(hash=data.hash, status=data.status, kwargs=data.kwargs, task_name=data.task_name, result="")
    # await Task.filter(hash=data.hash).update(status=data.status, kwargs=data.kwargs, task_name=data.task_name, result="")

@app.post("/result")
async def get_result(data: ReqTaskData):
    try:
        res = await Task.get(hash=data.hash).values('status','hash','result')
    except DoesNotExist:
        return dict(status=-1, result='[{"message": "No such task."}]')
    else:
        if not res:
            return dict(status=-1, result='[{"message": "No such task."}]')
        else:
            return res

if __name__ == "__main__":
    config = get_config()
    uvicorn.run(
        app, 
        # port=int(os.environ.get("FASTAPI_PORT", 3000)),
        # host="127.0.0.1",
        host=config.get('Host', "127.0.0.1"),
        port=config.get('Port', 6310),
    )
