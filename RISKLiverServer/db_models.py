from tortoise.models import Model
from tortoise.fields import SmallIntField, TextField, CharField, DatetimeField

class Task(Model):
    hash = CharField(primary_key=True, max_length=32)    # A unique MD5 hash as task identifier
    status = SmallIntField()                # 0=In queue, 1=Running, 2=Complete, 3=Error
    kwargs = TextField()                    # Input data in json format
    task_name = TextField()                 # Task to run, to be filtered by corresponding components
    result = TextField(null=True)           # Output data if status==2 or traceback information if status==3
    create_time = DatetimeField(auto_now_add=True)
    update_time = DatetimeField(auto_now=True)
