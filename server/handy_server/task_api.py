"""Authenticated task workflows with explicit import and edit conflicts."""
import re
from datetime import date
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .notes_store import NoteConflict
from .task_store import TaskStore

ID = re.compile(r"^[0-9a-f]{32}$")
TaskStatus = Literal["todo", "in_progress", "done", "cancelled"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskCreate(Strict):
    title: str = Field(min_length=1, max_length=500)
    assignee: str | None = Field(default=None, max_length=120)
    due_date: date | None = None


class TaskUpdate(TaskCreate):
    version: int = Field(ge=1)
    status: TaskStatus


def attach_task_routes(app: FastAPI, config, meetings, require_auth):
    store = TaskStore(config.db_path)
    protected = [Depends(require_auth)]

    @app.get("/v1/tasks", dependencies=protected)
    def list_tasks(status: TaskStatus | None = None, offset: int = 0, limit: int = 50):
        if offset < 0 or not 1 <= limit <= 100:
            raise HTTPException(status_code=422, detail="Invalid task pagination")
        return store.list(status, offset, limit)

    @app.post("/v1/tasks", dependencies=protected, status_code=201)
    def create_task(body: TaskCreate):
        return store.create(body.title.strip(), body.assignee,
                            body.due_date.isoformat() if body.due_date else None)

    @app.get("/v1/tasks/{task_id}", dependencies=protected)
    def get_task(task_id: str):
        if not ID.fullmatch(task_id) or (task := store.get(task_id)) is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return task

    @app.put("/v1/tasks/{task_id}", dependencies=protected)
    def update_task(task_id: str, body: TaskUpdate):
        if not ID.fullmatch(task_id) or store.get(task_id) is None:
            raise HTTPException(status_code=404, detail="Task not found")
        try:
            return store.update(task_id, body.version, body.title.strip(), body.assignee,
                                body.due_date.isoformat() if body.due_date else None,
                                body.status)
        except NoteConflict:
            raise HTTPException(status_code=409, detail="Task changed on another device") from None

    @app.delete("/v1/tasks/{task_id}", dependencies=protected, status_code=204)
    def delete_task(task_id: str):
        if not ID.fullmatch(task_id) or not store.delete(task_id):
            raise HTTPException(status_code=404, detail="Task not found")

    @app.post("/v1/tasks/from-meeting/{job_id}", dependencies=protected)
    def import_tasks(job_id: str):
        if not ID.fullmatch(job_id):
            raise HTTPException(status_code=404, detail="Meeting not found")
        try:
            return store.import_meeting(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Meeting not found") from None
        except ValueError:
            raise HTTPException(status_code=409, detail="Meeting analysis incomplete") from None
