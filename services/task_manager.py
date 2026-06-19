import os
import sys
import json
import uuid
import time
import asyncio
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime

from config import TASK_RESULT_TTL_HOURS, TASK_MAX_QUEUE_SIZE, TASK_MAX_WORKERS


class TaskStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class Task:
    task_id: str
    task_type: str
    params: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: TaskStatus = TaskStatus.PENDING
    progress: int = 0
    progress_message: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    callback_url: Optional[str] = None
    worker_id: Optional[str] = None

    def to_dict(self, include_result: bool = True) -> Dict[str, Any]:
        d = {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status.value,
            "progress": self.progress,
            "progress_message": self.progress_message,
            "created_at": self._fmt_time(self.created_at),
            "started_at": self._fmt_time(self.started_at),
            "completed_at": self._fmt_time(self.completed_at),
            "elapsed_seconds": self._elapsed(),
            "callback_url": self.callback_url is not None,
            "error": self.error,
        }
        if include_result and self.result is not None:
            d["result"] = self.result
        if self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            d["result_url"] = f"/tasks/{self.task_id}/result"
        return d

    @staticmethod
    def _fmt_time(ts: Optional[float]) -> Optional[str]:
        if ts is None:
            return None
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

    def _elapsed(self) -> float:
        end = self.completed_at or time.time()
        if self.started_at is None:
            return 0.0
        return round(end - self.started_at, 3)


class AsyncTaskManager:
    def __init__(self, max_workers: int = TASK_MAX_WORKERS,
                  max_queue: int = TASK_MAX_QUEUE_SIZE,
                  ttl_hours: int = TASK_RESULT_TTL_HOURS):
        self.tasks: Dict[str, Task] = {}
        self.max_workers = max_workers
        self.max_queue = max_queue
        self.ttl_seconds = ttl_hours * 3600
        self.task_handlers: Dict[str, Callable] = {}
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        self._workers: List[asyncio.Task] = []
        self._lock = threading.Lock()
        self._running = False
        self._progress_callbacks: Dict[str, List[Callable]] = {}

    def register_handler(self, task_type: str, handler: Callable):
        self.task_handlers[task_type] = handler

    def start(self):
        if self._running:
            return
        self._running = True
        loop = asyncio.get_event_loop()
        for i in range(self.max_workers):
            worker = loop.create_task(self._worker_loop(i))
            self._workers.append(worker)
        loop.create_task(self._cleanup_loop())
        print(f"[TaskManager] 任务管理器启动: workers={self.max_workers}, queue={self.max_queue}")

    def stop(self):
        self._running = False
        for w in self._workers:
            w.cancel()
        self._workers.clear()
        print("[TaskManager] 任务管理器已停止")

    async def _worker_loop(self, worker_id: int):
        while self._running:
            try:
                task_id = await asyncio.wait_for(self._queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue

            task = self.tasks.get(task_id)
            if task is None or task.status == TaskStatus.CANCELLED:
                self._queue.task_done()
                continue

            task.worker_id = f"worker-{worker_id}"
            task.status = TaskStatus.PROCESSING
            task.started_at = time.time()
            task.progress_message = "正在处理"
            self._emit_progress(task)

            handler = self.task_handlers.get(task.task_type)
            if handler is None:
                task.status = TaskStatus.FAILED
                task.error = f"未注册的任务类型: {task.task_type}"
                task.completed_at = time.time()
                self._emit_progress(task)
                self._queue.task_done()
                await self._try_callback(task)
                continue

            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(task.params, self._make_progress_fn(task))
                else:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, handler, task.params, self._make_progress_fn(task)
                    )

                task.result = result
                task.progress = 100
                task.progress_message = "处理完成"
                task.status = TaskStatus.COMPLETED
                task.completed_at = time.time()

            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = f"{type(e).__name__}: {str(e)}"
                task.completed_at = time.time()
                print(f"[TaskManager] 任务 {task_id} 失败: {task.error}")

            finally:
                self._emit_progress(task)
                self._queue.task_done()
                await self._try_callback(task)

    def _make_progress_fn(self, task: Task) -> Callable[[int, str], None]:
        def _update(progress: int, message: str = ""):
            if task.status != TaskStatus.PROCESSING:
                return
            task.progress = max(0, min(100, int(progress)))
            if message:
                task.progress_message = message
            self._emit_progress(task)
        return _update

    def _emit_progress(self, task: Task):
        callbacks = self._progress_callbacks.get(task.task_id, [])
        for cb in callbacks:
            try:
                cb(task.to_dict())
            except Exception:
                pass

    async def _try_callback(self, task: Task):
        if not task.callback_url or task.status not in (
            TaskStatus.COMPLETED, TaskStatus.FAILED
        ):
            return
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                await session.post(task.callback_url, json=task.to_dict(), timeout=aiohttp.ClientTimeout(10))
        except Exception as e:
            print(f"[TaskManager] 回调失败 {task.task_id} -> {task.callback_url}: {e}")

    async def _cleanup_loop(self):
        while self._running:
            await asyncio.sleep(600)
            with self._lock:
                now = time.time()
                expired = []
                for tid, task in self.tasks.items():
                    if task.completed_at and (now - task.completed_at) > self.ttl_seconds:
                        expired.append(tid)
                for tid in expired:
                    del self.tasks[tid]
                    if tid in self._progress_callbacks:
                        del self._progress_callbacks[tid]
                if expired:
                    print(f"[TaskManager] 清理过期任务: {len(expired)} 个")

    def submit(self, task_type: str, params: Dict[str, Any],
                callback_url: Optional[str] = None) -> Dict[str, Any]:
        if task_type not in self.task_handlers:
            return {
                "success": False,
                "error": f"未注册的任务类型: {task_type}. 可用: {list(self.task_handlers.keys())}"
            }
        if self._queue.qsize() >= self.max_queue:
            return {
                "success": False,
                "error": f"任务队列已满 ({self.max_queue})，请稍后重试"
            }

        task_id = uuid.uuid4().hex
        task = Task(
            task_id=task_id,
            task_type=task_type,
            params=params,
            status=TaskStatus.QUEUED,
            progress_message="已加入队列，等待处理",
            callback_url=callback_url
        )

        with self._lock:
            self.tasks[task_id] = task
        try:
            self._queue.put_nowait(task_id)
        except asyncio.QueueFull:
            with self._lock:
                del self.tasks[task_id]
            return {"success": False, "error": "任务队列已满"}

        return {
            "success": True,
            "task_id": task_id,
            "status": task.status.value,
            "queue_position": self._queue.qsize(),
            "poll_url": f"/tasks/{task_id}",
            "result_url": f"/tasks/{task_id}/result",
            "cancel_url": f"/tasks/{task_id}/cancel"
        }

    def get_status(self, task_id: str, include_result: bool = False) -> Optional[Dict[str, Any]]:
        task = self.tasks.get(task_id)
        if task is None:
            return None
        return task.to_dict(include_result=include_result)

    def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        task = self.tasks.get(task_id)
        if task is None:
            return None
        if task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            return {"status": task.status.value, "message": "任务尚未完成"}
        return task.to_dict(include_result=True)

    def cancel(self, task_id: str) -> Dict[str, Any]:
        task = self.tasks.get(task_id)
        if task is None:
            return {"success": False, "error": "任务不存在"}
        if task.status in (TaskStatus.PROCESSING,):
            task.progress_message = "任务正在执行，取消请求已记录，将尽快终止"
            task.status = TaskStatus.CANCELLED
            return {"success": True, "message": "已请求取消，任务将在下一个检查点终止", "task_id": task_id}
        if task.status in (TaskStatus.QUEUED, TaskStatus.PENDING):
            task.status = TaskStatus.CANCELLED
            task.progress_message = "任务已取消"
            return {"success": True, "message": "任务已取消", "task_id": task_id}
        return {"success": False, "error": f"任务处于 {task.status.value} 状态，无法取消", "task_id": task_id}

    def list_tasks(self, status_filter: Optional[TaskStatus] = None,
                    limit: int = 50) -> List[Dict[str, Any]]:
        tasks = sorted(
            self.tasks.values(),
            key=lambda t: t.created_at,
            reverse=True
        )
        if status_filter:
            tasks = [t for t in tasks if t.status == status_filter]
        return [t.to_dict(include_result=False) for t in tasks[:limit]]

    def on_progress(self, task_id: str, callback: Callable):
        if task_id not in self._progress_callbacks:
            self._progress_callbacks[task_id] = []
        self._progress_callbacks[task_id].append(callback)


_global_task_manager: Optional[AsyncTaskManager] = None


def get_task_manager() -> AsyncTaskManager:
    global _global_task_manager
    if _global_task_manager is None:
        _global_task_manager = AsyncTaskManager()
    return _global_task_manager
