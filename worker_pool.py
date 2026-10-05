import base64
import json
import queue
import tempfile
import threading
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Literal

import socketio
from PIL import Image

from invokeai.app.invocations.baseinvocation import BaseInvocation
from invokeai.app.invocations.primitives import ImageOutput, VideoOutput
from invokeai.app.services.board_records.board_records_common import BoardVisibility
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.session_processor.session_processor_common import CanceledException, ProgressImage
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.video_thumbnails import probe_video_with_codec

from .model_transfer import ModelTransferCancelled, install_missing_model
from .remote_client import RemoteConfig, RemoteInvokeClient, RemoteInvokeError
from .remote_config import ConfiguredRemote, enabled_remotes, find_remote

_LOCAL_GALLERY_IMPORT_LOCK = threading.Lock()
_REMOTE_QUEUE_COORDINATORS_LOCK = threading.Lock()
_REMOTE_QUEUE_COORDINATORS: dict[tuple[str, str], "_RemoteQueueCoordinator"] = {}
_REMOTE_WORKER_SLOT_LOCKS_GUARD = threading.Lock()
_REMOTE_WORKER_SLOT_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_REMOTE_ONLY_HANDOFF_WAITERS_LOCK = threading.Lock()
_REMOTE_ONLY_HANDOFF_WAITERS: dict[tuple[str, str], int] = {}


REMOTE_NODE_TYPES = {
    "irw_remote_workflow",
    "irw_builtin_remote_worker_dispatch",
}
REMOTE_NODE_IDS = {"__irw_remote_worker_dispatch__"}


def _strip_remote_nodes(graph: dict[str, Any]) -> list[str]:
    nodes = graph.get("nodes")
    if not isinstance(nodes, dict):
        raise RemoteInvokeError("Current workflow has no graph nodes")

    removed = {
        str(node_id)
        for node_id, node in nodes.items()
        if isinstance(node, dict)
        and (
            str(node.get("type") or "") in REMOTE_NODE_TYPES
            or str(node_id) in REMOTE_NODE_IDS
        )
    }
    for node_id in removed:
        nodes.pop(node_id, None)

    edges = graph.get("edges")
    if isinstance(edges, list) and removed:
        graph["edges"] = [
            edge
            for edge in edges
            if not (
                isinstance(edge, dict)
                and (
                    str((edge.get("source") or {}).get("node_id")) in removed
                    or str((edge.get("destination") or {}).get("node_id")) in removed
                )
            )
        ]

    if not nodes:
        raise RemoteInvokeError("Nothing remains after removing Remote Workflow helper nodes")
    return sorted(removed)


def _get_board_id(node: dict[str, Any]) -> str | None:
    board = node.get("board")
    if isinstance(board, dict):
        board_id = board.get("board_id")
        if isinstance(board_id, str) and board_id.strip():
            return board_id.strip()

    board_id = node.get("board_id")
    if isinstance(board_id, str) and board_id.strip():
        return board_id.strip()

    return None


def _capture_local_output_board(graph: dict[str, Any]) -> str | None:
    """Capture an explicit board only from nodes that save visible Gallery media.

    Workflow graphs can contain many board-aware nodes whose outputs are
    intermediate. Those board values are not the destination for the finished
    remote result. In webv2, Save to Gallery compiles to
    is_intermediate=False, so only those nodes may supply the explicit
    destination board here.

    Auto/None compile without a board value; those are resolved separately from
    the persisted Gallery selection.
    """
    nodes = graph.get("nodes")
    if not isinstance(nodes, dict):
        return None

    for node in nodes.values():
        if not isinstance(node, dict) or node.get("is_intermediate") is not False:
            continue
        board_id = _get_board_id(node)
        if board_id:
            return board_id

    return None


def _strip_board_assignments(graph: dict[str, Any]) -> int:
    nodes = graph.get("nodes")
    if not isinstance(nodes, dict):
        return 0

    changed = 0
    for node in nodes.values():
        if not isinstance(node, dict):
            continue
        touched = False
        if isinstance(node.get("board"), dict):
            node["board"] = None
            touched = True
        if "board_id" in node and node.get("board_id") is not None:
            node["board_id"] = None
            touched = True
        if touched:
            changed += 1
    return changed


def _disable_cache(graph: dict[str, Any]) -> int:
    nodes = graph.get("nodes")
    if not isinstance(nodes, dict):
        return 0

    changed = 0
    for node in nodes.values():
        if isinstance(node, dict):
            node["use_cache"] = False
            changed += 1
    return changed


def _enrich_model_hashes(graph: dict[str, Any], services: Any) -> int:
    changed = 0

    def visit(value: Any) -> None:
        nonlocal changed
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return

        if all(field in value for field in ("key", "name", "base", "type")):
            key = str(value.get("key") or "")
            if key and not value.get("hash"):
                try:
                    config = services.model_manager.store.get_model(key)
                    model_hash = str(getattr(config, "hash", "") or "").strip()
                except Exception:
                    model_hash = ""
                if model_hash:
                    value["hash"] = model_hash
                    changed += 1

        for item in value.values():
            visit(item)

    visit(graph.get("nodes"))
    return changed


_DATE_BOARD_ID_PREFIX = "by_date:"


def _iter_project_gallery_values(project_data: dict[str, Any]):
    """Yield Gallery widget value dictionaries from current and legacy project shapes."""

    widget_instances = project_data.get("widgetInstances")
    if isinstance(widget_instances, dict):
        for instance in widget_instances.values():
            if not isinstance(instance, dict) or instance.get("typeId") != "gallery":
                continue
            state = instance.get("state")
            values = state.get("values") if isinstance(state, dict) else None
            if isinstance(values, dict):
                yield values

    widget_states = project_data.get("widgetStates")
    if isinstance(widget_states, dict):
        gallery = widget_states.get("gallery")
        values = gallery.get("values") if isinstance(gallery, dict) else None
        if isinstance(values, dict):
            yield values


def _board_exists(services: Any, board_id: str) -> bool:
    try:
        services.board_records.get(board_id)
        return True
    except Exception:
        return False


def _resolve_gallery_board(
    queue_item: Any,
    services: Any,
    logger: Any,
) -> str | None:
    """Resolve webv2's persisted Gallery destination for a queue item."""

    project_id = getattr(queue_item, "project_id", None)
    user_id = getattr(queue_item, "user_id", None)
    if not isinstance(project_id, str) or not project_id or not isinstance(user_id, str) or not user_id:
        return None

    project_records = getattr(services, "project_records", None)
    if project_records is None:
        return None

    project_board_id: str | None = None
    try:
        candidate = project_records.get_board_id(user_id, project_id)
        if isinstance(candidate, str) and candidate:
            project_board_id = candidate
    except Exception:
        pass

    try:
        try:
            record = project_records.get(
                user_id,
                project_id,
                max_canvas_schema_version=2_147_483_647,
            )
        except TypeError:
            record = project_records.get(user_id, project_id)
    except Exception as exc:
        if project_board_id:
            logger.warning(
                f"Remote Workflow: could not read project Gallery selection; "
                f"using project board {project_board_id}: {exc}"
            )
        return project_board_id

    project_data = getattr(record, "data", None)
    if not isinstance(project_data, dict):
        return project_board_id

    for values in _iter_project_gallery_values(project_data):
        selected_board_id = values.get("selectedBoardId")
        selected_board_id = selected_board_id.strip() if isinstance(selected_board_id, str) else None

        if selected_board_id:
            if selected_board_id == "none":
                logger.info(
                    "Remote Workflow: resolved Gallery destination to Uncategorized "
                    "from the selected Gallery board"
                )
                return None

            if not selected_board_id.startswith(_DATE_BOARD_ID_PREFIX) and _board_exists(
                services, selected_board_id
            ):
                logger.info(
                    f"Remote Workflow: resolved Gallery destination board {selected_board_id} "
                    "from the selected Gallery board"
                )
                return selected_board_id

        widget_project_board_id = values.get("projectBoardId")
        if isinstance(widget_project_board_id, str):
            widget_project_board_id = widget_project_board_id.strip()
            if widget_project_board_id and _board_exists(services, widget_project_board_id):
                logger.info(
                    f"Remote Workflow: resolved Gallery destination board {widget_project_board_id} "
                    "from the Gallery project board"
                )
                return widget_project_board_id

    if project_board_id and _board_exists(services, project_board_id):
        logger.info(
            f"Remote Workflow: resolved Gallery destination board {project_board_id} "
            "from the persisted project"
        )
        return project_board_id

    return None

def _build_remote_graph_from_queue_item(
    queue_item: Any,
    services: Any,
    logger: Any,
) -> tuple[dict[str, Any], str | None]:
    try:
        current_item = queue_item.model_dump(mode="json")
        source_graph = current_item["session"]["graph"]
    except Exception as exc:
        raise RemoteInvokeError("Could not read executable graph from queue item") from exc

    if not isinstance(source_graph, dict) or not isinstance(source_graph.get("nodes"), dict):
        raise RemoteInvokeError("Queue item did not contain a usable executable graph")

    remote_graph = deepcopy(source_graph)
    remote_graph["id"] = str(uuid.uuid4())

    local_board_id = _capture_local_output_board(remote_graph)
    removed = _strip_remote_nodes(remote_graph)
    boards = _strip_board_assignments(remote_graph)
    cache_nodes = _disable_cache(remote_graph)
    hashes = _enrich_model_hashes(remote_graph, services)

    logger.info(
        f"Remote Workflow: captured {len(source_graph.get('nodes', {}))} nodes; "
        f"removed remote helpers={removed}; stripped board fields from {boards} nodes; "
        f"disabled cache on {cache_nodes} nodes; enriched {hashes} model hashes"
    )
    return remote_graph, local_board_id


def _decode_progress_image(event: dict[str, Any]) -> tuple[Image.Image | None, tuple[int, int] | None]:
    raw = event.get("image")
    if not isinstance(raw, dict):
        return None, None

    data_url = raw.get("dataURL")
    if not isinstance(data_url, str) or not data_url:
        return None, None

    try:
        encoded = data_url.split(",", 1)[1] if "," in data_url else data_url
        payload = base64.b64decode(encoded)
        with Image.open(BytesIO(payload)) as image:
            image.load()
            pil = image.copy()
        width = int(raw.get("width") or pil.width)
        height = int(raw.get("height") or pil.height)
        return pil, (width, height)
    except Exception:
        return None, None


def _decode_progress_event(
    event: dict[str, Any],
) -> tuple[str, float | None, Image.Image | None, tuple[int, int] | None]:
    message = str(event.get("message") or "Remote rendering")
    percentage = event.get("percentage")
    try:
        percentage = float(percentage) if percentage is not None else None
    except (TypeError, ValueError):
        percentage = None
    image, image_size = _decode_progress_image(event)
    return message, percentage, image, image_size


class _RemoteProgressSocket:
    def __init__(self, client: RemoteInvokeClient, queue_id: str, context: InvocationContext):
        self.client = client
        self.queue_id = queue_id
        self.context = context
        self.events: queue.Queue[dict[str, Any]] = queue.Queue()
        self.sio = socketio.Client(
            reconnection=False,
            logger=False,
            engineio_logger=False,
        )

        @self.sio.on("invocation_progress")
        def _on_progress(data):
            if isinstance(data, dict):
                self.events.put(data)

    def connect(self) -> bool:
        try:
            self.client.ensure_auth()
            token = self.client.token
            auth = {"token": token} if token else None
            self.sio.connect(
                self.client.config.base_url,
                auth=auth,
                socketio_path="ws/socket.io",
                wait_timeout=10,
            )
            self.sio.emit("subscribe_queue", {"queue_id": self.queue_id})
            return True
        except Exception as exc:
            self.context.logger.warning(
                f"Remote Workflow: preview socket unavailable; continuing without live preview: {exc}"
            )
            try:
                self.sio.disconnect()
            except Exception:
                pass
            return False

    def _next_matching_event(self, item_id: int) -> dict[str, Any] | None:
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                return None

            try:
                if int(event.get("item_id")) != int(item_id):
                    continue
            except Exception:
                continue
            return event

    def drain_for_item(self, item_id: int, remote_name: str = "Remote") -> None:
        while True:
            event = self._next_matching_event(item_id)
            if event is None:
                return

            message, percentage, image, image_size = _decode_progress_event(event)
            self.context.util.signal_progress(
                f"{remote_name} · {message}",
                percentage=percentage,
                image=image,
                image_size=image_size,
            )

    def drain_for_queue_item(
        self,
        item_id: int,
        import_meta: dict[str, Any],
        local_item_id: int,
        event_invocation: BaseInvocation,
        remote_name: str,
    ) -> None:
        while True:
            event = self._next_matching_event(item_id)
            if event is None:
                return

            message, percentage, image, image_size = _decode_progress_event(event)
            _emit_queue_item_progress(
                import_meta=import_meta,
                item_id=local_item_id,
                event_invocation=event_invocation,
                message=f"{remote_name} · {message}",
                percentage=percentage,
                image=image,
                image_size=image_size,
            )

    def close(self) -> None:
        try:
            self.sio.emit("unsubscribe_queue", {"queue_id": self.queue_id})
        except Exception:
            pass
        try:
            self.sio.disconnect()
        except Exception:
            pass


def _remote_workflow_invocation(queue_item: Any) -> tuple[str, BaseInvocation]:
    """Return the source id and real Remote Workflow invocation from a queue item."""
    try:
        graph_nodes = queue_item.session.graph.nodes
    except Exception as exc:
        raise RemoteInvokeError("Queue item has no executable graph nodes") from exc

    if not isinstance(graph_nodes, dict):
        raise RemoteInvokeError("Queue item has no executable graph nodes")

    for node_id, node in graph_nodes.items():
        get_type = getattr(node, "get_type", None)
        try:
            node_type = str(get_type()) if callable(get_type) else str(getattr(node, "type", "") or "")
        except Exception:
            node_type = str(getattr(node, "type", "") or "")
        if node_type == "irw_remote_workflow":
            return str(node_id), node

    raise RemoteInvokeError("Remote Workflow node was not found in the queue item")


def _build_import_metadata(
    queue_item: Any,
    services: Any,
    logger: Any,
    board_id: str | None,
    *,
    node_id: str | None = None,
) -> dict[str, Any]:
    workflow_json = None
    if queue_item.workflow:
        try:
            workflow_json = queue_item.workflow.model_dump_json()
        except Exception:
            pass

    graph_json = None
    try:
        if queue_item.session.graph:
            graph_json = queue_item.session.graph.model_dump_json()
    except Exception:
        pass

    return {
        "services": services,
        "workflow": workflow_json,
        "graph": graph_json,
        "session_id": str(queue_item.session_id),
        "node_id": node_id or _remote_workflow_invocation(queue_item)[0],
        "user_id": str(queue_item.user_id),
        "board_id": board_id,
        "logger": logger,
    }

def _get_event_queue_item(
    import_meta: dict[str, Any],
    item_id: int,
    event_invocation: BaseInvocation | None = None,
):
    """Load a real queue item and prepare an in-memory source mapping for events.

    Remotely-owned real queue rows bypass InvokeAI's normal invocation preparation
    for the Remote Workflow helper. Event builders still require a
    prepared_source_mapping entry, so add it only to this hydrated in-memory copy.
    Never persist this synthetic mapping into GraphExecutionState.
    """

    queue_item = import_meta["services"].session_queue.get_queue_item(item_id)

    if event_invocation is not None:
        invocation_id = str(event_invocation.id)
        queue_item.session.prepared_source_mapping[invocation_id] = invocation_id

    return queue_item


def _emit_queue_item_progress(
    import_meta: dict[str, Any],
    item_id: int,
    event_invocation: BaseInvocation,
    message: str,
    percentage: float | None = None,
    image: Image.Image | None = None,
    image_size: tuple[int, int] | None = None,
) -> None:
    """Relay remote progress against the remotely-owned real queue item."""

    queue_item = _get_event_queue_item(
        import_meta,
        item_id,
        event_invocation,
    )

    progress_image = None
    if image is not None:
        try:
            progress_image = ProgressImage.build(image, size=image_size)
        except Exception:
            progress_image = None

    import_meta["services"].events.emit_invocation_progress(
        queue_item=queue_item,
        invocation=event_invocation,
        message=message,
        percentage=percentage,
        image=progress_image,
    )


def _assert_background_save_access(
    services: Any,
    user_id: str,
    board_id: str | None,
) -> None:
    if not getattr(services.configuration, "multiuser", False):
        return

    user = services.users.get(user_id)
    if user is None or not user.is_active:
        raise PermissionError("Queue user is not authorized to save returned remote media")

    if board_id:
        board = services.board_records.get(board_id)
        if (
            not user.is_admin
            and board.user_id != user_id
            and board.board_visibility != BoardVisibility.Public
        ):
            raise PermissionError(
                "Queue user is not authorized to save returned remote media to this board"
            )


def _save_local_gallery_image(
    import_meta: dict[str, Any],
    image: Image.Image,
    *,
    metadata: str | None = None,
    source_node_id: str | None = None,
):
    services = import_meta["services"]
    user_id = import_meta["user_id"]

    _assert_background_save_access(services, user_id, import_meta.get("board_id"))

    # Serialize only the short local image save; remote work remains concurrent.
    with _LOCAL_GALLERY_IMPORT_LOCK:
        image_dto = services.images.create(
            image=image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=source_node_id or import_meta["node_id"],
            session_id=import_meta["session_id"],
            board_id=import_meta.get("board_id"),
            is_intermediate=False,
            metadata=metadata,
            workflow=import_meta["workflow"],
            graph=import_meta["graph"],
            user_id=user_id,
        )

    # Background imports bypass the upload API, so emit its Gallery refresh event.
    board = None
    if image_dto.board_id:
        try:
            board = services.board_records.get(image_dto.board_id)
        except Exception as exc:
            import_meta["logger"].warning(
                f"Remote Workflow: imported image {image_dto.image_name}, but could not "
                f"load board {image_dto.board_id} for image_uploaded routing: {exc}"
            )

    services.events.emit_image_uploaded(
        image_dto,
        user_id=user_id,
        board=board,
        shared_user_ids=[],
    )
    import_meta["logger"].info(
        f"Remote Workflow: emitted image_uploaded for {image_dto.image_name}"
    )

    return image_dto


def _save_local_gallery_video(
    import_meta: dict[str, Any],
    video_bytes: bytes,
    metadata: str | None,
    *,
    source_node_id: str | None = None,
):
    services = import_meta["services"]
    user_id = import_meta["user_id"]

    _assert_background_save_access(services, user_id, import_meta.get("board_id"))

    outputs_path = services.configuration.outputs_path
    if outputs_path is None:
        raise RemoteInvokeError("Primary InvokeAI has no configured outputs path for video import")

    stage_dir = Path(outputs_path) / "videos"
    stage_dir.mkdir(parents=True, exist_ok=True)
    stage_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            prefix=".irw_remote_",
            suffix=".mp4",
            dir=stage_dir,
            delete=False,
        ) as file:
            stage_path = Path(file.name)
            file.write(video_bytes)

        width, height, duration, fps, _codec = probe_video_with_codec(stage_path)

        with _LOCAL_GALLERY_IMPORT_LOCK:
            video_dto = services.videos.create(
                source_path=stage_path,
                width=width,
                height=height,
                duration=duration,
                fps=fps,
                video_origin=ResourceOrigin.INTERNAL,
                video_category=ImageCategory.GENERAL,
                board_id=import_meta.get("board_id"),
                is_intermediate=False,
                metadata=metadata,
                workflow=import_meta["workflow"],
                graph=import_meta["graph"],
                session_id=import_meta["session_id"],
                node_id=source_node_id or import_meta["node_id"],
                user_id=user_id,
            )

        board = None
        if video_dto.board_id:
            try:
                board = services.board_records.get(video_dto.board_id)
            except Exception as exc:
                import_meta["logger"].warning(
                    f"Remote Workflow: imported video {video_dto.video_name}, but could not "
                    f"load board {video_dto.board_id} for video_uploaded routing: {exc}"
                )

        services.events.emit_video_uploaded(
            video_dto,
            user_id=user_id,
            board=board,
            shared_user_ids=[],
        )
        import_meta["logger"].info(
            f"Remote Workflow: emitted video_uploaded for {video_dto.video_name}"
        )
        return video_dto
    finally:
        if stage_path is not None:
            stage_path.unlink(missing_ok=True)


def _import_named_remote_video_dtos(
    client: RemoteInvokeClient,
    remote_names: list[str],
    import_meta: dict[str, Any],
    source_ids: dict[str, str] | None = None,
) -> list[Any]:
    imported: list[Any] = []
    source_ids = source_ids or {}
    for remote_video_name in remote_names:
        metadata = client.get_video_metadata(remote_video_name)
        video_bytes = client.download_video(remote_video_name)
        imported.append(
            _save_local_gallery_video(
                import_meta,
                video_bytes,
                metadata,
                source_node_id=source_ids.get(remote_video_name),
            )
        )
    return imported


def _import_named_remote_image_dtos(
    client: RemoteInvokeClient,
    remote_names: list[str],
    import_meta: dict[str, Any],
    source_ids: dict[str, str] | None = None,
) -> list[Any]:
    imported: list[Any] = []
    source_ids = source_ids or {}
    for remote_image_name in remote_names:
        metadata = client.get_image_metadata(remote_image_name)
        image = client.download_image(remote_image_name)
        imported.append(
            _save_local_gallery_image(
                import_meta,
                image,
                metadata=metadata,
                source_node_id=source_ids.get(remote_image_name),
            )
        )
    return imported


def _cleanup_remote_artifacts(
    *,
    client: RemoteInvokeClient,
    remote: ConfiguredRemote,
    remote_item_id: int,
    remote_image_names: list[str],
    remote_video_names: list[str],
    delete_remote_images: bool,
    delete_remote_videos: bool,
    delete_remote_queue_item: bool,
    logger: Any,
) -> None:
    if delete_remote_images and remote_image_names:
        deleted_images = 0
        for remote_image_name in remote_image_names:
            try:
                client.delete_image(remote_image_name)
                deleted_images += 1
            except Exception as exc:
                logger.warning(
                    f"Remote Workflow [{remote.name}]: could not delete remote image "
                    f"{remote_image_name}: {exc}"
                )
        if deleted_images:
            logger.info(
                f"Remote Workflow [{remote.name}]: deleted {deleted_images} remote image(s)"
            )

    if delete_remote_videos and remote_video_names:
        deleted_videos = 0
        for remote_video_name in remote_video_names:
            try:
                client.delete_video(remote_video_name)
                deleted_videos += 1
            except Exception as exc:
                logger.warning(
                    f"Remote Workflow [{remote.name}]: could not delete remote video "
                    f"{remote_video_name}: {exc}"
                )
        if deleted_videos:
            logger.info(
                f"Remote Workflow [{remote.name}]: deleted {deleted_videos} remote video(s)"
            )

    if delete_remote_queue_item:
        try:
            client.delete_item(remote_item_id, queue_id=remote.queue_id)
            logger.info(
                f"Remote Workflow [{remote.name}]: deleted remote queue item {remote_item_id}"
            )
        except Exception as exc:
            logger.warning(
                f"Remote Workflow [{remote.name}]: could not delete remote queue item "
                f"{remote_item_id}: {exc}"
            )


def _mark_remaining_local_nodes_skipped(context: InvocationContext) -> None:
    """Make Remote Only end the local session successfully after this node returns.

    This deliberately uses GraphExecutionState internals. The Remote Workflow node is
    scheduled first, so no normal local work should have executed yet.
    """

    session = context._data.queue_item.session
    current_exec_id = str(context._data.invocation.id)
    current_source_id = str(session.prepared_source_mapping.get(current_exec_id, current_exec_id))

    # Prepared nodes may already be sitting in class ready queues. Mark them skipped
    # and remove them so they cannot run after this invocation completes.
    for exec_id in list(session.execution_graph.nodes.keys()):
        exec_id = str(exec_id)
        if exec_id == current_exec_id:
            continue
        session.executed.add(exec_id)
        try:
            session._set_prepared_exec_state(exec_id, "skipped")
        except Exception:
            pass
        try:
            session._remove_from_ready_queues(exec_id)
        except Exception:
            pass

    # is_complete() checks the source graph node ids. Mark all source nodes except
    # this helper as executed/skipped. The normal completion path will add this
    # helper's source id when invoke() returns its output.
    for source_id in list(session.graph.nodes.keys()):
        source_id = str(source_id)
        if source_id == current_source_id:
            continue
        session.executed.add(source_id)
        if source_id not in session.executed_history:
            session.executed_history.append(source_id)

    context.logger.info(
        f"Remote Workflow: Remote Only marked {max(0, len(session.graph.nodes) - 1)} local source nodes skipped"
    )


@dataclass(frozen=True)
class _RemoteWorkflowSettings:
    mode: Literal["Distributed", "Remote Only"]
    workers: Literal["All Remotes", "Specific"]
    specific_remote: str
    import_to_gallery: bool
    cleanup_remote_when_done: bool
    cancel_remote_with_local: bool
    install_missing_models: bool
    use_primary_hf_token: bool
    poll_interval_seconds: float
    timeout_seconds: int


@dataclass
class _RemoteJob:
    remote: ConfiguredRemote
    client: RemoteInvokeClient
    item_id: int
    preview_socket: _RemoteProgressSocket | None = None
    input_image_names: list[str] = field(default_factory=list)
    input_video_names: list[str] = field(default_factory=list)


@dataclass
class _ImportedRemoteResult:
    image_names: list[str]
    video_names: list[str]
    image_dtos: list[Any]
    video_dtos: list[Any]
    final_preview: Image.Image | None = None


def _remote_media_source_ids(
    completed_item: dict[str, Any],
) -> tuple[dict[str, str], dict[str, str]]:
    """Map worker media names back to their real source graph node ids."""
    session = completed_item.get("session")
    if not isinstance(session, dict):
        return {}, {}

    results = session.get("results")
    if not isinstance(results, dict):
        return {}, {}

    raw_mapping = session.get("prepared_source_mapping")
    prepared_source_mapping = raw_mapping if isinstance(raw_mapping, dict) else {}
    image_sources: dict[str, str] = {}
    video_sources: dict[str, str] = {}

    for result_id, result in results.items():
        if not isinstance(result, dict):
            continue

        result_key = str(result_id)
        mapped = prepared_source_mapping.get(result_key)
        source_id = str(mapped) if isinstance(mapped, str) and mapped else result_key

        image = result.get("image")
        if isinstance(image, dict):
            image_name = image.get("image_name")
            if isinstance(image_name, str) and image_name:
                image_sources[image_name] = source_id

        images = result.get("images")
        if isinstance(images, list):
            for entry in images:
                if isinstance(entry, dict):
                    image_name = entry.get("image_name")
                    if isinstance(image_name, str) and image_name:
                        image_sources[image_name] = source_id

        video_candidates: list[Any] = [result.get("video"), result]
        videos = result.get("videos")
        if isinstance(videos, list):
            video_candidates.extend(videos)

        for entry in video_candidates:
            if not isinstance(entry, dict):
                continue
            video_name = entry.get("video_name")
            if isinstance(video_name, str) and video_name:
                video_sources[video_name] = source_id

    return image_sources, video_sources


def _import_completed_remote_result(
    job: _RemoteJob,
    completed_item: dict[str, Any],
    import_meta: dict[str, Any],
    settings: _RemoteWorkflowSettings,
    *,
    download_final_preview: bool = False,
) -> _ImportedRemoteResult:
    image_names = job.client.gallery_image_names(completed_item)
    video_names = job.client.gallery_video_names(completed_item)
    image_source_ids, video_source_ids = _remote_media_source_ids(completed_item)

    if settings.import_to_gallery and not image_names and not video_names:
        raise RemoteInvokeError(
            f"Remote item {job.item_id} produced no Gallery image or video output"
        )

    image_dtos: list[Any] = []
    video_dtos: list[Any] = []
    final_preview: Image.Image | None = None

    if settings.import_to_gallery:
        image_dtos = _import_named_remote_image_dtos(
            job.client,
            image_names,
            import_meta,
            image_source_ids,
        )
        video_dtos = _import_named_remote_video_dtos(
            job.client,
            video_names,
            import_meta,
            video_source_ids,
        )
        if download_final_preview and image_dtos and image_names:
            try:
                final_preview = job.client.download_image(image_names[-1])
            except Exception:
                pass

    return _ImportedRemoteResult(
        image_names=image_names,
        video_names=video_names,
        image_dtos=image_dtos,
        video_dtos=video_dtos,
        final_preview=final_preview,
    )


def _cleanup_remote_job(
    *,
    job: _RemoteJob,
    settings: _RemoteWorkflowSettings,
    logger: Any,
    result: _ImportedRemoteResult | None = None,
    delete_queue_item: bool = True,
) -> None:
    if not settings.cleanup_remote_when_done:
        return

    image_names = list(job.input_image_names)
    video_names = list(job.input_video_names)

    if result is not None and settings.import_to_gallery:
        if result.image_dtos:
            image_names.extend(result.image_names)
        if result.video_dtos:
            video_names.extend(result.video_names)

    image_names = list(dict.fromkeys(image_names))
    video_names = list(dict.fromkeys(video_names))

    _cleanup_remote_artifacts(
        client=job.client,
        remote=job.remote,
        remote_item_id=job.item_id,
        remote_image_names=image_names,
        remote_video_names=video_names,
        delete_remote_images=bool(image_names),
        delete_remote_videos=bool(video_names),
        delete_remote_queue_item=delete_queue_item,
        logger=logger,
    )


def _find_media_references(
    value: Any,
    image_names: set[str],
    video_names: set[str],
) -> None:
    if isinstance(value, list):
        for item in value:
            _find_media_references(item, image_names, video_names)
        return
    if not isinstance(value, dict):
        return

    image_name = value.get("image_name")
    video_name = value.get("video_name")
    if isinstance(image_name, str) and image_name:
        image_names.add(image_name)
    if isinstance(video_name, str) and video_name:
        video_names.add(video_name)

    for child in value.values():
        _find_media_references(child, image_names, video_names)


def _graph_media_references(graph: dict[str, Any]) -> tuple[list[str], list[str]]:
    image_names: set[str] = set()
    video_names: set[str] = set()
    nodes = graph.get("nodes")
    if isinstance(nodes, dict):
        for node in nodes.values():
            _find_media_references(node, image_names, video_names)
    return sorted(image_names), sorted(video_names)


def _remap_graph_media_names(
    value: Any,
    field_name: str,
    mapped: dict[str, str],
) -> int:
    """Rewrite nested media-field references without touching unrelated strings."""
    changed = 0
    if isinstance(value, list):
        for entry in value:
            changed += _remap_graph_media_names(entry, field_name, mapped)
    elif isinstance(value, dict):
        original = value.get(field_name)
        if isinstance(original, str) and original in mapped:
            value[field_name] = mapped[original]
            changed += 1
        for entry in value.values():
            changed += _remap_graph_media_names(entry, field_name, mapped)
    return changed


def _transfer_source_media_to_remote(
    *,
    context: InvocationContext,
    graph: dict[str, Any],
    local_names: list[str],
    remote_name: str,
    uploaded_names: list[str],
    field_name: str,
    media_label: str,
    load_local: Callable[[str], Any],
    upload_remote: Callable[[Any], str],
) -> None:
    """Copy each distinct local media input once and remap every matching field."""
    mapped: dict[str, str] = {}

    for local_name in local_names:
        try:
            local_value = load_local(local_name)
        except Exception as exc:
            raise RemoteInvokeError(
                f"Remote Workflow [{remote_name}]: cannot read primary source "
                f"{media_label} '{local_name}': {exc}"
            ) from exc

        try:
            remote_media_name = upload_remote(local_value)
        except Exception as exc:
            raise RemoteInvokeError(
                f"Remote Workflow [{remote_name}]: could not transfer source "
                f"{media_label} '{local_name}': {exc}"
            ) from exc

        mapped[local_name] = remote_media_name
        uploaded_names.append(remote_media_name)
        context.logger.debug(
            f"Remote Workflow [{remote_name}]: transferred input {media_label} "
            f"'{local_name}' -> '{remote_media_name}'"
        )

    changed = _remap_graph_media_names(graph.get("nodes", {}), field_name, mapped)
    context.logger.info(
        f"Remote Workflow [{remote_name}]: remapped {changed} {media_label} field(s) "
        f"from {len(mapped)} transferred source {media_label}(s)"
    )


def _client_config(remote: ConfiguredRemote, *, probe: bool = False) -> RemoteConfig:
    return RemoteConfig(
        base_url=remote.url,
        email=remote.email,
        password=remote.password,
        timeout_seconds=remote.probe_timeout_seconds if probe else remote.timeout_seconds,
    )


def _probe_remote(remote: ConfiguredRemote) -> None:
    """Fast reachability/auth probe against the stock queue API."""
    probe_client = RemoteInvokeClient(_client_config(remote, probe=True))
    probe_client.get_current_item(queue_id=remote.queue_id)


def _available_remote_candidates(
    candidates: list[ConfiguredRemote],
) -> tuple[list[ConfiguredRemote], list[tuple[ConfiguredRemote, Exception]]]:
    """Probe only the remotes allowed by the current workflow selection."""
    available: list[ConfiguredRemote] = []
    unavailable: list[tuple[ConfiguredRemote, Exception]] = []

    for remote in candidates:
        try:
            _probe_remote(remote)
        except Exception as exc:
            unavailable.append((remote, exc))
        else:
            available.append(remote)

    return available, unavailable


def _dispatch_one(
    remote: ConfiguredRemote,
    base_graph: dict[str, Any],
    logger: Any,
    *,
    services: Any,
    settings: _RemoteWorkflowSettings,
    cancelled: Callable[[], bool] | None = None,
    preview_context: InvocationContext | None = None,
    source_context: InvocationContext | None = None,
) -> _RemoteJob:
    _probe_remote(remote)

    client = RemoteInvokeClient(_client_config(remote))
    graph = deepcopy(base_graph)

    missing_model_handler = None
    if settings.install_missing_models:
        def missing_model_handler(identifier: dict[str, Any]) -> None:
            install_missing_model(
                services=services,
                client=client,
                remote_name=remote.name,
                identifier=identifier,
                timeout_seconds=settings.timeout_seconds,
                poll_interval_seconds=settings.poll_interval_seconds,
                logger=logger,
                cancelled=cancelled,
                use_primary_hf_token=settings.use_primary_hf_token,
            )

    remaps = client.remap_model_identifiers(
        graph,
        missing_model_handler=missing_model_handler,
    )
    for message in remaps:
        logger.info(f"Remote Workflow [{remote.name}] model remap: {message}")

    uploaded_image_names: list[str] = []
    uploaded_video_names: list[str] = []
    preview_socket: _RemoteProgressSocket | None = None

    try:
        image_names, video_names = _graph_media_references(graph)
        if image_names or video_names:
            if source_context is None:
                raise RemoteInvokeError(
                    f"Remote Workflow [{remote.name}]: cannot transfer source media without a local invocation context"
                )
            if image_names:
                _transfer_source_media_to_remote(
                    context=source_context,
                    graph=graph,
                    local_names=image_names,
                    remote_name=remote.name,
                    uploaded_names=uploaded_image_names,
                    field_name="image_name",
                    media_label="image",
                    load_local=source_context.images.get_pil,
                    upload_remote=client.upload_input_image,
                )
            if video_names:
                _transfer_source_media_to_remote(
                    context=source_context,
                    graph=graph,
                    local_names=video_names,
                    remote_name=remote.name,
                    uploaded_names=uploaded_video_names,
                    field_name="video_name",
                    media_label="video",
                    load_local=source_context.videos.get_path,
                    upload_remote=client.upload_input_video,
                )

        if preview_context is not None:
            preview_socket = _RemoteProgressSocket(client, remote.queue_id, preview_context)
            preview_socket.connect()

        item_id = client.enqueue_graph(graph, queue_id=remote.queue_id)
    except Exception:
        if preview_socket is not None:
            preview_socket.close()
        if settings.cleanup_remote_when_done and (uploaded_image_names or uploaded_video_names):
            _cleanup_remote_artifacts(
                client=client,
                remote=remote,
                remote_item_id=0,
                remote_image_names=uploaded_image_names,
                remote_video_names=uploaded_video_names,
                delete_remote_images=bool(uploaded_image_names),
                delete_remote_videos=bool(uploaded_video_names),
                delete_remote_queue_item=False,
                logger=logger,
            )
        raise

    logger.info(
        f"Remote Workflow [{remote.name}]: queued item {item_id} on {remote.url}"
    )
    return _RemoteJob(
        remote=remote,
        client=client,
        item_id=item_id,
        preview_socket=preview_socket,
        input_image_names=uploaded_image_names,
        input_video_names=uploaded_video_names,
    )


def _candidate_remotes(
    remotes: list[ConfiguredRemote],
    workers: str,
    specific_remote: str,
) -> list[ConfiguredRemote]:
    """Return enabled remotes allowed by this workflow's worker selection."""
    active = enabled_remotes(remotes)

    if workers == "All Remotes":
        return active

    if workers == "Specific":
        match = find_remote(remotes, specific_remote)
        if match is None or not match.enabled:
            return []
        return [match]

    return []

def _remote_worker_slot_key(remote: ConfiguredRemote) -> tuple[str, str]:
    return (remote.url.rstrip("/").casefold(), remote.queue_id)


def _remote_worker_slot_lock(remote: ConfiguredRemote) -> threading.Lock:
    # A physical remote is shared across users/coordinators. Key by endpoint +
    # remote queue so two users cannot accidentally dispatch concurrent local
    # jobs through separate coordinator instances to the same worker.
    key = _remote_worker_slot_key(remote)
    with _REMOTE_WORKER_SLOT_LOCKS_GUARD:
        lock = _REMOTE_WORKER_SLOT_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _REMOTE_WORKER_SLOT_LOCKS[key] = lock
        return lock


def _register_remote_only_handoff_waiter(candidates: list[ConfiguredRemote]) -> None:
    with _REMOTE_ONLY_HANDOFF_WAITERS_LOCK:
        for remote in candidates:
            key = _remote_worker_slot_key(remote)
            _REMOTE_ONLY_HANDOFF_WAITERS[key] = _REMOTE_ONLY_HANDOFF_WAITERS.get(key, 0) + 1


def _unregister_remote_only_handoff_waiter(candidates: list[ConfiguredRemote]) -> None:
    with _REMOTE_ONLY_HANDOFF_WAITERS_LOCK:
        for remote in candidates:
            key = _remote_worker_slot_key(remote)
            count = _REMOTE_ONLY_HANDOFF_WAITERS.get(key, 0)
            if count <= 1:
                _REMOTE_ONLY_HANDOFF_WAITERS.pop(key, None)
            else:
                _REMOTE_ONLY_HANDOFF_WAITERS[key] = count - 1


def _remote_has_local_handoff_waiter(remote: ConfiguredRemote) -> bool:
    key = _remote_worker_slot_key(remote)
    with _REMOTE_ONLY_HANDOFF_WAITERS_LOCK:
        return _REMOTE_ONLY_HANDOFF_WAITERS.get(key, 0) > 0


def _background_try_acquire_remote_worker_slot(
    remote: ConfiguredRemote,
    lock: threading.Lock,
) -> bool:
    """Acquire a remote slot only when no Local-held Remote Only item is waiting.

    The waiter check and non-blocking acquire are serialized by the waiter lock,
    so a background lane cannot jump ahead after a Local handoff has registered.
    """

    key = _remote_worker_slot_key(remote)
    with _REMOTE_ONLY_HANDOFF_WAITERS_LOCK:
        if _REMOTE_ONLY_HANDOFF_WAITERS.get(key, 0) > 0:
            return False
        return lock.acquire(blocking=False)


def _acquire_remote_worker_slot(
    *,
    candidates: list[ConfiguredRemote],
    context: InvocationContext,
    attempted: set[str] | None = None,
    prioritize_local_handoff: bool = False,
) -> tuple[ConfiguredRemote, threading.Lock]:
    attempted = attempted or set()
    registered: list[ConfiguredRemote] = []

    if prioritize_local_handoff:
        registered = [
            remote
            for remote in candidates
            if remote.name.casefold() not in attempted
        ]
        _register_remote_only_handoff_waiter(registered)

    try:
        while True:
            if context.util.is_canceled():
                raise CanceledException

            usable = [
                remote
                for remote in candidates
                if remote.name.casefold() not in attempted
            ]
            if not usable:
                raise RemoteInvokeError("No allowed remote worker remains for this item")

            for remote in usable:
                lock = _remote_worker_slot_lock(remote)
                if lock.acquire(blocking=False):
                    return remote, lock

            time.sleep(0.05)
    finally:
        if registered:
            _unregister_remote_only_handoff_waiter(registered)

def _remote_queue_item_status(services: Any, item_id: int) -> str:
    try:
        return str(services.session_queue.get_queue_item(item_id).status or "").lower()
    except Exception:
        return ""


def _is_remote_oom_error(errors: Any) -> bool:
    """Return True only for clear remote device-memory exhaustion failures."""
    try:
        detail = json.dumps(errors, ensure_ascii=False).casefold()
    except Exception:
        detail = str(errors).casefold()
    return (
        "outofmemoryerror" in detail
        or "cuda out of memory" in detail
        or "would exceed allowed memory" in detail
    )


def _fail_remote_oom(
    services: Any,
    queue_item: Any,
    remote: ConfiguredRemote,
    errors: Any,
    logger: Any,
) -> None:
    """Make a deterministic remote OOM terminal instead of requeueing it."""
    item_id = int(queue_item.item_id)
    if _remote_queue_item_status(services, item_id) not in {"in_progress", "waiting", "pending"}:
        return

    try:
        detail = json.dumps(errors, ensure_ascii=False)[:4000]
    except Exception:
        detail = str(errors)[:4000]

    services.session_queue.fail_queue_item(
        item_id=item_id,
        error_type="OutOfMemoryError",
        error_message=f"{remote.name} ran out of memory: {detail}",
        error_traceback="",
    )
    logger.warning(
        f"Remote Workflow [{remote.name}]: remote item failed with an out-of-memory error; "
        f"marked local item {item_id} failed instead of requeueing it"
    )


def _remote_queue_settings_from_queue_item(queue_item: Any) -> _RemoteWorkflowSettings | None:
    """Read queue-worker policy from an enabled Remote Workflow invocation."""

    try:
        _, node = _remote_workflow_invocation(queue_item)
    except RemoteInvokeError:
        return None

    if getattr(node, "enabled", True) is False:
        return None

    mode = str(getattr(node, "mode", "") or "")
    workers = str(getattr(node, "dispatch", "All Remotes") or "All Remotes")
    if mode not in {"Distributed", "Remote Only"} or workers not in {"All Remotes", "Specific"}:
        return None

    try:
        poll_interval_seconds = float(getattr(node, "poll_interval_seconds", 0.35))
    except (TypeError, ValueError):
        poll_interval_seconds = 0.35

    try:
        timeout_seconds = int(getattr(node, "timeout_seconds", 1800))
    except (TypeError, ValueError):
        timeout_seconds = 1800

    return _RemoteWorkflowSettings(
        mode=mode,
        workers=workers,
        specific_remote=str(getattr(node, "specific_remote", "") or "").strip(),
        import_to_gallery=bool(getattr(node, "import_to_gallery", True)),
        cleanup_remote_when_done=bool(getattr(node, "cleanup_remote_when_done", False)),
        cancel_remote_with_local=bool(getattr(node, "cancel_remote_with_local", True)),
        install_missing_models=bool(getattr(node, "install_missing_models", False)),
        use_primary_hf_token=bool(getattr(node, "use_primary_hf_token", False)),
        poll_interval_seconds=max(0.1, min(10.0, poll_interval_seconds)),
        timeout_seconds=max(10, min(86400, timeout_seconds)),
    )

def _remote_queue_item_ids(
    services: Any,
    *,
    queue_id: str,
    user_id: str,
    statuses: tuple[str, ...],
) -> list[int]:
    if not statuses:
        return []

    db = getattr(services.session_queue, "_db", None)
    if db is None:
        raise RuntimeError("Remote worker pool requires InvokeAI's SQLite queue internals (_db)")

    placeholders = ", ".join("?" for _ in statuses)
    with db.transaction() as cursor:
        cursor.execute(
            f"""--sql
            SELECT item_id
            FROM session_queue
            WHERE queue_id = ?
              AND user_id = ?
              AND status IN ({placeholders})
              AND parent_item_id IS NULL
            ORDER BY priority DESC, item_id ASC
            """,
            (queue_id, user_id, *statuses),
        )
        return [int(row[0]) for row in cursor.fetchall()]


def _has_in_progress_remote_queue_item(
    services: Any,
    *,
    queue_id: str,
    user_id: str,
) -> bool:
    queue_service = services.session_queue
    for item_id in _remote_queue_item_ids(
        services,
        queue_id=queue_id,
        user_id=user_id,
        statuses=("in_progress",),
    ):
        try:
            queue_item = queue_service.get_queue_item(item_id)
        except Exception:
            continue
        if _remote_queue_settings_from_queue_item(queue_item) is not None:
            return True
    return False

def _requeue_remote_queue_item(
    services: Any,
    item_id: int,
    logger: Any,
    reason: str,
    *,
    mode: str,
) -> None:
    if _remote_queue_item_status(services, item_id) != "in_progress":
        return

    target_status = "waiting" if mode == "Remote Only" else "pending"
    queue_service = services.session_queue
    set_status = getattr(queue_service, "_set_queue_item_status", None)
    if set_status is None:
        raise RuntimeError("Remote worker pool requires _set_queue_item_status()")

    set_status(item_id=item_id, status=target_status)
    logger.warning(
        f"Remote Workflow: returned remote-owned local item {item_id} to "
        f"{target_status}: {reason}"
    )


def _start_remote_queue_progress(
    import_meta: dict[str, Any],
    item_id: int,
    event_invocation: BaseInvocation,
    message: str,
) -> None:
    queue_item = _get_event_queue_item(import_meta, item_id, event_invocation)
    services = import_meta["services"]
    services.events.emit_invocation_started(
        queue_item=queue_item,
        invocation=event_invocation,
    )
    services.events.emit_invocation_progress(
        queue_item=queue_item,
        invocation=event_invocation,
        message=message,
        percentage=None,
        image=None,
    )


def _emit_remote_queue_result(
    import_meta: dict[str, Any],
    queue_item: Any,
    item_id: int,
    event_invocation: BaseInvocation,
    *,
    source_id: str,
    output: ImageOutput | VideoOutput,
) -> None:
    """Persist an imported result under its real remote source node id."""

    services = import_meta["services"]
    event_item = _get_event_queue_item(import_meta, item_id, event_invocation)

    synthetic_id = str(uuid.uuid4())
    synthetic_invocation = event_invocation.model_copy(update={"id": synthetic_id})
    source_id = str(source_id or event_invocation.id)

    previous = queue_item.session.results.get(source_id)
    if previous is not None:
        queue_item.session.results[synthetic_id] = previous
    queue_item.session.results[source_id] = output

    # Synthetic mappings are event-only. Persisting them would make
    # GraphExecutionState reference execution nodes that do not exist.
    event_item.session.prepared_source_mapping[synthetic_id] = source_id
    event_item.session.results[synthetic_id] = output

    services.events.emit_invocation_started(
        queue_item=event_item,
        invocation=synthetic_invocation,
    )
    services.events.emit_invocation_complete(
        queue_item=event_item,
        invocation=synthetic_invocation,
        output=output,
    )


def _persist_remote_results(
    services: Any,
    queue_item: Any,
    remote_name: str,
    logger: Any,
    *,
    phase: str,
) -> None:
    try:
        services.session_queue.save_queue_item_session(
            int(queue_item.item_id),
            queue_item.session,
        )
    except Exception as exc:
        logger.warning(
            f"Remote Workflow [{remote_name}]: could not persist imported result history "
            f"{phase}: {exc}"
        )


class _RemoteQueueCoordinator:
    """Queue-wide remote worker pool for Distributed and Remote Only jobs."""

    def __init__(
        self,
        *,
        remotes: list[ConfiguredRemote],
        starter_context: InvocationContext,
    ) -> None:
        queue_item = starter_context._data.queue_item
        self.remotes = remotes
        self.context = starter_context
        self.services = starter_context._services
        self.logger = starter_context.logger
        self.queue_id = str(queue_item.queue_id)
        self.user_id = str(queue_item.user_id)
        self.key = (self.queue_id, self.user_id)
        self.thread = threading.Thread(
            target=self._run,
            name=f"remote-worker-pool-{self.user_id}",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def _remote_allowed_for_item(
        self,
        remote: ConfiguredRemote,
        settings: _RemoteWorkflowSettings,
    ) -> bool:
        workers = settings.workers

        if workers == "All Remotes":
            return True

        if workers == "Specific":
            return settings.specific_remote.casefold() == remote.name.casefold()

        return False

    def _has_allowed_available_remote(
        self,
        settings: _RemoteWorkflowSettings,
        available_remotes: list[ConfiguredRemote],
    ) -> bool:
        return any(
            self._remote_allowed_for_item(remote, settings)
            for remote in available_remotes
        )

    def _park_pending_remote_only_items(
        self,
        available_remotes: list[ConfiguredRemote],
    ) -> int:
        """Move eligible Remote Only rows to waiting so Local cannot dequeue them."""

        queue_service = self.services.session_queue
        dequeue_lock = getattr(queue_service, "_dequeue_lock", None)
        set_status = getattr(queue_service, "_set_queue_item_status", None)
        if dequeue_lock is None or set_status is None:
            raise RuntimeError(
                "Remote worker pool requires InvokeAI's _dequeue_lock and "
                "_set_queue_item_status()"
            )

        parked = 0
        for item_id in _remote_queue_item_ids(
            self.services,
            queue_id=self.queue_id,
            user_id=self.user_id,
            statuses=("pending",),
        ):
            try:
                candidate = queue_service.get_queue_item(item_id)
            except Exception:
                continue

            settings = _remote_queue_settings_from_queue_item(candidate)
            if settings is None or settings.mode != "Remote Only":
                continue
            if not self._has_allowed_available_remote(settings, available_remotes):
                continue

            with dequeue_lock:
                try:
                    fresh = queue_service.get_queue_item(item_id)
                except Exception:
                    continue

                if str(fresh.status or "").lower() != "pending":
                    continue

                fresh_settings = _remote_queue_settings_from_queue_item(fresh)
                if fresh_settings is None or fresh_settings.mode != "Remote Only":
                    continue
                if not self._has_allowed_available_remote(
                    fresh_settings,
                    available_remotes,
                ):
                    continue

                parked_item = set_status(
                    item_id=item_id,
                    status="waiting",
                    queue_item=fresh,
                )
                if str(parked_item.status or "").lower() == "waiting":
                    parked += 1
                    self.logger.info(
                        f"Remote Workflow: parked Remote Only local item "
                        f"{parked_item.item_id} until a remote worker is free"
                    )

        return parked

    def _restore_parked_remote_only_items(self) -> int:
        """Return any unclaimed parked rows to pending when the pool exits."""

        queue_service = self.services.session_queue
        dequeue_lock = getattr(queue_service, "_dequeue_lock", None)
        set_status = getattr(queue_service, "_set_queue_item_status", None)
        if dequeue_lock is None or set_status is None:
            return 0

        item_ids = _remote_queue_item_ids(
            self.services,
            queue_id=self.queue_id,
            user_id=self.user_id,
            statuses=("waiting",),
        )

        restored = 0
        for item_id in item_ids:
            try:
                candidate = queue_service.get_queue_item(item_id)
            except Exception:
                continue
            settings = _remote_queue_settings_from_queue_item(candidate)
            if settings is None or settings.mode != "Remote Only":
                continue

            with dequeue_lock:
                try:
                    fresh = queue_service.get_queue_item(item_id)
                except Exception:
                    continue
                if str(fresh.status or "").lower() != "waiting":
                    continue
                fresh_settings = _remote_queue_settings_from_queue_item(fresh)
                if fresh_settings is None or fresh_settings.mode != "Remote Only":
                    continue

                restored_item = set_status(
                    item_id=item_id,
                    status="pending",
                    queue_item=fresh,
                )
                if str(restored_item.status or "").lower() == "pending":
                    restored += 1

        if restored:
            self.logger.info(
                f"Remote Workflow: restored {restored} unclaimed Remote Only "
                "item(s) to pending as the worker pool stopped"
            )
        return restored

    def _claim(
        self,
        remote: ConfiguredRemote,
        excluded_item_ids: set[int],
    ) -> tuple[Any, dict[str, Any]] | None:
        queue_service = self.services.session_queue
        dequeue_lock = getattr(queue_service, "_dequeue_lock", None)
        set_status = getattr(queue_service, "_set_queue_item_status", None)
        if dequeue_lock is None or set_status is None:
            raise RuntimeError(
                "Remote worker pool requires InvokeAI's _dequeue_lock and "
                "_set_queue_item_status()"
            )

        for item_id in _remote_queue_item_ids(
            self.services,
            queue_id=self.queue_id,
            user_id=self.user_id,
            statuses=("pending", "waiting"),
        ):
            if item_id in excluded_item_ids:
                continue

            try:
                candidate = queue_service.get_queue_item(item_id)
            except Exception:
                continue

            settings = _remote_queue_settings_from_queue_item(candidate)
            if settings is None or not self._remote_allowed_for_item(remote, settings):
                continue

            with dequeue_lock:
                try:
                    fresh = queue_service.get_queue_item(item_id)
                except Exception:
                    continue

                fresh_status = str(fresh.status or "").lower()
                if fresh_status not in {"pending", "waiting"}:
                    continue

                fresh_settings = _remote_queue_settings_from_queue_item(fresh)
                if fresh_settings is None or not self._remote_allowed_for_item(remote, fresh_settings):
                    continue

                if fresh_settings.mode == "Distributed" and fresh_status != "pending":
                    continue
                if fresh_settings.mode == "Remote Only" and fresh_status not in {"pending", "waiting"}:
                    continue

                claimed = set_status(
                    item_id=item_id,
                    status="in_progress",
                    device=f"remote:{remote.name}",
                    queue_item=fresh,
                )
                if str(claimed.status or "").lower() != "in_progress":
                    continue

                self.logger.info(
                    f"Remote Workflow [{remote.name}]: claimed {fresh_settings.mode} "
                    f"local item {claimed.item_id} from batch {claimed.batch_id}"
                )
                return claimed, fresh_settings

        return None

    def _run_claimed_remote_job(
        self,
        remote: ConfiguredRemote,
        queue_item: Any,
        settings: _RemoteWorkflowSettings,
        job: _RemoteJob,
        import_meta: dict[str, Any],
    ) -> str:
        local_item_id = int(queue_item.item_id)
        mode = str(settings.mode)
        _, event_invocation = _remote_workflow_invocation(queue_item)
        started = time.monotonic()
        last_network_warning = 0.0

        try:
            _start_remote_queue_progress(
                import_meta,
                local_item_id,
                event_invocation,
                f"{remote.name} · {mode}",
            )

            while True:
                local_status = _remote_queue_item_status(self.services, local_item_id)
                if local_status in {"canceled", "cancelled"}:
                    if settings.cancel_remote_with_local:
                        try:
                            job.client.cancel_item(job.item_id, queue_id=remote.queue_id)
                        except Exception as exc:
                            self.logger.warning(
                                f"Remote Workflow [{remote.name}]: could not cancel "
                                f"remote item {job.item_id}: {exc}"
                            )
                        else:
                            _cleanup_remote_job(
                                job=job,
                                settings=settings,
                                logger=self.logger,
                            )
                    if job.preview_socket is not None:
                        job.preview_socket.close()
                    return "canceled"

                if time.monotonic() - started > float(settings.timeout_seconds):
                    try:
                        job.client.cancel_item(job.item_id, queue_id=remote.queue_id)
                    except Exception as exc:
                        self.logger.warning(
                            f"Remote Workflow [{remote.name}]: timed out but could not cancel "
                            f"remote item {job.item_id}; preserving its inputs: {exc}"
                        )
                    else:
                        _cleanup_remote_job(
                            job=job,
                            settings=settings,
                            logger=self.logger,
                        )
                    if job.preview_socket is not None:
                        job.preview_socket.close()
                    return "requeue"

                if job.preview_socket is not None:
                    job.preview_socket.drain_for_queue_item(
                        job.item_id,
                        import_meta,
                        local_item_id,
                        event_invocation,
                        remote.name,
                    )

                try:
                    item = job.client.get_item(job.item_id, queue_id=remote.queue_id)
                except Exception as exc:
                    now = time.monotonic()
                    if now - last_network_warning >= 10:
                        self.logger.warning(
                            f"Remote Workflow [{remote.name}]: temporarily unavailable "
                            f"while waiting for {mode} item {job.item_id}; will retry: {exc}"
                        )
                        last_network_warning = now
                    time.sleep(settings.poll_interval_seconds)
                    continue

                status = str(item.get("status") or "").lower()

                if status == "completed":
                    try:
                        result = _import_completed_remote_result(
                            job,
                            item,
                            import_meta,
                            settings,
                        )
                    except Exception as exc:
                        self.logger.error(
                            f"Remote Workflow [{remote.name}]: {mode} result import failed "
                            f"for local item {local_item_id}: {exc}"
                        )
                        _cleanup_remote_job(
                            job=job,
                            settings=settings,
                            logger=self.logger,
                            delete_queue_item=False,
                        )
                        if job.preview_socket is not None:
                            job.preview_socket.close()
                        return "fail_local"

                    if _remote_queue_item_status(self.services, local_item_id) in {"canceled", "cancelled"}:
                        _cleanup_remote_job(
                            job=job,
                            result=result,
                            settings=settings,
                            logger=self.logger,
                        )
                        if job.preview_socket is not None:
                            job.preview_socket.close()
                        return "canceled"

                    image_source_ids, video_source_ids = _remote_media_source_ids(item)
                    for remote_name, image_dto in zip(result.image_names, result.image_dtos, strict=False):
                        _emit_remote_queue_result(
                            import_meta,
                            queue_item,
                            local_item_id,
                            event_invocation,
                            source_id=image_source_ids.get(remote_name, str(event_invocation.id)),
                            output=ImageOutput.build(image_dto=image_dto),
                        )
                    for remote_name, video_dto in zip(result.video_names, result.video_dtos, strict=False):
                        _emit_remote_queue_result(
                            import_meta,
                            queue_item,
                            local_item_id,
                            event_invocation,
                            source_id=video_source_ids.get(remote_name, str(event_invocation.id)),
                            output=VideoOutput.build(video_dto=video_dto),
                        )

                    _persist_remote_results(
                        self.services,
                        queue_item,
                        remote.name,
                        self.logger,
                        phase="before queue completion",
                    )
                    self.services.session_queue.complete_queue_item(local_item_id)
                    _persist_remote_results(
                        self.services,
                        queue_item,
                        remote.name,
                        self.logger,
                        phase="after queue completion",
                    )
                    _cleanup_remote_job(
                        job=job,
                        result=result,
                        settings=settings,
                        logger=self.logger,
                    )

                    if job.preview_socket is not None:
                        job.preview_socket.close()

                    self.logger.info(
                        f"Remote Workflow [{remote.name}]: {mode} local item "
                        f"{local_item_id} completed from remote item {job.item_id}"
                    )
                    return "completed"

                if status in {"failed", "canceled", "cancelled"}:
                    errors = (
                        (item.get("session") or {}).get("errors")
                        if isinstance(item.get("session"), dict)
                        else None
                    )
                    _cleanup_remote_job(
                        job=job,
                        settings=settings,
                        logger=self.logger,
                        delete_queue_item=status != "failed",
                    )
                    if status == "failed" and _is_remote_oom_error(errors):
                        _fail_remote_oom(
                            self.services,
                            queue_item,
                            remote,
                            errors,
                            self.logger,
                        )
                        if job.preview_socket is not None:
                            job.preview_socket.close()
                        return "terminal_failure"

                    if job.preview_socket is not None:
                        job.preview_socket.close()
                    return "requeue"

                time.sleep(settings.poll_interval_seconds)

        except Exception as exc:
            self.logger.error(
                f"Remote Workflow [{remote.name}]: {mode} monitor failed for "
                f"local item {local_item_id}: {exc}"
            )
            try:
                if job.preview_socket is not None:
                    job.preview_socket.close()
            except Exception:
                pass
            return "requeue"

    def _run_remote(
        self,
        remote: ConfiguredRemote,
        relevant_remotes: list[ConfiguredRemote],
    ) -> None:
        excluded_item_ids: set[int] = set()
        remote_lock = _remote_worker_slot_lock(remote)
        last_unavailable_log = 0.0
        idle_checks = 0

        while True:
            self._park_pending_remote_only_items(relevant_remotes)

            eligible = False
            for item_id in _remote_queue_item_ids(
                self.services,
                queue_id=self.queue_id,
                user_id=self.user_id,
                statuses=("pending", "waiting"),
            ):
                if item_id in excluded_item_ids:
                    continue
                try:
                    candidate = self.services.session_queue.get_queue_item(item_id)
                except Exception:
                    continue
                candidate_settings = _remote_queue_settings_from_queue_item(candidate)
                if candidate_settings is not None and self._remote_allowed_for_item(remote, candidate_settings):
                    eligible = True
                    break

            if not eligible:
                # Keep spare lanes alive while another remote-owned item is active.
                # A newly queued compatible Invoke can then be claimed immediately
                # without needing a new coordinator bootstrap.
                if _has_in_progress_remote_queue_item(
                    self.services,
                    queue_id=self.queue_id,
                    user_id=self.user_id,
                ):
                    idle_checks = 0
                    time.sleep(0.25)
                    continue

                idle_checks += 1
                if idle_checks >= 4:
                    return
                time.sleep(0.25)
                continue

            idle_checks = 0
            # A queue item already dequeued by Local is more urgent than another
            # still-pending row. Let that handoff take the next free remote slot.
            if not _background_try_acquire_remote_worker_slot(remote, remote_lock):
                if _remote_has_local_handoff_waiter(remote):
                    time.sleep(0.02)
                    continue
                time.sleep(0.05)
                continue

            try:
                # Match PR #9642: never claim the real local row until the worker
                # has just proven reachable/authenticated. Keep retrying while
                # eligible work remains so a recovered worker can join the queue.
                try:
                    _probe_remote(remote)
                except Exception as exc:
                    now = time.monotonic()
                    if last_unavailable_log == 0.0 or now - last_unavailable_log >= 30.0:
                        self.logger.warning(
                            f"Remote Workflow [{remote.name}]: unavailable; "
                            f"will retry while eligible work remains: {exc}"
                        )
                        last_unavailable_log = now
                    time.sleep(2.0)
                    continue

                claimed = self._claim(remote, excluded_item_ids)

                if claimed is not None:
                    queue_item, settings = claimed
                    local_item_id = int(queue_item.item_id)

                    try:
                        graph, graph_board_id = _build_remote_graph_from_queue_item(
                            queue_item,
                            self.services,
                            self.logger,
                        )
                        local_board_id = graph_board_id
                        if local_board_id is None:
                            local_board_id = _resolve_gallery_board(queue_item, self.services, self.logger)

                        import_meta = _build_import_metadata(
                            queue_item,
                            self.services,
                            self.logger,
                            local_board_id,
                        )
                        job = _dispatch_one(
                            remote,
                            graph,
                            self.logger,
                            services=self.services,
                            settings=settings,
                            cancelled=lambda: _remote_queue_item_status(
                                self.services, local_item_id
                            ) in {"canceled", "cancelled"},
                            preview_context=self.context,
                            source_context=self.context,
                        )
                    except ModelTransferCancelled:
                        self.logger.info(
                            f"Remote Workflow [{remote.name}]: model installation canceled "
                            f"for local item {local_item_id}"
                        )
                        continue
                    except Exception as exc:
                        excluded_item_ids.add(local_item_id)
                        _requeue_remote_queue_item(
                            self.services,
                            local_item_id,
                            self.logger,
                            f"{remote.name} did not accept it ({exc})",
                            mode=str(settings.mode),
                        )
                        return

                    if job is not None:
                        outcome = self._run_claimed_remote_job(
                            remote,
                            queue_item,
                            settings,
                            job,
                            import_meta,
                        )

                        if outcome == "completed":
                            continue

                        if outcome == "requeue":
                            excluded_item_ids.add(local_item_id)
                            _requeue_remote_queue_item(
                                self.services,
                                local_item_id,
                                self.logger,
                                f"{remote.name} could not complete the remote job",
                                mode=str(settings.mode),
                            )
                            # Stop this lane for now so another remote, or Local
                            # for Distributed, gets a chance to take the item.
                            return

                        if outcome == "fail_local":
                            self.services.session_queue.fail_queue_item(
                                local_item_id,
                                "RemoteInvokeError",
                                f"Remote item {job.item_id} completed but its result could not be imported locally",
                                "",
                            )
                            return

                        if outcome == "terminal_failure":
                            continue

                        if outcome == "canceled":
                            continue
            finally:
                remote_lock.release()

            if _has_in_progress_remote_queue_item(
                self.services,
                queue_id=self.queue_id,
                user_id=self.user_id,
            ):
                time.sleep(0.25)
                continue

            # Re-check pending eligibility before exiting. Another lane may have
            # completed between our claim scan and the in-progress check.
            eligible_pending = False
            for item_id in _remote_queue_item_ids(
                self.services,
                queue_id=self.queue_id,
                user_id=self.user_id,
                statuses=("pending", "waiting"),
            ):
                try:
                    candidate = self.services.session_queue.get_queue_item(item_id)
                except Exception:
                    continue
                settings = _remote_queue_settings_from_queue_item(candidate)
                if settings is not None and self._remote_allowed_for_item(remote, settings):
                    eligible_pending = True
                    break

            if eligible_pending:
                time.sleep(0.05)
                continue
            return

    def run_current_remote_only(
        self,
        *,
        context: InvocationContext,
        remote_graph: dict[str, Any],
        import_meta: dict[str, Any],
        settings: _RemoteWorkflowSettings,
        candidates: list[ConfiguredRemote],
        initial_remote: ConfiguredRemote,
        initial_lock: threading.Lock,
    ) -> Image.Image | None:
        """Run the Local-dequeued Remote Only item on one remote worker.

        Local already owns this queue row, so the invocation stays alive until the
        remote finishes, then the caller marks the remaining local graph skipped.
        Remote worker slot locks keep one active local job per configured remote.
        """

        attempted: set[str] = set()
        last_error: Exception | None = None
        remote = initial_remote
        remote_lock = initial_lock

        while len(attempted) < len(candidates):
            attempted.add(remote.name.casefold())

            try:
                try:
                    job = _dispatch_one(
                        remote,
                        remote_graph,
                        context.logger,
                        services=context._services,
                        settings=settings,
                        cancelled=context.util.is_canceled,
                        preview_context=context,
                        source_context=context,
                    )
                except ModelTransferCancelled as exc:
                    raise CanceledException from exc
                except Exception as exc:
                    last_error = exc
                    context.logger.warning(
                        f"Remote Workflow [{remote.name}]: could not accept current "
                        f"Remote Only item; trying another allowed remote: {exc}"
                    )
                    job = None

                if job is not None:
                    started = time.monotonic()
                    last_network_warning = 0.0
                    final_preview: Image.Image | None = None

                    try:
                        while True:
                            if context.util.is_canceled():
                                if settings.cancel_remote_with_local:
                                    try:
                                        job.client.cancel_item(job.item_id, queue_id=remote.queue_id)
                                    except Exception as exc:
                                        context.logger.warning(
                                            f"Remote Workflow [{remote.name}]: could not cancel "
                                            f"item {job.item_id}: {exc}"
                                        )
                                    else:
                                        try:
                                            _cleanup_remote_job(
                                                job=job,
                                                settings=settings,
                                                logger=context.logger,
                                            )
                                        except Exception:
                                            pass
                                raise CanceledException

                            if time.monotonic() - started > float(settings.timeout_seconds):
                                try:
                                    job.client.cancel_item(job.item_id, queue_id=remote.queue_id)
                                except Exception as exc:
                                    context.logger.warning(
                                        f"Remote Workflow [{remote.name}]: timed out but could not cancel "
                                        f"remote item {job.item_id}; preserving its inputs: {exc}"
                                    )
                                else:
                                    _cleanup_remote_job(
                                        job=job,
                                        settings=settings,
                                        logger=context.logger,
                                    )
                                raise RemoteInvokeError(
                                    f"Remote Only item {job.item_id} timed out on {remote.name}"
                                )

                            if job.preview_socket is not None:
                                job.preview_socket.drain_for_item(job.item_id, remote.name)

                            try:
                                item = job.client.get_item(
                                    job.item_id,
                                    queue_id=remote.queue_id,
                                )
                            except Exception as exc:
                                now = time.monotonic()
                                if now - last_network_warning >= 10:
                                    context.logger.warning(
                                        f"Remote Workflow [{remote.name}]: temporarily unavailable while "
                                        f"monitoring item {job.item_id}; will retry: {exc}"
                                    )
                                    last_network_warning = now
                                time.sleep(float(settings.poll_interval_seconds))
                                continue

                            status = str(item.get("status") or "").lower()
                            if status == "completed":
                                result = _import_completed_remote_result(
                                    job,
                                    item,
                                    import_meta,
                                    settings,
                                    download_final_preview=True,
                                )

                                local_queue_item = context._data.queue_item
                                local_item_id = int(local_queue_item.item_id)
                                event_invocation = context._data.invocation
                                image_source_ids, video_source_ids = _remote_media_source_ids(item)
                                for remote_name, image_dto in zip(result.image_names, result.image_dtos, strict=False):
                                    _emit_remote_queue_result(
                                        import_meta,
                                        local_queue_item,
                                        local_item_id,
                                        event_invocation,
                                        source_id=image_source_ids.get(remote_name, str(event_invocation.id)),
                                        output=ImageOutput.build(image_dto=image_dto),
                                    )
                                for remote_name, video_dto in zip(result.video_names, result.video_dtos, strict=False):
                                    _emit_remote_queue_result(
                                        import_meta,
                                        local_queue_item,
                                        local_item_id,
                                        event_invocation,
                                        source_id=video_source_ids.get(remote_name, str(event_invocation.id)),
                                        output=VideoOutput.build(video_dto=video_dto),
                                    )

                                _persist_remote_results(
                                    context._services,
                                    local_queue_item,
                                    remote.name,
                                    context.logger,
                                    phase="during Local Remote Only handoff",
                                )

                                _cleanup_remote_job(
                                    job=job,
                                    result=result,
                                    settings=settings,
                                    logger=context.logger,
                                )

                                context.logger.info(
                                    f"Remote Workflow [{remote.name}]: current Remote Only "
                                    f"item completed from remote item {job.item_id}"
                                )
                                return result.final_preview

                            if status in {"failed", "canceled", "cancelled"}:
                                errors = (
                                    (item.get("session") or {}).get("errors")
                                    if isinstance(item.get("session"), dict)
                                    else None
                                )
                                last_error = RemoteInvokeError(
                                    f"Remote Workflow [{remote.name}]: item {job.item_id} ended "
                                    f"with status '{status}': {json.dumps(errors)[:2000]}"
                                )
                                _cleanup_remote_job(
                                    job=job,
                                    settings=settings,
                                    logger=context.logger,
                                    delete_queue_item=status != "failed",
                                )
                                if status == "failed" and _is_remote_oom_error(errors):
                                    context.logger.warning(
                                        f"Remote Workflow [{remote.name}]: remote item {job.item_id} "
                                        "failed with an out-of-memory error; failing the current local item "
                                        "instead of trying another remote"
                                    )
                                    raise last_error
                                break

                            time.sleep(float(settings.poll_interval_seconds))
                    finally:
                        if job.preview_socket is not None:
                            job.preview_socket.close()
            finally:
                remote_lock.release()

            remaining = [
                candidate
                for candidate in candidates
                if candidate.name.casefold() not in attempted
            ]
            if not remaining:
                break

            remote, remote_lock = _acquire_remote_worker_slot(
                candidates=remaining,
                context=context,
                attempted=attempted,
                prioritize_local_handoff=True,
            )

        if last_error is not None:
            raise RemoteInvokeError(
                f"Remote Only completed with no successful remote worker: {last_error}"
            ) from last_error
        raise RemoteInvokeError("Remote Only completed with no successful remote worker")

    def _run(self) -> None:
        try:
            active_remotes = enabled_remotes(self.remotes)
            if not active_remotes:
                self.logger.warning(
                    "Remote Workflow: no remote is enabled for the current worker pool"
                )
                return

            # Start a lightweight lane for each globally enabled remote. A lane
            # checks queue eligibility before probing, so Specific workflows do
            # not contact unrelated workers. Keeping all lanes available also
            # lets a later compatible Invoke join an already-running pool.
            self._park_pending_remote_only_items(active_remotes)

            workers: list[threading.Thread] = []
            for remote in active_remotes:
                thread = threading.Thread(
                    target=self._run_remote,
                    args=(remote, active_remotes),
                    name=f"remote-worker-pool-{remote.name}",
                    daemon=True,
                )
                workers.append(thread)
                thread.start()

            for thread in workers:
                thread.join()
        finally:
            self._restore_parked_remote_only_items()
            with _REMOTE_QUEUE_COORDINATORS_LOCK:
                if _REMOTE_QUEUE_COORDINATORS.get(self.key) is self:
                    _REMOTE_QUEUE_COORDINATORS.pop(self.key, None)
            self.logger.info(
                f"Remote Workflow: queue-wide worker pool finished for user {self.user_id}"
            )


def _ensure_remote_queue_coordinator(
    remotes: list[ConfiguredRemote],
    context: InvocationContext,
) -> tuple[_RemoteQueueCoordinator, bool]:
    queue_item = context._data.queue_item
    key = (str(queue_item.queue_id), str(queue_item.user_id))

    with _REMOTE_QUEUE_COORDINATORS_LOCK:
        existing = _REMOTE_QUEUE_COORDINATORS.get(key)
        if existing is not None and existing.thread.is_alive():
            return existing, False

        coordinator = _RemoteQueueCoordinator(
            remotes=remotes,
            starter_context=context,
        )
        _REMOTE_QUEUE_COORDINATORS[key] = coordinator
        coordinator.start()
        return coordinator, True
