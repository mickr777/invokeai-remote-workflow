from __future__ import annotations

import mimetypes
import secrets
import socket
import threading
import time
import urllib.parse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from huggingface_hub import get_token as hf_get_token

from .model_transfer_state import (
    another_generation_needs_model,
    register_model_transfer,
    unregister_model_transfer,
)
from .remote_client import RemoteInvokeClient, RemoteInvokeError


class ModelTransferError(RemoteInvokeError):
    pass


class ModelTransferCancelled(ModelTransferError):
    pass


@dataclass(frozen=True)
class LocalModelInfo:
    path: Path
    key: str
    name: str
    hash: str
    source: str
    source_type: str
    repo_variant: str


def _enum_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return "" if raw is None else str(raw)


def resolve_local_model(services: Any, identifier: dict[str, Any]) -> LocalModelInfo:
    key = str(identifier.get("key") or "").strip()
    if not key:
        raise ModelTransferError("Model identifier has no local model key")

    try:
        config = services.model_manager.store.get_model(key)
    except Exception as exc:
        raise ModelTransferError(f"Could not read local model record {key}: {exc}") from exc

    config_path = getattr(config, "path", None)
    if not config_path:
        raise ModelTransferError(
            f"Local model '{getattr(config, 'name', identifier.get('name', key))}' has no filesystem path"
        )

    model_path = Path(str(config_path))
    if not model_path.is_absolute():
        model_path = Path(services.configuration.models_path) / model_path
    model_path = model_path.resolve()

    if not model_path.exists():
        raise ModelTransferError(f"Local model path does not exist: {model_path}")
    if not model_path.is_file() and not model_path.is_dir():
        raise ModelTransferError(f"Local model path is neither a file nor directory: {model_path}")

    model_hash = str(identifier.get("hash") or getattr(config, "hash", "") or "").strip()
    if not model_hash:
        raise ModelTransferError(
            f"Local model '{getattr(config, 'name', model_path.name)}' has no recorded hash; "
            "automatic installation cannot verify the remote copy safely"
        )

    return LocalModelInfo(
        path=model_path,
        key=key,
        name=str(getattr(config, "name", None) or identifier.get("name") or model_path.stem),
        hash=model_hash,
        source=str(getattr(config, "source", "") or "").strip(),
        source_type=_enum_value(getattr(config, "source_type", "")),
        repo_variant=_enum_value(getattr(config, "repo_variant", "")),
    )


def _route_local_ip(remote_url: str) -> str:
    parsed = urllib.parse.urlsplit(remote_url)
    host = parsed.hostname
    if not host:
        raise ModelTransferError(f"Cannot determine remote host from URL: {remote_url}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    try:
        addresses = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_DGRAM)
    except OSError as exc:
        raise ModelTransferError(f"Could not resolve remote host '{host}': {exc}") from exc
    if not addresses:
        raise ModelTransferError(f"Could not resolve an IPv4 route to remote host '{host}'")

    target = addresses[0][4]
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(target)
        local_ip = str(sock.getsockname()[0])
    except OSError as exc:
        raise ModelTransferError(f"Could not determine local LAN address for {remote_url}: {exc}") from exc
    finally:
        sock.close()

    if not local_ip or local_ip.startswith("127.") or local_ip == "0.0.0.0":
        raise ModelTransferError(
            f"Automatic LAN address detection returned '{local_ip}' for {remote_url}"
        )
    return local_ip


class TemporaryModelServer:
    """Serve exactly one local model file on an unguessable temporary URL."""

    def __init__(self, model: LocalModelInfo, remote_url: str) -> None:
        if not model.path.is_file():
            raise ModelTransferError("TemporaryModelServer only supports single-file models")
        self.model = model
        self.remote_url = remote_url
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        token = secrets.token_urlsafe(32)
        self._request_path = f"/{token}/{urllib.parse.quote(model.path.name, safe='')}"
        self.url = ""

    def __enter__(self) -> "TemporaryModelServer":
        file_path = self.model.path
        request_path = self._request_path
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        host = _route_local_ip(self.remote_url)

        class Handler(BaseHTTPRequestHandler):
            server_version = "InvokeAIRemoteWorkflowModelTransfer/0.9"

            def log_message(self, _format: str, *args: Any) -> None:
                return

            def _match(self) -> bool:
                return urllib.parse.urlsplit(self.path).path == request_path

            def _range(self, size: int) -> tuple[int, int] | None:
                header = self.headers.get("Range", "").strip()
                if not header:
                    return None
                if not header.startswith("bytes=") or "," in header:
                    return None

                value = header[6:]
                start_text, _, end_text = value.partition("-")
                try:
                    if start_text:
                        start = int(start_text)
                        end = int(end_text) if end_text else size - 1
                    else:
                        suffix = int(end_text)
                        start = max(0, size - suffix)
                        end = size - 1
                except ValueError:
                    return None

                if start < 0 or start >= size or end < start:
                    return None
                return start, min(end, size - 1)

            def _send_headers(self) -> tuple[int, int] | None:
                if not self._match():
                    self.send_error(404)
                    return None

                size = file_path.stat().st_size
                byte_range = self._range(size)
                if self.headers.get("Range") and byte_range is None:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return None

                if byte_range is None:
                    start, end = 0, size - 1
                    self.send_response(200)
                else:
                    start, end = byte_range
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")

                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(max(0, end - start + 1)))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Disposition", f'attachment; filename="{file_path.name}"')
                self.end_headers()
                return start, end

            def do_HEAD(self) -> None:  # noqa: N802
                self._send_headers()

            def do_GET(self) -> None:  # noqa: N802
                selected = self._send_headers()
                if selected is None:
                    return

                start, end = selected
                remaining = end - start + 1
                with file_path.open("rb") as stream:
                    stream.seek(start)
                    while remaining > 0:
                        chunk = stream.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        try:
                            self.wfile.write(chunk)
                        except (BrokenPipeError, ConnectionResetError):
                            break
                        remaining -= len(chunk)

        server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
        server.daemon_threads = True
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever,
            name="invokeai-remote-workflow-model-transfer",
            daemon=True,
        )
        self._thread.start()

        self.url = f"http://{host}:{server.server_port}{self._request_path}"
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._server = None
        self._thread = None


def _hf_install_source(model: LocalModelInfo) -> str:
    """Preserve the local Diffusers repo variant in stock InvokeAI HF source syntax."""
    source = model.source.strip()
    if model.source_type != "hf_repo_id" or not source:
        return source

    variant = model.repo_variant.strip()
    if not variant:
        return source

    base, separator, subfolder = source.partition("::")
    if ":" not in base:
        base = f"{base}:{variant}"
    return f"{base}{separator}{subfolder}" if separator else base


def _is_huggingface_directory_source(model: LocalModelInfo) -> bool:
    if model.source_type == "hf_repo_id":
        return bool(model.source)

    if model.source_type != "url":
        return False

    try:
        parsed = urllib.parse.urlsplit(model.source)
    except Exception:
        return False

    if parsed.scheme not in {"http", "https"} or parsed.netloc.casefold() != "huggingface.co":
        return False

    parts = [part for part in parsed.path.split("/") if part]
    return len(parts) == 2


def _wait_for_install(
    *,
    client: RemoteInvokeClient,
    job: dict[str, Any],
    model: LocalModelInfo,
    remote_name: str,
    timeout_seconds: int,
    poll_interval_seconds: float,
    logger: Any,
    cancelled: Callable[[], bool] | None,
) -> None:
    try:
        job_id = int(job.get("id"))
    except (TypeError, ValueError) as exc:
        raise ModelTransferError(f"Remote model installer returned no usable job id: {job}") from exc

    status = str(job.get("status") or "").lower()
    last_status = ""
    started = time.monotonic()

    while True:
        if cancelled is not None and cancelled():
            try:
                client.cancel_model_install(job_id)
            except Exception as exc:
                logger.warning(
                    f"Remote Workflow [{remote_name}]: could not cancel model install job {job_id}: {exc}"
                )
            raise ModelTransferCancelled(
                f"Model installation for '{model.name}' was canceled"
            )

        if status != last_status:
            logger.info(
                f"Remote Workflow [{remote_name}]: model install '{model.name}' "
                f"job {job_id} status={status or 'unknown'}"
            )
            last_status = status

        if status == "completed":
            break

        if status in {"error", "failed", "cancelled", "canceled"}:
            detail = job.get("error") or job.get("error_reason") or ""
            raise ModelTransferError(
                f"Remote model install for '{model.name}' ended with status "
                f"'{status}': {str(detail)[:2000]}"
            )

        if status == "paused":
            raise ModelTransferError(
                f"Remote model install for '{model.name}' is paused; resume or remove the install job on {remote_name}"
            )

        if time.monotonic() - started > float(timeout_seconds):
            try:
                client.cancel_model_install(job_id)
            except Exception:
                pass
            raise ModelTransferError(
                f"Remote model install for '{model.name}' timed out after {timeout_seconds}s"
            )

        time.sleep(max(0.1, min(2.0, poll_interval_seconds)))
        job = client.get_model_install_job(job_id)
        status = str(job.get("status") or "").lower()


def install_missing_model(
    *,
    services: Any,
    client: RemoteInvokeClient,
    remote_name: str,
    identifier: dict[str, Any],
    timeout_seconds: int,
    poll_interval_seconds: float,
    logger: Any,
    cancelled: Callable[[], bool] | None = None,
    use_primary_hf_token: bool = False,
) -> None:
    """Install one missing model on a stock InvokeAI remote.

    Concurrent generations targeting the same worker/model share one install.
    Stale stock model-manager rows reported by /api/v2/models/missing are not
    accepted as installed models.
    """

    model = resolve_local_model(services, identifier)

    if client.get_usable_model_by_hash(model.hash) is not None:
        return

    transfer_id, transfer = register_model_transfer(client.config.base_url, model.hash)
    lock_acquired = False

    def request_is_cancelled() -> bool:
        if transfer.cancel_requested.is_set():
            return True
        if cancelled is not None and cancelled():
            transfer.cancel_requested.set()
            return True
        return False

    def shared_install_should_cancel() -> bool:
        if not request_is_cancelled():
            return False
        return not another_generation_needs_model(transfer)

    try:
        while not lock_acquired:
            if request_is_cancelled():
                raise ModelTransferCancelled(
                    f"Model installation for '{model.name}' was canceled"
                )
            lock_acquired = transfer.shared_lock.acquire(timeout=0.25)

        # Another generation may have completed the install while we waited.
        if client.get_usable_model_by_hash(model.hash) is not None:
            if request_is_cancelled():
                raise ModelTransferCancelled(
                    f"Model installation for '{model.name}' was canceled"
                )
            return

        # If this generation was canceled after obtaining the shared lock, keep
        # preparing only when another live generation is waiting on the same install.
        if request_is_cancelled() and not another_generation_needs_model(transfer):
            raise ModelTransferCancelled(
                f"Model installation for '{model.name}' was canceled"
            )

        if model.path.is_file():
            size_gib = model.path.stat().st_size / (1024**3)
            logger.warning(
                f"Remote Workflow [{remote_name}]: required model '{model.name}' is missing; "
                f"transferring {model.path.name} ({size_gib:.2f} GiB) from this InvokeAI host"
            )
            with TemporaryModelServer(model, client.config.base_url) as server:
                logger.info(
                    f"Remote Workflow [{remote_name}]: temporary model transfer endpoint ready"
                )
                job = client.install_model_source(server.url, name=model.name)
                _wait_for_install(
                    client=client,
                    job=job,
                    model=model,
                    remote_name=remote_name,
                    timeout_seconds=timeout_seconds,
                    poll_interval_seconds=poll_interval_seconds,
                    logger=logger,
                    cancelled=shared_install_should_cancel,
                )

        elif _is_huggingface_directory_source(model):
            source_access_token = None
            if use_primary_hf_token:
                source_access_token = hf_get_token()
                if not source_access_token:
                    raise ModelTransferError(
                        "Use Primary HF Token is enabled, but no Hugging Face token is available "
                        "to this primary InvokeAI installation"
                    )
                if urllib.parse.urlsplit(client.config.base_url).scheme.casefold() != "https":
                    logger.warning(
                        f"Remote Workflow [{remote_name}]: forwarding the primary Hugging Face token "
                        "to a remote over HTTP; the token is not encrypted in transit"
                    )

            install_source = _hf_install_source(model)
            logger.warning(
                f"Remote Workflow [{remote_name}]: required directory model '{model.name}' is missing; "
                f"asking the stock remote to install its saved Hugging Face source '{install_source}'"
            )
            job = client.install_model_source(
                install_source,
                name=model.name,
                source_access_token=source_access_token,
            )
            _wait_for_install(
                client=client,
                job=job,
                model=model,
                remote_name=remote_name,
                timeout_seconds=timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
                logger=logger,
                cancelled=shared_install_should_cancel,
            )

        else:
            source_text = model.source or "(none recorded)"
            raise ModelTransferError(
                f"Required directory model '{model.name}' is missing on {remote_name}. "
                f"Its recorded source is '{source_text}' (source_type={model.source_type or 'unknown'}). "
                "Automatic directory installation requires an original Hugging Face repo source; "
                "install this model on the remote manually."
            )

        # A stock worker cannot expose the PR's custom directory-layout endpoint.
        # Require at minimum an exact weight hash AND a root path that stock
        # /api/v2/models/missing does not report as absent.
        for _ in range(20):
            if client.get_usable_model_by_hash(model.hash) is not None:
                kind = "directory weight hash/root path" if model.path.is_dir() else "file hash"
                logger.info(
                    f"Remote Workflow [{remote_name}]: model '{model.name}' installed and verified by {kind}"
                )
                if request_is_cancelled():
                    raise ModelTransferCancelled(
                        f"Model installation for '{model.name}' was canceled"
                    )
                return
            time.sleep(0.25)

        raise ModelTransferError(
            f"Remote reported model install completion for '{model.name}', "
            f"but no usable model with hash {model.hash} could be found afterward"
        )
    finally:
        if lock_acquired:
            transfer.shared_lock.release()
        unregister_model_transfer(transfer_id)

