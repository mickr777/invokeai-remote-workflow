from __future__ import annotations

import json
import secrets
import urllib.error
import urllib.parse
import urllib.request
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

from PIL import Image


class RemoteInvokeError(RuntimeError):
    pass


@dataclass(frozen=True)
class RemoteConfig:
    base_url: str
    email: str = ""
    password: str = ""
    timeout_seconds: float = 30.0


class RemoteInvokeClient:
    def __init__(self, config: RemoteConfig):
        base_url = config.base_url.strip().rstrip("/")
        if not base_url:
            raise RemoteInvokeError("Remote URL is required")
        self.config = RemoteConfig(
            base_url=base_url,
            email=config.email.strip(),
            password=config.password,
            timeout_seconds=config.timeout_seconds,
        )
        self._token = ""
        self._auth_checked = False
        self._multiuser = False

    @property
    def token(self) -> str:
        self.ensure_auth()
        return self._token

    def _headers(self, *, json_body: bool = False, include_auth: bool = True) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if json_body:
            headers["Content-Type"] = "application/json"
        if include_auth and self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _request_raw(
        self,
        method: str,
        path: str,
        *,
        body: Any = None,
        json_body: bool = False,
        include_auth: bool = True,
        extra_headers: dict[str, str] | None = None,
    ) -> bytes:
        url = f"{self.config.base_url}{path}"
        headers = self._headers(json_body=json_body, include_auth=include_auth)
        if extra_headers:
            headers.update(extra_headers)
        request = urllib.request.Request(
            url=url,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.timeout_seconds,
            ) as response:
                return response.read()
        except urllib.error.HTTPError:
            raise
        except urllib.error.URLError as exc:
            raise RemoteInvokeError(f"Could not reach remote InvokeAI at {url}: {exc}") from exc

    def _login(self) -> None:
        if not self.config.email or not self.config.password:
            raise RemoteInvokeError("Remote has multi-user mode enabled. Configure email + password for it in remotes.json.")
        payload = json.dumps(
            {"email": self.config.email, "password": self.config.password, "remember_me": True},
            separators=(",", ":"),
        ).encode("utf-8")
        try:
            raw = self._request_raw(
                "POST",
                "/api/v1/auth/login",
                body=payload,
                json_body=True,
                include_auth=False,
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RemoteInvokeError(f"Remote login failed (HTTP {exc.code}): {detail[:2000]}") from exc
        try:
            data = json.loads(raw.decode("utf-8"))
            token = str(data.get("token") or "") if isinstance(data, dict) else ""
        except Exception as exc:
            raise RemoteInvokeError("Remote login returned invalid JSON") from exc
        if not token:
            raise RemoteInvokeError("Remote login succeeded but returned no token")
        self._token = token

    def ensure_auth(self) -> None:
        if self._auth_checked:
            if self._multiuser and not self._token:
                self._login()
            return
        try:
            raw = self._request_raw("GET", "/api/v1/auth/status", include_auth=False)
            status = json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 405):
                self._multiuser = False
                self._auth_checked = True
                return
            detail = exc.read().decode("utf-8", errors="replace")
            raise RemoteInvokeError(f"Could not read remote auth status (HTTP {exc.code}): {detail[:2000]}") from exc
        except Exception as exc:
            raise RemoteInvokeError(f"Could not read remote auth status: {exc}") from exc

        self._multiuser = bool(isinstance(status, dict) and status.get("multiuser_enabled"))
        self._auth_checked = True
        if self._multiuser and not self._token:
            self._login()

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: Any = None,
        json_body: bool = False,
        extra_headers: dict[str, str] | None = None,
    ) -> bytes:
        self.ensure_auth()
        try:
            return self._request_raw(
                method,
                path,
                body=body,
                json_body=json_body,
                include_auth=True,
                extra_headers=extra_headers,
            )
        except urllib.error.HTTPError as exc:
            if exc.code == 401 and self._multiuser and self.config.email and self.config.password:
                self._token = ""
                self._login()
                try:
                    return self._request_raw(
                        method,
                        path,
                        body=body,
                        json_body=json_body,
                        include_auth=True,
                        extra_headers=extra_headers,
                    )
                except urllib.error.HTTPError as retry_exc:
                    exc = retry_exc
            detail = exc.read().decode("utf-8", errors="replace")
            raise RemoteInvokeError(f"Remote returned HTTP {exc.code} for {path}: {detail[:2000]}") from exc

    def _json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> Any:
        body = None
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        raw = self._request(
            method,
            path,
            body=body,
            json_body=payload is not None,
            extra_headers=extra_headers,
        )
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise RemoteInvokeError(f"Expected JSON from {path}") from exc

    @staticmethod
    def _model_list(data: Any) -> list[dict[str, Any]]:
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            for key in ("models", "items", "data"):
                value = data.get(key)
                if isinstance(value, list):
                    return [x for x in value if isinstance(x, dict)]
        raise RemoteInvokeError("Remote model list response was not understood")

    def list_models(self) -> list[dict[str, Any]]:
        return self._model_list(self._json("GET", "/api/v2/models/"))

    def list_missing_models(self) -> list[dict[str, Any]]:
        """Return stock InvokeAI model records whose root path is missing on disk."""
        try:
            return self._model_list(self._json("GET", "/api/v2/models/missing"))
        except RemoteInvokeError as exc:
            # Keep compatibility with older v7 builds that predate this stock endpoint.
            if "HTTP 404" in str(exc) or "HTTP 405" in str(exc):
                return []
            raise

    def missing_model_keys(self) -> set[str]:
        return {
            str(self._model_payload(model).get("key") or "").strip()
            for model in self.list_missing_models()
            if str(self._model_payload(model).get("key") or "").strip()
        }

    def get_usable_model_by_hash(self, model_hash: str) -> dict[str, Any] | None:
        """Find a same-hash model whose registered root path still exists."""
        wanted = str(model_hash or "").strip()
        if not wanted:
            return None
        missing_keys = self.missing_model_keys()
        for model in self.list_models():
            payload = self._model_payload(model)
            key = str(payload.get("key") or "").strip()
            if key and key in missing_keys:
                continue
            if str(payload.get("hash") or "").strip() == wanted:
                return model
        return None

    def get_model(self, key: str) -> dict[str, Any]:
        data = self._json("GET", f"/api/v2/models/i/{urllib.parse.quote(str(key), safe='')}")
        if not isinstance(data, dict):
            raise RemoteInvokeError("Remote model detail was not an object")
        return data

    def get_model_by_hash(self, model_hash: str) -> dict[str, Any] | None:
        try:
            data = self._json(
                "GET",
                f"/api/v2/models/get_by_hash?hash={urllib.parse.quote(str(model_hash), safe='')}",
            )
        except RemoteInvokeError as exc:
            if "HTTP 404" in str(exc):
                return None
            raise
        return data if isinstance(data, dict) else None

    def install_model_source(
        self,
        source: str,
        *,
        name: str = "",
        source_access_token: str | None = None,
    ) -> dict[str, Any]:
        """Ask a stock InvokeAI remote to install a model source."""
        encoded = urllib.parse.quote(str(source), safe="")
        payload: dict[str, Any] = {}
        if name.strip():
            payload["name"] = name.strip()
        headers = None
        if source_access_token:
            headers = {"X-Model-Source-Access-Token": source_access_token}
        try:
            data = self._json(
                "POST",
                f"/api/v2/models/install?source={encoded}&inplace=false",
                payload,
                extra_headers=headers,
            )
        except RemoteInvokeError as exc:
            if "HTTP 401" in str(exc) or "HTTP 403" in str(exc):
                raise RemoteInvokeError(
                    "Remote model installation requires an administrator account "
                    "when multi-user mode is enabled"
                ) from exc
            raise
        if not isinstance(data, dict):
            raise RemoteInvokeError("Remote model installer returned an unexpected response")
        return data

    def get_model_install_job(self, job_id: int) -> dict[str, Any]:
        data = self._json("GET", f"/api/v2/models/install/{int(job_id)}")
        if not isinstance(data, dict):
            raise RemoteInvokeError("Remote model install job response was not an object")
        return data

    def cancel_model_install(self, job_id: int) -> None:
        self._request("DELETE", f"/api/v2/models/install/{int(job_id)}")

    @staticmethod
    def _model_payload(detail: dict[str, Any]) -> dict[str, Any]:
        nested = detail.get("model")
        return nested if isinstance(nested, dict) else detail

    def remap_model_identifiers(
        self,
        graph: dict[str, Any],
        missing_model_handler: Callable[[dict[str, Any]], None] | None = None,
        model_match_validator: Callable[[dict[str, Any], dict[str, Any]], bool] | None = None,
    ) -> list[str]:
        """Replace local model keys with compatible remote installation keys.

        Stock InvokeAI's /models/missing endpoint is authoritative for stale model
        records whose database row survives after the on-disk model was removed.
        Those records are never considered valid remap targets.
        """

        remote_models = self.list_models()
        missing_keys = self.missing_model_keys()
        details_cache: dict[str, dict[str, Any]] = {}
        messages: list[str] = []
        handled_missing: set[str] = set()

        def payload_of(model: dict[str, Any]) -> dict[str, Any]:
            return self._model_payload(model)

        def detail_for(model: dict[str, Any]) -> dict[str, Any]:
            payload = payload_of(model)
            key = str(payload.get("key") or "")
            if not key:
                return model
            if key not in details_cache:
                try:
                    details_cache[key] = self.get_model(key)
                except Exception:
                    details_cache[key] = model
            return details_cache[key]

        def refresh_models() -> None:
            nonlocal remote_models, missing_keys
            remote_models = self.list_models()
            missing_keys = self.missing_model_keys()
            details_cache.clear()

        def candidate_is_usable(value: dict[str, Any], detail: dict[str, Any]) -> bool:
            payload = payload_of(detail)
            key = str(payload.get("key") or "").strip()
            if key and key in missing_keys:
                return False
            if model_match_validator is None:
                return True
            return bool(model_match_validator(deepcopy(value), payload))

        def resolve(value: dict[str, Any]) -> dict[str, Any] | None:
            local_hash = str(value.get("hash") or "").strip()
            if local_hash:
                # Do not use /get_by_hash here. It returns one record, which can
                # be stale while another valid duplicate with the same weight hash exists.
                for model in remote_models:
                    payload = payload_of(model)
                    if str(payload.get("hash") or "").strip() != local_hash:
                        continue
                    detail = detail_for(model)
                    if candidate_is_usable(value, detail):
                        return detail

            local_name = value.get("name")
            local_base = value.get("base")
            local_type = value.get("type")
            verified: list[dict[str, Any]] = []
            for candidate in remote_models:
                payload = payload_of(candidate)
                if (
                    payload.get("name") != local_name
                    or payload.get("base") != local_base
                    or payload.get("type") != local_type
                ):
                    continue
                detail = detail_for(candidate)
                detail_payload = payload_of(detail)
                remote_hash = str(detail_payload.get("hash") or "").strip()
                if local_hash and remote_hash and local_hash != remote_hash:
                    continue
                if not candidate_is_usable(value, detail):
                    continue
                verified.append(detail)

            if len(verified) == 1:
                return verified[0]
            if len(verified) > 1:
                raise RemoteInvokeError(
                    f"Remote has multiple usable matching models named '{local_name}'"
                )
            return None

        def remap(value: Any) -> Any:
            if isinstance(value, list):
                return [remap(x) for x in value]
            if not isinstance(value, dict):
                return value

            if all(k in value for k in ("key", "name", "base", "type")):
                local_name = str(value.get("name") or value.get("key") or "model")
                local_hash = str(value.get("hash") or "").strip()
                resolved = resolve(value)

                missing_identity = local_hash or str(value.get("key") or local_name)
                if (
                    resolved is None
                    and missing_model_handler is not None
                    and missing_identity not in handled_missing
                ):
                    handled_missing.add(missing_identity)
                    missing_model_handler(deepcopy(value))
                    refresh_models()
                    resolved = resolve(value)

                if resolved is None:
                    raise RemoteInvokeError(
                        f"Remote does not have a usable required model '{local_name}' "
                        f"(base={value.get('base')}, type={value.get('type')}, "
                        f"hash={value.get('hash') or 'unknown'})"
                    )

                payload = payload_of(resolved)
                mapped = deepcopy(value)
                old_key = str(mapped.get("key") or "")
                for field in ("key", "hash", "name", "base", "type", "submodel_type"):
                    if payload.get(field) is not None:
                        mapped[field] = payload[field]

                remote_hash = str(mapped.get("hash") or "").strip()
                if local_hash and remote_hash and local_hash != remote_hash:
                    raise RemoteInvokeError(
                        f"Remote model '{local_name}' resolved to hash {remote_hash}, expected {local_hash}"
                    )

                new_key = str(mapped.get("key") or "")
                if old_key != new_key:
                    messages.append(f"{local_name}: {old_key} -> {new_key}")
                return mapped

            return {k: remap(v) for k, v in value.items()}

        graph_nodes = graph.get("nodes")
        if not isinstance(graph_nodes, dict):
            raise RemoteInvokeError("Graph has no nodes object")
        for node_id, node in list(graph_nodes.items()):
            graph_nodes[node_id] = remap(node)
        return messages

    def get_current_item(self, queue_id: str = "default") -> dict[str, Any] | None:
        data = self._json(
            "GET",
            f"/api/v1/queue/{urllib.parse.quote(queue_id, safe='')}/current",
        )
        if data is None:
            return None
        if not isinstance(data, dict):
            raise RemoteInvokeError("Remote current queue item response was not an object or null")
        return data

    def enqueue_graph(self, graph: dict[str, Any], queue_id: str = "default") -> int:
        payload = {
            "batch": {
                "graph": graph,
                "runs": 1,
                "origin": "invokeai-remote-workflow",
            }
        }
        data = self._json(
            "POST",
            f"/api/v1/queue/{urllib.parse.quote(queue_id, safe='')}/enqueue_batch",
            payload,
        )
        if not isinstance(data, dict) or not isinstance(data.get("item_ids"), list) or not data["item_ids"]:
            raise RemoteInvokeError(f"Remote enqueue returned no item id: {data}")
        return int(data["item_ids"][0])

    def get_item(self, item_id: int, queue_id: str = "default") -> dict[str, Any]:
        data = self._json(
            "GET", f"/api/v1/queue/{urllib.parse.quote(queue_id, safe='')}/i/{int(item_id)}"
        )
        if not isinstance(data, dict):
            raise RemoteInvokeError("Remote queue item response was not an object")
        return data

    def cancel_item(self, item_id: int, queue_id: str = "default") -> None:
        self._json(
            "PUT",
            f"/api/v1/queue/{urllib.parse.quote(queue_id, safe='')}/i/{int(item_id)}/cancel",
        )

    def _delete_ignoring_404(self, path: str) -> None:
        try:
            self._request("DELETE", path)
        except RemoteInvokeError as exc:
            if "HTTP 404" not in str(exc):
                raise

    def delete_item(self, item_id: int, queue_id: str = "default") -> None:
        self._delete_ignoring_404(
            f"/api/v1/queue/{urllib.parse.quote(queue_id, safe='')}/i/{int(item_id)}"
        )

    def delete_image(self, image_name: str) -> None:
        self._delete_ignoring_404(
            f"/api/v1/images/i/{urllib.parse.quote(image_name, safe='')}"
        )

    @staticmethod
    def extract_image_names(item: dict[str, Any]) -> list[str]:
        session = item.get("session")
        results = session.get("results") if isinstance(session, dict) else None
        if not isinstance(results, dict):
            return []
        names: list[str] = []
        seen: set[str] = set()
        for result in results.values():
            if not isinstance(result, dict):
                continue
            candidates: list[Any] = []
            if isinstance(result.get("image"), dict):
                candidates.append(result["image"])
            if isinstance(result.get("images"), list):
                candidates.extend(result["images"])
            for entry in candidates:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("image_name") or "")
                if name and name not in seen:
                    seen.add(name)
                    names.append(name)
        return names

    def get_image_dto(self, image_name: str) -> dict[str, Any]:
        data = self._json("GET", f"/api/v1/images/i/{urllib.parse.quote(image_name, safe='')}")
        if not isinstance(data, dict):
            raise RemoteInvokeError(f"Remote image '{image_name}' DTO was not an object")
        return data

    @staticmethod
    def _gallery_names(
        names: list[str],
        get_dto: Callable[[str], dict[str, Any]],
    ) -> list[str]:
        return [name for name in names if get_dto(name).get("is_intermediate") is False]

    def gallery_image_names(self, item: dict[str, Any]) -> list[str]:
        return self._gallery_names(self.extract_image_names(item), self.get_image_dto)

    def get_image_metadata(self, image_name: str) -> str | None:
        data = self._json(
            "GET",
            f"/api/v1/images/i/{urllib.parse.quote(image_name, safe='')}/metadata",
        )
        if data is None:
            return None
        if not isinstance(data, dict):
            raise RemoteInvokeError(f"Remote image '{image_name}' metadata was not an object")
        return json.dumps(data, separators=(",", ":"), ensure_ascii=False)

    def download_image(self, image_name: str) -> Image.Image:
        raw = self._request("GET", f"/api/v1/images/i/{urllib.parse.quote(image_name, safe='')}/full")
        try:
            with Image.open(BytesIO(raw)) as image:
                image.load()
                return image.copy()
        except Exception as exc:
            raise RemoteInvokeError(f"Could not decode remote image '{image_name}'") from exc

    def upload_input_image(self, image: Image.Image) -> str:
        """Upload a local source image to the remote as an intermediate input."""
        png = BytesIO()
        try:
            image.save(png, format="PNG")
        except Exception as exc:
            raise RemoteInvokeError(f"Could not encode source image as PNG: {exc}") from exc

        boundary = f"irw-{secrets.token_hex(16)}"
        prefix = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="remote-input.png"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode("utf-8")
        body = prefix + png.getvalue() + f"\r\n--{boundary}--\r\n".encode("utf-8")
        query = urllib.parse.urlencode({"image_category": "user", "is_intermediate": "true"})
        raw = self._request(
            "POST",
            f"/api/v1/images/upload?{query}",
            body=body,
            extra_headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            record = json.loads(raw.decode("utf-8"))
        except (UnicodeError, ValueError) as exc:
            raise RemoteInvokeError("Remote image upload did not return valid JSON") from exc
        remote_name = record.get("image_name") if isinstance(record, dict) else None
        if not isinstance(remote_name, str) or not remote_name:
            raise RemoteInvokeError("Remote image upload did not return image_name")
        return remote_name

    @staticmethod
    def extract_video_names(item: dict[str, Any]) -> list[str]:
        session = item.get("session")
        results = session.get("results") if isinstance(session, dict) else None
        if not isinstance(results, dict):
            return []

        names: list[str] = []
        seen: set[str] = set()
        for result in results.values():
            if not isinstance(result, dict):
                continue

            candidates: list[Any] = [result.get("video")]
            videos = result.get("videos")
            if isinstance(videos, list):
                candidates.extend(videos)
            candidates.append(result)

            for entry in candidates:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("video_name") or "")
                if name and name not in seen:
                    seen.add(name)
                    names.append(name)
        return names

    def upload_input_video(self, video_path: Path) -> str:
        """Stream a local source video to the remote as an intermediate input."""
        path = Path(video_path)
        try:
            file_size = path.stat().st_size
        except OSError as exc:
            raise RemoteInvokeError(f"Could not read source video '{path}': {exc}") from exc

        boundary = f"irw-{secrets.token_hex(16)}"
        prefix = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="remote-input.mp4"\r\n'
            "Content-Type: video/mp4\r\n\r\n"
        ).encode("utf-8")
        suffix = f"\r\n--{boundary}--\r\n".encode("utf-8")

        class MultipartVideoBody:
            def __iter__(self):
                yield prefix
                try:
                    with path.open("rb") as source:
                        while chunk := source.read(1024 * 1024):
                            yield chunk
                except OSError as exc:
                    raise RemoteInvokeError(f"Could not read source video '{path}': {exc}") from exc
                yield suffix

        query = urllib.parse.urlencode({"video_category": "user", "is_intermediate": "true"})
        raw = self._request(
            "POST",
            f"/api/v1/videos/upload?{query}",
            body=MultipartVideoBody(),
            extra_headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(prefix) + file_size + len(suffix)),
            },
        )
        try:
            record = json.loads(raw.decode("utf-8"))
        except (UnicodeError, ValueError) as exc:
            raise RemoteInvokeError("Remote video upload did not return valid JSON") from exc
        remote_name = record.get("video_name") if isinstance(record, dict) else None
        if not isinstance(remote_name, str) or not remote_name:
            raise RemoteInvokeError("Remote video upload did not return video_name")
        return remote_name

    def get_video_dto(self, video_name: str) -> dict[str, Any]:
        data = self._json("GET", f"/api/v1/videos/i/{urllib.parse.quote(video_name, safe='')}")
        if not isinstance(data, dict):
            raise RemoteInvokeError(f"Remote video '{video_name}' DTO was not an object")
        return data

    def get_video_metadata(self, video_name: str) -> str | None:
        data = self._json(
            "GET",
            f"/api/v1/videos/i/{urllib.parse.quote(video_name, safe='')}/metadata",
        )
        if data is None:
            return None
        if not isinstance(data, dict):
            raise RemoteInvokeError(f"Remote video '{video_name}' metadata was not an object")
        return json.dumps(data, separators=(",", ":"), ensure_ascii=False)

    def gallery_video_names(self, item: dict[str, Any]) -> list[str]:
        return self._gallery_names(self.extract_video_names(item), self.get_video_dto)

    def download_video(self, video_name: str) -> bytes:
        return self._request(
            "GET",
            f"/api/v1/videos/i/{urllib.parse.quote(video_name, safe='')}/full",
        )

    def delete_video(self, video_name: str) -> None:
        self._delete_ignoring_404(
            f"/api/v1/videos/i/{urllib.parse.quote(video_name, safe='')}"
        )
