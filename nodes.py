from typing import Literal

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import InputField
from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.app.services.shared.invocation_context import InvocationContext

from .remote_client import RemoteInvokeError
from .remote_config import (
    ConfiguredRemote,
    RemoteConfigFileError,
    enabled_remotes,
    load_remotes,
)
from .worker_pool import (
    _RemoteWorkflowSettings,
    _acquire_remote_worker_slot,
    _available_remote_candidates,
    _build_import_metadata,
    _build_remote_graph_from_queue_item,
    _candidate_remotes,
    _ensure_remote_queue_coordinator,
    _mark_remaining_local_nodes_skipped,
    _remote_queue_item_status,
    _resolve_gallery_board,
)

@invocation_output("remote_workflow_output")
class RemoteWorkflowOutput(BaseInvocationOutput):
    """No-connectors output for the side-effect-only Remote Workflow node."""

    pass


@invocation(
    "irw_remote_workflow",
    title="Remote Workflow",
    tags=["remote", "invokeai", "worker", "distributed"],
    category="Remote Workflow",
    version="0.10.0",
    use_cache=False,
)
class AAARemoteWorkflowInvocation(BaseInvocation):
    """Mark queued workflow items for the remote worker pool."""

    enabled: bool = InputField(default=True, description="Enable remote worker scheduling")
    mode: Literal["Distributed", "Remote Only"] = InputField(
        default="Distributed",
        description=(
            "Distributed lets Local and remotes share compatible queued jobs. "
            "Remote Only sends compatible jobs only to remotes; if Local dequeues one first, "
            "it hands that item to one allowed remote and skips the remaining local graph."
        ),
    )
    dispatch: Literal["All Remotes", "Specific"] = InputField(
        default="All Remotes",
        title="Workers",
        description=(
            "All Remotes lets every enabled remote take work as it becomes free. "
            "Specific limits this workflow to the named remote."
        ),
    )
    specific_remote: str = InputField(
        default="R1",
        description="Remote name from remotes.json when Workers is Specific",
    )
    import_to_gallery: bool = InputField(
        default=True,
        description="Import completed remote gallery images into the local Gallery",
    )
    cleanup_remote_when_done: bool = InputField(
        default=False,
        description="Delete remote queue items, transferred inputs, and successfully imported outputs when safe",
    )
    cancel_remote_with_local: bool = InputField(
        default=True,
        description="Cancel the active remote job when its local queue item is canceled",
    )
    install_missing_models: bool = InputField(
        default=False,
        title="Install Missing Models",
        description=(
            "When a required model is missing, transfer single-file models from this host over LAN. "
            "For directory models originally installed from Hugging Face, ask the stock remote to "
            "install the saved Hugging Face source. Multi-user remotes require administrator credentials."
        ),
    )
    use_primary_hf_token: bool = InputField(
        default=False,
        title="Use Primary HF Token",
        description=(
            "For missing Hugging Face directory models, forward this primary InvokeAI installation's "
            "Hugging Face token with the remote install request. Useful for gated models. "
            "Leave off to let the remote use its own Hugging Face credentials."
        ),
    )
    poll_interval_seconds: float = InputField(default=0.35, ge=0.1, le=10.0)
    timeout_seconds: int = InputField(default=1800, ge=10, le=86400)

    def _load_candidates(
        self,
        context: InvocationContext,
    ) -> tuple[list[ConfiguredRemote], list[ConfiguredRemote]]:
        try:
            remotes = load_remotes()
        except RemoteConfigFileError:
            raise
        except Exception as exc:
            raise RemoteConfigFileError(f"Could not load remotes.json: {exc}") from exc

        candidates = _candidate_remotes(
            remotes,
            self.dispatch,
            self.specific_remote,
        )

        if self.dispatch == "Specific" and not candidates:
            message = (
                f"Specific remote '{self.specific_remote}' was not found or is disabled in remotes.json"
            )
            if self.mode == "Distributed":
                context.logger.warning(f"Remote Workflow: {message}; Local will continue normally")
                return remotes, []
            raise RemoteInvokeError(message)

        if not enabled_remotes(remotes):
            if self.mode == "Distributed":
                context.logger.warning(
                    "Remote Workflow: no remotes are enabled in remotes.json; Local will continue normally"
                )
                return remotes, []
            raise RemoteInvokeError("Remote Only requested but no remotes are enabled in remotes.json")

        return remotes, candidates

    def _settings(self) -> _RemoteWorkflowSettings:
        return _RemoteWorkflowSettings(
            mode=self.mode,
            workers=self.dispatch,
            specific_remote=self.specific_remote.strip(),
            import_to_gallery=bool(self.import_to_gallery),
            cleanup_remote_when_done=bool(self.cleanup_remote_when_done),
            cancel_remote_with_local=bool(self.cancel_remote_with_local),
            install_missing_models=bool(self.install_missing_models),
            use_primary_hf_token=bool(self.use_primary_hf_token),
            poll_interval_seconds=float(self.poll_interval_seconds),
            timeout_seconds=int(self.timeout_seconds),
        )

    def invoke(self, context: InvocationContext) -> RemoteWorkflowOutput:
        if not self.enabled:
            context.logger.info("Remote Workflow: disabled; local workflow will run normally")
            return RemoteWorkflowOutput()

        if context.util.is_canceled():
            raise CanceledException

        remotes, candidates = self._load_candidates(context)

        if self.mode == "Distributed":
            if not candidates:
                return RemoteWorkflowOutput()

            _coordinator, started = _ensure_remote_queue_coordinator(
                remotes,
                context,
            )
            if started:
                context.logger.info(
                    f"Remote Workflow: queue-wide worker pool started ({self.dispatch}); "
                    "Local and remotes can continue across compatible queued Invokes"
                )
            else:
                context.logger.info(
                    "Remote Workflow: queue-wide worker pool already active for this user"
                )
            return RemoteWorkflowOutput()

        # Remote Only. Local has already dequeued this first/occasional item, so
        # use Local only as the coordinator: execute the graph on one allowed
        # remote, keep the other remote lanes consuming pending real queue rows,
        # then skip the remaining local graph.
        remote_graph, local_board_id = _build_remote_graph_from_queue_item(
            context._data.queue_item,
            context._services,
            context.logger,
        )
        if local_board_id is None:
            local_board_id = _resolve_gallery_board(context._data.queue_item, context._services, context.logger)
        import_meta = _build_import_metadata(
            context._data.queue_item,
            context._services,
            context.logger,
            local_board_id,
            node_id=str(context._data.invocation.id),
        )

        if local_board_id:
            context.logger.info(
                f"Remote Workflow: returned remote media will import to local board {local_board_id}"
            )
        else:
            context.logger.info(
                "Remote Workflow: returned remote media will import to Uncategorized"
            )

        settings = self._settings()

        candidates, unavailable_candidates = _available_remote_candidates(candidates)
        if not candidates:
            if self.dispatch == "Specific":
                selected = self.specific_remote.strip() or "(unnamed remote)"
                detail = str(unavailable_candidates[0][1]) if unavailable_candidates else "not available"
                raise RemoteInvokeError(
                    f"Selected remote '{selected}' is unavailable: {detail}"
                )

            detail = "; ".join(
                f"{remote.name}: {exc}"
                for remote, exc in unavailable_candidates
            ) or "no enabled remote is available"
            raise RemoteInvokeError(
                f"No allowed remote worker is currently available: {detail}"
            )

        for unavailable, exc in unavailable_candidates:
            context.logger.warning(
                f"Remote Workflow [{unavailable.name}]: allowed by this workflow but unavailable; "
                f"skipping it for the current Remote Only item: {exc}"
            )

        queue_service = context._services.session_queue
        set_status = getattr(queue_service, "_set_queue_item_status", None)
        if set_status is None:
            raise RemoteInvokeError("Remote Only requires _set_queue_item_status()")

        local_queue_item = context._data.queue_item
        local_item_id = int(local_queue_item.item_id)
        if _remote_queue_item_status(context._services, local_item_id) == "in_progress":
            set_status(
                item_id=local_item_id,
                status="waiting",
                queue_item=local_queue_item,
            )
            context.logger.info(
                f"Remote Workflow: Remote Only local item {local_item_id} is waiting "
                "for a free remote worker"
            )

        initial_remote, initial_lock = _acquire_remote_worker_slot(
            candidates=candidates,
            context=context,
            prioritize_local_handoff=True,
        )

        if _remote_queue_item_status(context._services, local_item_id) == "waiting":
            set_status(
                item_id=local_item_id,
                status="in_progress",
                device=f"remote:{initial_remote.name}",
                queue_item=local_queue_item,
            )

        try:
            coordinator, started = _ensure_remote_queue_coordinator(
                remotes,
                context,
            )
        except Exception:
            initial_lock.release()
            raise

        if started:
            context.logger.info(
                f"Remote Workflow: queue-wide worker pool started ({self.dispatch}) for Remote Only"
            )

        final_preview = coordinator.run_current_remote_only(
            context=context,
            remote_graph=remote_graph,
            import_meta=import_meta,
            settings=settings,
            candidates=candidates,
            initial_remote=initial_remote,
            initial_lock=initial_lock,
        )

        _mark_remaining_local_nodes_skipped(context)
        context.util.signal_progress(
            "Remote Only completed",
            1.0,
            image=final_preview,
            image_size=final_preview.size if final_preview is not None else None,
        )
        return RemoteWorkflowOutput()
