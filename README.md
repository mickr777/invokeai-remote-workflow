# InvokeAI Remote Workflow

A standalone InvokeAI v7 custom node for sharing queued workflow jobs with one or more **stock InvokeAI remote instances**.

Remote machines do **not** need this node pack installed.

> This is a community project and is not an official InvokeAI feature.

Current version: **v0.10.0**

## What it does

Add the **Remote Workflow** node to a workflow and leave it unconnected. The node coordinates real InvokeAI queue items across the local machine and configured remote workers.

Two modes are available:

- **Distributed** — Local plus the allowed remotes share compatible queued jobs.
- **Remote Only** — Only the allowed remotes render compatible jobs; Local acts as the coordinator when needed.

## Features

- Real queue-item claiming for Distributed mode
- Remote Only scheduling using stock InvokeAI workers
- **All Remotes** or **Specific** worker selection per workflow
- Offline workers are skipped before they claim local work
- Recovered workers can rejoin while eligible queued work remains
- Remote GPU OOM errors fail the local queue item instead of requeueing forever
- Live queue progress and preview relay
- Returned image and MP4 video import
- Original remote output source-node IDs are preserved in local queue results
- Queue result history persists across completion and refresh
- Source ImageField and VideoField inputs are copied and remapped automatically
- Missing single-file models can be transferred over the LAN
- Missing directory/Diffusers models can be installed by the stock remote from a saved Hugging Face source
- Exact Hugging Face subfolder syntax and explicit repo variants are preserved
- Optional primary Hugging Face token forwarding for gated model installs
- Concurrent generations share one missing-model install per worker/model
- Stale remote model records reported by stock `/api/v2/models/missing` are rejected
- Optional cleanup of remote queue items, transferred inputs, and imported outputs
- Multi-user primary installs preserve queue-user and board access checks

## Requirements

- InvokeAI v7 on the primary
- Compatible InvokeAI v7 on each remote
- Network access from the primary to each remote
- Python custom nodes enabled on the primary

On each remote, make InvokeAI reachable over the network:

```yaml
host: 0.0.0.0
```

If you want automatic **single-file LAN model transfer**, the remote must also allow private-network downloads:

```yaml
allow_private_download_urls: true
```

This is not required for directory/Diffusers models that the remote downloads directly from Hugging Face.

Only enable private download URLs on remotes and networks you trust.

## Install

Clone the repository into the primary InvokeAI custom nodes folder:

```bash
cd <invoke userfiles>/nodes
git clone https://github.com/mickr777/invokeai-remote-workflow.git invokeai_remote_workflow
```

Copy the example remote configuration:

```text
remotes.json.example -> remotes.json
```

Edit `remotes.json`, then restart InvokeAI.

To update later:

```bash
cd <invoke userfiles>/nodes/invokeai_remote_workflow
git pull --ff-only
```

## Configure remotes

Example:

```json
{
  "remotes": [
    {
      "name": "R1",
      "url": "http://192.168.15.100:9090",
      "enabled": true,
      "email": "",
      "password": "",
      "queue_id": "default",
      "timeout_seconds": 30,
      "probe_timeout_seconds": 2.5
    }
  ]
}
```

For single-user InvokeAI, leave `email` and `password` empty.

For multi-user InvokeAI, configure the normal InvokeAI login for that worker. Automatic model installation requires an administrator account on the remote because the stock model-install API is admin-only.

`remotes.json` is ignored by Git because it may contain credentials.

## Node settings

- **Enabled**
- **Mode** — `Distributed` or `Remote Only`
- **Workers** — `All Remotes` or `Specific`
- **Specific Remote**
- **Import To Gallery**
- **Cleanup Remote When Done**
- **Cancel Remote With Local**
- **Install Missing Models**
- **Use Primary HF Token**
- **Poll Interval**
- **Timeout**

## Distributed mode

Local and the allowed remote workers compete for compatible queue items.

```text
Distributed + All Remotes
    Local + R1 + R2 + ...

Distributed + Specific R1
    Local + R1
```

A worker is probed against its stock queue API immediately before it claims a real local queue row. If the worker is offline, it does not steal the item.

Remote workers can continue into later queued Invokes while another worker is still busy.

## Remote Only mode

Only the selected remote workers render the job.

```text
Remote Only + All Remotes
    R1 + R2 + ...

Remote Only + Specific R2
    R2 only
```

A stock Local worker can occasionally dequeue the first Remote Only row before the background pool has claimed it. In that case the Remote Workflow node uses Local only as the coordinator, hands the work to an allowed remote, waits for the result, and skips the remaining local graph.

## Missing models

**Install Missing Models** is off by default.

### Single-file models

The primary temporarily serves the model over the LAN and asks the stock remote to install that URL through its normal model-install API.

The remote requires:

```yaml
allow_private_download_urls: true
```

### Directory / Diffusers models

Automatic installation is supported when the local model record preserves a usable Hugging Face source.

Examples include sources such as:

```text
InvokeAI/t5-v1_1-xxl::bnb_llm_int8
```

The remote downloads that source itself through stock InvokeAI. Saved subfolders and explicit repo variants are preserved.

A remote model record is not considered usable when stock InvokeAI reports its root path missing through `/api/v2/models/missing`.

### Primary Hugging Face token

With **Use Primary HF Token** enabled, the primary reads its Hugging Face token and sends it only with the stock remote install request.

The token is not stored in `remotes.json` and is not logged by this node pack.

If the remote URL is plain HTTP, the token is not encrypted in transit.

## Workflow media

Supported:

- image results
- MP4 video results
- image inputs, masks, control/reference images, and nested/list ImageField references
- video inputs and nested/list VideoField references

Each distinct source image/video is transferred once per remote job and matching graph references are rewritten to the remote media names.

Only non-intermediate remote media is treated as final Gallery output.

## Gallery and boards

Remote board UUIDs are removed before the graph is sent to a worker because board IDs are installation-specific.

Returned media is imported into the resolved board on the primary. The selected Gallery board is followed for Auto destination where the stored Workbench state is available.

Imported media keeps the real source output node ID from the remote queue result so stock InvokeAI history and destination behavior can identify the correct result source.

## Cleanup

When **Cleanup Remote When Done** is enabled, the node can remove:

- transferred source image/video inputs after the remote job is safely finished
- successfully imported remote image/video outputs
- completed or canceled remote queue rows

Failed remote queue rows are deliberately retained for diagnosis/recovery.

If a remote cancellation request fails, transferred inputs are preserved because the remote may still be using them.

## Current limitations

- InvokeAI v7 only
- automatic LAN transfer supports single-file models only
- directory/Diffusers auto-install requires a usable saved Hugging Face source
- local-only directory models cannot be copied to an unmodified stock remote
- some capabilities in the full Remote Workers implementation require changes to InvokeAI itself and cannot be provided by a standalone custom node
- directory validation is therefore limited to the stock model hash plus stock missing-root detection
- the standalone node cannot hook queue enqueue as early as a core InvokeAI feature can
- queue-wide claiming relies on InvokeAI's SQLite session-queue internals and may require updates when InvokeAI changes those internals

## Background

This project is a standalone custom-node implementation of the Remote Workers work developed in [InvokeAI PR #9642](https://github.com/invoke-ai/InvokeAI/pull/9642).

The PR provides deeper integration into InvokeAI, including frontend controls and worker-side APIs. This project brings as much of that functionality as possible to a custom node while keeping remote InvokeAI instances completely stock.

It also provides a way to continue using the Remote Workers functionality independently if PR #9642 is not merged upstream.

**This node does not require PR #9642 or a modified InvokeAI installation on the remote workers.**

The following capabilities from the full Remote Workers implementation are not included because they require core/frontend or worker-side changes:

- frontend Remote Workers settings UI
- primary credential/settings APIs
- queue-enqueue hook
- custom worker-side model-layout endpoint
- custom worker-side Diffusers directory receiver
- frontend helper-node injection

## Security

Remote credentials are stored only in the local `remotes.json` file used by this node pack. Keep that file private.

For multi-user remotes, use a dedicated InvokeAI account where practical. Administrator permission is required only for stock model installation operations.

## License

Licensed under the Apache License, Version 2.0.

Portions of the implementation and design are derived from or adapted from InvokeAI and InvokeAI PR #9642. See `NOTICE`.
