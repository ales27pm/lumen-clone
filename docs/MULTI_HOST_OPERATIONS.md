# Lumen Multi-Host Operations

## Purpose

Lumen uses two independent developer hosts:

| Executor | Owns | Does not own |
| --- | --- | --- |
| macOS virtual machine | Xcode, iOS/macOS compilation and tests, Simulator/device work, signing, archives, and App Store delivery | CUDA training, GPU inference, or memory-heavy model processing |
| Physical Ubuntu machine | NVIDIA/CUDA workloads, controlled adapter training, heavyweight model conversion, high-memory generation and verification | Xcode, Apple signing, Simulator/device work, or App Store delivery |

Either host can initiate work. The initiating host is a dispatcher; the host with the required capability is the executor. This avoids making macOS a bottleneck while preserving the Apple toolchain boundary.

## Source And Workspace Contract

GitHub is the source synchronization boundary. Each host has its own clone and its own machine-local caches, SDKs, model storage, Docker state, Xcode `DerivedData`, and credentials.

1. Make and review source changes in one clone.
2. Push the intended commit to GitHub.
3. The executor fetches that exact commit into its own clean checkout or dedicated worktree.
4. Dispatch only the immutable commit, task parameters, and approved input locations.
5. Return a concise result containing the executed commit, task outcome, and evidence/artifact references.

Never dispatch uncommitted edits by copying a worktree. Never share a mutable checkout, `DerivedData`, virtual environment, Docker state, model cache, or output directory between the hosts.

## Dispatch Contract

Cross-host transport is SSH with per-host keys and aliases configured outside the repository. A request must identify:

- the executor: `macos` or `ubuntu`;
- the immutable source commit;
- a named operation and its safe preflight mode, if one exists;
- inputs, expected outputs, and whether the job can allocate GPU, stop a service, upload, sign, archive, or otherwise create external state;
- the evidence required for success and the return route for artifacts or reports.

The receiving host independently checks capability, repository identity, commit availability, clean-worktree policy, free disk, and task-specific preconditions before starting. It must fail closed if any identity check fails.

## Lumen-Specific Routing

- Send `xcodebuild`, Simulator XCTest, device checks, signing, archiving, and App Store operations to macOS.
- Send the controlled Ubuntu training pipeline, CUDA/container preflight, adapter evaluation, GGUF conversion, and high-memory work to Ubuntu.
- The existing Ubuntu training launcher remains guarded: it requires an exact clean checkout and records source closure and lineage. A remote dispatch does not weaken those guards.
- Training, conversion, and evaluation results are candidates only. Promotion into the iOS runtime still requires the repository's artifact, runtime-binding, and Apple-side validation evidence.

## Lumen Clone Training Handoff

For Lumen Clone, the intended handoff is deliberately staged:

1. **macOS prepares repository intelligence.** Run the source crawler, self-awareness/manifest generation, and their local validation against the current Apple-side checkout. Review the resulting source and generated-artifact scope together, then commit and push the accepted state.
2. **Ubuntu receives a frozen source revision.** Fetch the named commit into an independent clean checkout or dedicated training worktree. Re-run the source-integrity and generated-input checks there; do not transfer a macOS worktree or cache.
3. **Ubuntu prepares external training inputs.** Download and validate public corpus inputs, apply the repository's provenance, licensing, deduplication, contamination, and cleanliness rules, then regenerate or verify the controlled training inputs. Keep downloaded source data and large caches local to Ubuntu unless the repository explicitly tracks their derived, reviewed outputs.
4. **Ubuntu trains and verifies.** Run the controlled GPU preflight, training, evaluation, artifact hashes, and runtime-binding checks. Preserve the run evidence even when a run fails.
5. **macOS evaluates a candidate for Apple use.** Only an evaluated, lineage-bound candidate returns for iOS-compatible conversion/loading and Apple-side validation. A training result is never copied directly into the shipping app without that promotion gate.

This ordering makes the Mac the authoritative producer of source-derived intelligence while using Ubuntu for network-heavy corpus preparation and GPU-intensive work. Corpus preparation may be performed on macOS only when a task specifically needs Apple-side tooling; its large downloads, caches, and training execution still belong on Ubuntu by default.

## Artifact And Evidence Return

Git commits return through GitHub. Large artifacts and reports return through an explicitly configured artifact store or transfer path, with checksums and producing-commit metadata. Do not place model weights, Xcode archives, credentials, or machine-local caches in Git.

Completion messages must distinguish: preflight passed, task executed, artifact produced, artifact verified, and artifact promoted. An Ubuntu success does not prove iOS runtime integration; an Apple build does not prove GPU training quality.

## Failure And Concurrency Rules

- Long GPU jobs run from a dedicated clean Ubuntu worktree and preserve their run directory on failure.
- Apple build/test jobs use macOS-local build products only.
- Do not overwrite a remote worktree, stop a GPU service, cancel a training job, upload artifacts, sign, archive, or submit a build without explicit operator intent.
- If either host is unavailable, retain the exact commit and task description, report the missing capability, and do not silently reroute an Apple task to Ubuntu or a GPU task to macOS.

## Initial Machine Setup Checklist

Before enabling automated dispatch, verify both directions independently: SSH host-key trust and least-privilege keys, GitHub authentication, repository remote identity, a clean checkout fetch at a named commit, disk capacity, and a harmless capability probe. On macOS, verify Xcode and the intended simulator/device tooling. On Ubuntu, verify NVIDIA access, Docker GPU access, and the existing training preflight. Store aliases, keys, paths, and credentials only in each host's local configuration.
