# RESULTS: Kubernetes attempt (follow-up, 2026-11-06)

Outcome: no cluster could be run in this environment. The project's k8s/
manifests remain authored and validated only (YAML parsing and required
field checks in the test suite), and they were never applied to any cluster.
This file records exactly what was attempted and the decisive blocker.

## Pre-checks

- No Docker on this VM, so the attempt used k3s, which runs containerd
  directly and needs no Docker.
- PID 1 is systemd and systemctl reports the system as running.
- cgroup v2 is mounted (cgroup2fs), kernel 7.0.0-43-generic.
- Two warning signs found before installing: /dev/kmsg does not exist, and
  /sys/fs/cgroup/cgroup.controllers is EMPTY (see the blocker below).

## What was attempted

1. Official install script from get.k3s.io, channel stable, release
   v1.37.2+k3s1. The script's own binary download failed on the first run
   ([ERROR] Download failed). A direct curl of the release binary to /tmp
   then failed with curl exit 23 because /tmp is a 512 MB tmpfs that was
   95 percent full. The binary (78 MB) was therefore downloaded to the
   workspace filesystem instead, and its sha256 matched the official
   sha256sum-amd64.txt exactly:
   265b3ee6a3198ef14033c63ef46cbb32370d83795199fb6d1b01f21575eddbda.
2. Installed with INSTALL_K3S_SKIP_DOWNLOAD=true and
   INSTALL_K3S_EXEC="--disable=traefik,servicelb,metrics-server".
   The install completed and created the k3s systemd service.
3. The service never became ready. journalctl -u k3s shows the decisive
   failure, repeated on every restart attempt:

   time="2026-11-06T19:11:09Z" level=fatal msg="Error: failed to find cpu cgroup (v2)"
   k3s.service: Main process exited, code=exited, status=1/FAILURE

4. A manual foreground run (k3s server with the same disable flags) hit the
   identical fatal line within seconds, confirming it is not a systemd
   unit problem.

## The blocker, precisely

This VM runs inside a nested cgroup namespace in which no controllers are
delegated: /sys/fs/cgroup/cgroup.controllers is empty (no cpu, memory, io
or pids controller is available to processes here), and /proc/self/cgroup
shows the process at the namespace root (0::/). k3s and its kubelet need a
working cpu cgroup (v2) to start; without one the server exits fatally
before the API server ever listens (kubectl only ever saw connection
refused on the fallback localhost:8080, and port 6443 never opened).
Fixing this would require the host to delegate cgroup controllers into
this environment, which cannot be done from inside it. No manifest was
ever applied: there was never an API server to apply one to.

## Teardown

- systemctl stop k3s, then the official /usr/local/bin/k3s-uninstall.sh.
- Verified afterwards: the k3s.service unit no longer exists, the k3s and
  kubectl binaries were removed from /usr/local/bin, /etc/rancher is gone,
  /var/lib/rancher was emptied by the uninstaller and its empty shell
  removed, port 6443 is closed, and no k3s processes remain. Nothing from
  this attempt is left running.

## What would be needed to make this a really-run item

A host (or a VM with full cgroup delegation) where
/sys/fs/cgroup/cgroup.controllers lists cpu and memory, or a managed
cluster kubeconfig. On such a host: kubectl apply -f k8s/ would still not
make the deployment ready on its own, because the deployment references a
Secret named travel-telemetry-secrets that has no manifest (by design, no
secret values live in this repo) and an image travel-telemetry-platform:latest
that must be built and loaded first (no Docker here either). Those two gaps
are recorded so a future run does not mistake a cluster for a deployment.
