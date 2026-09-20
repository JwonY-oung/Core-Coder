"""Pluggable sandbox backends for the bash tool.

The regex blacklist in tools/bash.py is a *first* layer: it catches the
obvious foot-guns but is trivially bypassed (base64, a script file, a
`python -c`, a renamed binary...). A real sandbox is the *second* layer:
it constrains what the spawned process can do at the OS level, so even a
command that slips past the blacklist cannot wreck the host.

Backends, weakest to strongest:

  none    - run the command as-is (default; identical to old behaviour)
  seccomp - Linux only. Install a seccomp filter in the child that blocks
            dangerous syscalls (mount, ptrace, reboot, ...). Cheap, but
            only filters syscalls -- the filesystem is still wide open.
  bwrap   - Linux only. bubblewrap: unshare namespaces, bind-mount only
            the working dir, drop the network. Strong FS + net isolation.
  docker  - any host with a container runtime. Run the command inside a
            throwaway container with the working dir mounted, no network,
            non-root user. Strongest and most portable.

Pick one with CORECODER_SANDBOX=<name>. The interface is deliberately tiny
so a new backend is ~30 lines: given a command and a cwd, return the argv
to exec and the cwd to exec it in.
"""

from __future__ import annotations

import os
import shutil
import sys
from abc import ABC, abstractmethod


class SandboxError(RuntimeError):
    """Raised when a requested backend is unavailable on this host."""


class Sandbox(ABC):
    """A backend that wraps a shell command so it runs isolated."""

    name: str = "none"

    @abstractmethod
    def wrap(self, command: str, cwd: str) -> tuple[list[str], str]:
        """Return (argv, cwd) that runs `command` inside the sandbox.

        argv is passed to subprocess without shell=True; the backend is
        responsible for invoking a shell itself (e.g. ["sh", "-c", cmd]).
        """
        ...

    def available(self) -> bool:
        """Whether this backend can run on the current host."""
        return True


class NoneSandbox(Sandbox):
    """No isolation. Kept as the default so behaviour is unchanged."""

    name = "none"

    def wrap(self, command: str, cwd: str) -> tuple[list[str], str]:
        # signal to the caller that it should use shell=True directly
        return ["__shell__", command], cwd


class DockerSandbox(Sandbox):
    """Run the command in a throwaway container.

    The working directory is bind-mounted read-write at the same path so
    relative paths and `cd` tracking keep working. Network is off by
    default; set CORECODER_SANDBOX_NET=1 to allow it (e.g. for pip).
    """

    name = "docker"

    def __init__(
        self,
        image: str = "python:3.12-slim",
        network: bool = False,
        memory: str = "2g",
        cpus: str = "2",
    ):
        self.image = image
        self.network = network
        self.memory = memory
        self.cpus = cpus

    def _runtime(self) -> str | None:
        for exe in ("docker", "podman"):
            if shutil.which(exe):
                return exe
        return None

    def available(self) -> bool:
        return self._runtime() is not None

    def wrap(self, command: str, cwd: str) -> tuple[list[str], str]:
        runtime = self._runtime()
        if runtime is None:
            raise SandboxError("docker/podman not found on PATH")
        # mount the working dir at the identical path inside the container
        host_cwd = os.path.abspath(cwd)
        # Docker Desktop on Windows wants the MSYS form (/d/x/y) for -v/-w;
        # both backslashes and "D:/x/y" are rejected as "not absolute".
        # Convert here so the same code works on Linux/macOS (no-op) and Windows.
        mount_cwd = host_cwd
        if os.name == "nt":
            drive, rest = os.path.splitdrive(host_cwd)
            mount_cwd = "/" + drive.rstrip(":").lower() + rest.replace("\\", "/")
        argv = [
            runtime,
            "run",
            "--rm",
            "-i",
            "--network", "bridge" if self.network else "none",
            "--memory", self.memory,
            "--cpus", self.cpus,
            # drop every capability, then add back nothing
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            # run as the host uid so files created are owned by the user
            "--user", f"{os.getuid()}:{os.getgid()}" if hasattr(os, "getuid") else "1000:1000",
            "-v", f"{mount_cwd}:{mount_cwd}",
            "-w", mount_cwd,
            self.image,
            "sh", "-c", command,
        ]
        return argv, host_cwd


class BwrapSandbox(Sandbox):
    """bubblewrap: namespaces + bind mounts, no daemon required.

    Only the working directory is writable; the rest of the host FS is
    read-only (or hidden). Network is unshared unless CORECODER_SANDBOX_NET.
    """

    name = "bwrap"

    def __init__(self, network: bool = False):
        self.network = network

    def available(self) -> bool:
        return sys.platform.startswith("linux") and shutil.which("bwrap") is not None

    def wrap(self, command: str, cwd: str) -> tuple[list[str], str]:
        if not self.available():
            raise SandboxError("bwrap not available (Linux + bubblewrap required)")
        host_cwd = os.path.abspath(cwd)
        argv = [
            "bwrap",
            "--die-with-parent",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            # read-only system dirs
            "--ro-bind", "/usr", "/usr",
            "--ro-bind", "/bin", "/bin",
            "--ro-bind", "/lib", "/lib",
            "--ro-bind", "/lib64", "/lib64",
            "--ro-bind", "/etc", "/etc",
            "--proc", "/proc",
            "--dev", "/dev",
            "--tmpfs", "/tmp",
            # the working dir is the only writable window
            "--bind", host_cwd, host_cwd,
            "--chdir", host_cwd,
        ]
        if not self.network:
            argv.append("--unshare-net")
        argv += ["sh", "-c", command]
        return argv, host_cwd


class SeccompSandbox(Sandbox):
    """Linux seccomp filter installed in the child via preexec_fn.

    This is the weakest real sandbox: it blocks dangerous *syscalls* but
    leaves the filesystem and network reachable. Useful as a cheap extra
    layer, or on hosts where bwrap/docker are unavailable. Requires the
    `pyseccomp` package (pip install pyseccomp).
    """

    name = "seccomp"

    # syscalls that are almost never needed by a build/test command but are
    # the building blocks of container escapes and host tampering
    _BLOCKED = [
        "mount", "umount2", "pivot_root", "chroot",
        "ptrace", "process_vm_readv", "process_vm_writev",
        "reboot", "kexec_load", "init_module", "finit_module", "delete_module",
        "setns", "unshare", "bpf", "userfaultfd", "perf_event_open",
    ]

    def available(self) -> bool:
        if not sys.platform.startswith("linux"):
            return False
        try:
            import seccomp  # noqa: F401
        except ImportError:
            return False
        return True

    def wrap(self, command: str, cwd: str) -> tuple[list[str], str]:
        if not self.available():
            raise SandboxError("seccomp backend needs Linux + pyseccomp")
        # the filter is installed by the caller through preexec_fn; here we
        # only describe the command. See install_filter().
        return ["__shell__", command], cwd

    @staticmethod
    def install_filter() -> None:
        """preexec_fn: runs in the child right before exec."""
        import seccomp

        f = seccomp.SyscallFilter(defaction=seccomp.ALLOW)
        for name in SeccompSandbox._BLOCKED:
            try:
                f.add_rule(seccomp.ERRNO(1), name)  # EPERM
            except Exception:  # noqa: BLE001 - syscall may not exist on this kernel
                pass
        f.load()


_BACKENDS = {
    "none": NoneSandbox,
    "docker": DockerSandbox,
    "bwrap": BwrapSandbox,
    "seccomp": SeccompSandbox,
}


def get_sandbox(name: str | None = None, **kwargs) -> Sandbox:
    """Build a backend by name, falling back to `none` if unavailable.

    Reads CORECODER_SANDBOX when name is None. If the requested backend
    cannot run here we degrade to `none` rather than crashing the agent,
    but we print a one-line warning so the downgrade is never silent.
    """
    name = (name or os.getenv("CORECODER_SANDBOX", "none")).strip().lower()
    cls = _BACKENDS.get(name)
    if cls is None:
        raise SandboxError(f"unknown sandbox backend: {name!r} (have {list(_BACKENDS)})")

    net = os.getenv("CORECODER_SANDBOX_NET", "0") == "1"
    if name == "docker":
        sb = DockerSandbox(
            image=os.getenv("CORECODER_SANDBOX_IMAGE", "python:3.12-slim"),
            network=net,
        )
    elif name == "bwrap":
        sb = BwrapSandbox(network=net)
    else:
        sb = cls(**kwargs)

    if not sb.available():
        print(
            f"[sandbox] backend {name!r} unavailable on this host; "
            f"falling back to 'none' (no isolation)",
            file=sys.stderr,
        )
        return NoneSandbox()
    return sb
