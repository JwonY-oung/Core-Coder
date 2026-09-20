"""Tests for the pluggable sandbox backends."""

import os
import sys

import pytest

from corecoder.sandbox import (
    BwrapSandbox,
    DockerSandbox,
    NoneSandbox,
    SandboxError,
    SeccompSandbox,
    get_sandbox,
)


def test_none_backend_uses_shell_sentinel():
    argv, cwd = NoneSandbox().wrap("echo hi", os.getcwd())
    assert argv == ["__shell__", "echo hi"]
    assert cwd == os.getcwd()


def test_get_sandbox_default_is_none(monkeypatch):
    monkeypatch.delenv("CORECODER_SANDBOX", raising=False)
    assert get_sandbox().name == "none"


def test_get_sandbox_strips_whitespace(monkeypatch):
    # env values picked up from cmd.exe can carry a trailing space
    monkeypatch.setenv("CORECODER_SANDBOX", " none ")
    assert get_sandbox().name == "none"


def test_get_sandbox_unknown_raises():
    with pytest.raises(SandboxError):
        get_sandbox("nope")


def test_unavailable_backend_degrades_to_none(monkeypatch):
    # force docker to look unavailable regardless of host
    monkeypatch.setattr(DockerSandbox, "available", lambda self: False)
    assert get_sandbox("docker").name == "none"


def test_docker_wrap_shape(monkeypatch):
    sb = DockerSandbox(image="alpine", network=False)
    monkeypatch.setattr(sb, "_runtime", lambda: "docker")
    argv, cwd = sb.wrap("echo hi", os.getcwd())
    assert argv[0] in ("docker", "podman")
    assert "run" in argv
    assert "--network" in argv and "none" in argv
    assert "alpine" in argv
    # command is passed to a shell inside the container
    assert argv[-3:] == ["sh", "-c", "echo hi"]
    assert cwd == os.path.abspath(os.getcwd())


def test_docker_network_toggle(monkeypatch):
    sb = DockerSandbox(network=True)
    monkeypatch.setattr(sb, "_runtime", lambda: "docker")
    argv, _ = sb.wrap("x", os.getcwd())
    i = argv.index("--network")
    assert argv[i + 1] == "bridge"


def test_bwrap_wrap_shape(monkeypatch):
    sb = BwrapSandbox(network=False)
    monkeypatch.setattr(sb, "available", lambda: True)
    argv, cwd = sb.wrap("echo hi", os.getcwd())
    assert argv[0] == "bwrap"
    assert "--unshare-net" in argv
    assert argv[-3:] == ["sh", "-c", "echo hi"]


def test_bwrap_available_only_on_linux():
    if not sys.platform.startswith("linux"):
        assert BwrapSandbox().available() is False


def test_seccomp_available_only_on_linux():
    if not sys.platform.startswith("linux"):
        assert SeccompSandbox().available() is False


def test_seccomp_blocked_list_has_escape_syscalls():
    for name in ("mount", "ptrace", "reboot", "setns", "unshare"):
        assert name in SeccompSandbox._BLOCKED
