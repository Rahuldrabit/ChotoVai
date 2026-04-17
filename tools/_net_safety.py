"""
Network safety helpers for read-only web tools.

Goal: prevent SSRF and accidental access to private/internal resources.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse


_BLOCKED_HOSTNAMES = {
    "localhost",
    "127.0.0.1",
    "::1",
}


@dataclass(frozen=True)
class UrlSafety:
    ok: bool
    reason: str = ""


def _is_private_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True
    return bool(
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    )


def _resolved_ips(hostname: str) -> Iterable[str]:
    # Resolve all A/AAAA records; if resolution fails treat as unsafe.
    infos = socket.getaddrinfo(hostname, None)
    for family, _, _, _, sockaddr in infos:
        if family == socket.AF_INET:
            yield sockaddr[0]
        elif family == socket.AF_INET6:
            yield sockaddr[0]


def validate_public_http_url(url: str) -> UrlSafety:
    """
    Allow only http(s) URLs that resolve to public IPs.
    Blocks localhost/private ranges and non-http schemes.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return UrlSafety(False, "invalid URL parse")

    if parsed.scheme not in {"http", "https"}:
        return UrlSafety(False, f"blocked scheme: {parsed.scheme!r}")
    if not parsed.netloc:
        return UrlSafety(False, "missing host")

    host = parsed.hostname or ""
    if not host:
        return UrlSafety(False, "missing hostname")

    if host.lower() in _BLOCKED_HOSTNAMES:
        return UrlSafety(False, "blocked hostname")

    # Block direct IP literals if private-ish.
    try:
        ipaddress.ip_address(host)
        if _is_private_ip(host):
            return UrlSafety(False, "blocked IP address")
        return UrlSafety(True)
    except ValueError:
        pass

    try:
        ips = list(_resolved_ips(host))
    except Exception:
        return UrlSafety(False, "DNS resolution failed")

    if not ips:
        return UrlSafety(False, "no resolved IPs")
    if any(_is_private_ip(ip) for ip in ips):
        return UrlSafety(False, "hostname resolves to private IP")

    return UrlSafety(True)

