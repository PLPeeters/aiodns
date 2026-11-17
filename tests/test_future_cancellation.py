"""Future cancellation through public resolver APIs and a local DNS peer."""

import asyncio
import gc
import logging
import socket
import struct
import warnings
import weakref
from collections.abc import AsyncIterator, Callable
from contextvars import Context
from dataclasses import dataclass
from typing import Any

import pycares
import pytest

import aiodns


@dataclass
class Request:
    packet: bytes
    address: tuple[str, int]
    name: str
    qtype: int
    question_end: int


class ControlledDNS(asyncio.DatagramProtocol):
    def __init__(self) -> None:
        self.transport: asyncio.DatagramTransport | None = None
        self.requests: asyncio.Queue[Request] = asyncio.Queue()

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore[assignment]

    def datagram_received(self, packet: bytes, address: Any) -> None:
        offset = 12
        labels = []
        while packet[offset]:
            size = packet[offset]
            offset += 1
            labels.append(packet[offset : offset + size].decode('ascii'))
            offset += size
        qtype = struct.unpack('!H', packet[offset + 1 : offset + 3])[0]
        self.requests.put_nowait(
            Request(
                packet, address, '.'.join(labels).lower(), qtype, offset + 5
            )
        )

    async def receive(self, count: int) -> list[Request]:
        return [
            await asyncio.wait_for(self.requests.get(), 1)
            for _ in range(count)
        ]

    def answer(self, request: Request) -> None:
        assert self.transport is not None
        data = (
            socket.inet_pton(socket.AF_INET, '192.0.2.1')
            if request.qtype == 1
            else socket.inet_pton(socket.AF_INET6, '2001:db8::1')
        )
        header = request.packet[:2] + struct.pack('!HHHHH', 0x8180, 1, 1, 0, 0)
        answer = (
            b'\xc0\x0c'
            + struct.pack('!HHIH', request.qtype, 1, 0, len(data))
            + data
        )
        self.transport.sendto(
            header + request.packet[12 : request.question_end] + answer,
            request.address,
        )


@pytest.fixture(params=[True, False], ids=['event-thread', 'fallback'])
async def integration(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[aiodns.DNSResolver, ControlledDNS]]:
    loop = asyncio.get_running_loop()
    transport, server = await loop.create_datagram_endpoint(
        ControlledDNS, local_addr=('127.0.0.1', 0)
    )
    native_channel = pycares.Channel
    if not request.param:

        def channel_factory(*args: Any, **kwargs: Any) -> pycares.Channel:
            if 'sock_state_cb' not in kwargs:
                raise pycares.AresError(1, 'force fallback integration')
            return native_channel(*args, **kwargs)

        monkeypatch.setattr(pycares, 'Channel', channel_factory)
    resolver = aiodns.DNSResolver(
        loop=loop,
        timeout=1,
        tries=2,
        flags=pycares.ARES_FLAG_STAYOPEN | pycares.ARES_FLAG_NOSEARCH,
    )
    resolver.nameservers = [
        f'127.0.0.1:{transport.get_extra_info("sockname")[1]}'
    ]
    assert resolver._event_thread is request.param
    errors: list[dict[str, Any]] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: errors.append(context))
    try:
        yield resolver, server
    finally:
        await resolver.close()
        await asyncio.sleep(0)
        transport.close()
        loop.set_exception_handler(previous_handler)
        assert not errors


def submit(
    resolver: aiodns.DNSResolver, method: str, name: str
) -> asyncio.Future[Any]:
    if method == 'query_dns':
        return resolver.query_dns(name, 'A')
    if method == 'getaddrinfo':
        return resolver.getaddrinfo(name, socket.AF_UNSPEC)
    assert method == 'gethostbyname'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        return resolver.gethostbyname(name, socket.AF_UNSPEC)


@pytest.mark.parametrize(
    'method', ['query_dns', 'gethostbyname', 'getaddrinfo']
)
async def test_future_cancel_stops_native_work_and_preserves_unrelated_query(
    integration: tuple[aiodns.DNSResolver, ControlledDNS], method: str
) -> None:
    resolver, server = integration
    count = 1 if method == 'query_dns' else 2
    cancelled = submit(resolver, method, 'cancel.example.')
    pending_requests = await server.receive(count)
    assert {item.name for item in pending_requests} == {'cancel.example'}
    other = submit(resolver, method, 'other.example.')
    other_requests = await server.receive(count)
    assert {item.name for item in other_requests} == {'other.example'}

    assert cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    await asyncio.sleep(0)
    assert not other.done()
    for item in other_requests:
        server.answer(item)
    assert await asyncio.wait_for(other, 1)
    assert resolver._channel.wait(0)
    assert cancelled.cancelled()
    assert not resolver._pending_queries
    if not resolver._event_thread:
        assert resolver._timer is None


@pytest.mark.parametrize(
    'method', ['query_dns', 'gethostbyname', 'getaddrinfo']
)
async def test_cancelled_temporary_resolver_lives_through_native_terminal(
    integration: tuple[aiodns.DNSResolver, ControlledDNS],
    method: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    configured, server = integration
    # The forced-fallback warning contains a setup exception traceback.
    # Do not let pytest log capture retain that traceback and the resolver.
    caplog.set_level(logging.ERROR, logger='aiodns')
    resolver = aiodns.DNSResolver(
        loop=asyncio.get_running_loop(),
        nameservers=configured._channel.servers,
        timeout=1,
        tries=2,
        flags=pycares.ARES_FLAG_STAYOPEN | pycares.ARES_FLAG_NOSEARCH,
    )
    reference = weakref.ref(resolver)
    future = submit(resolver, method, 'temporary.example.')
    await server.receive(1 if method == 'query_dns' else 2)
    del resolver
    gc.collect()
    try:
        assert reference() is not None
        assert future.cancel()
        # Cancellation's done callback has not run yet. Native callback state
        # must still keep the temporary resolver alive.
        gc.collect()
        assert reference() is not None
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        gc.collect()
        assert reference() is None
        assert future.cancelled()
    finally:
        remaining = reference()
        if remaining is not None:
            await remaining.close()


@pytest.mark.parametrize(
    'method', ['query_dns', 'gethostbyname', 'getaddrinfo']
)
@pytest.mark.parametrize('cancel_first', [True, False])
async def test_future_cancellation_races_resolver_close(
    integration: tuple[aiodns.DNSResolver, ControlledDNS],
    method: str,
    cancel_first: bool,
) -> None:
    resolver, server = integration
    future = submit(resolver, method, 'close.example.')
    await server.receive(1 if method == 'query_dns' else 2)
    if cancel_first:
        assert future.cancel()
        await resolver.close()
    else:
        await resolver.close()
        future.cancel()
    await asyncio.sleep(0)
    if future.cancelled():
        with pytest.raises(asyncio.CancelledError):
            await future
    else:
        with pytest.raises(aiodns.error.DNSError) as exception:
            await future
        assert exception.value.args[0] == pycares.errno.ARES_ECANCELLED
    assert not resolver._pending_queries
    assert resolver._timer is None


@pytest.mark.parametrize('method', ['gethostbyname', 'getaddrinfo'])
async def test_cancel_after_partial_address_completion_discards_partial_result(
    integration: tuple[aiodns.DNSResolver, ControlledDNS], method: str
) -> None:
    resolver, server = integration
    future = submit(resolver, method, 'partial.example.')
    requests = await server.receive(2)
    server.answer(next(item for item in requests if item.qtype == 1))
    # Receive a separate successful response after the A reply, so native
    # response processing has advanced while the AAAA reply remains withheld.
    barrier = resolver.query_dns('barrier.example.', 'A')
    for item in await server.receive(1):
        server.answer(item)
    await asyncio.wait_for(barrier, 1)
    assert not future.done()
    assert future.cancel()
    await asyncio.sleep(0)
    with pytest.raises(asyncio.CancelledError):
        await future
    assert resolver._channel.wait(0)
    assert not resolver._pending_queries


@pytest.mark.parametrize(
    'method', ['query_dns', 'gethostbyname', 'getaddrinfo']
)
async def test_completed_future_rejects_cancellation(
    integration: tuple[aiodns.DNSResolver, ControlledDNS], method: str
) -> None:
    resolver, server = integration
    future = submit(resolver, method, 'completed.example.')
    for item in await server.receive(1 if method == 'query_dns' else 2):
        server.answer(item)
    result = await asyncio.wait_for(future, 1)
    assert not future.cancel()
    assert future.result() is result
    assert resolver._channel.wait(0)


@pytest.mark.parametrize(
    'method', ['query_dns', 'gethostbyname', 'getaddrinfo']
)
async def test_cancelled_future_ignores_queued_native_completion(
    integration: tuple[aiodns.DNSResolver, ControlledDNS],
    method: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver, server = integration
    if not resolver._event_thread:
        pytest.skip('Thread-to-loop delivery applies to event-thread mode')
    loop = asyncio.get_running_loop()
    queued: list[tuple[Callable[..., Any], tuple[Any, ...]]] = []

    def hold_delivery(
        callback: Callable[..., Any],
        *args: Any,
        context: Context | None = None,
    ) -> asyncio.Handle:
        queued.append((callback, args))
        return asyncio.Handle(callback, args, loop, context=context)

    # Hold the external thread-to-loop scheduler boundary while native work
    # completes. Cancellation can then win the Future before queued delivery.
    with monkeypatch.context() as patch:
        patch.setattr(loop, 'call_soon_threadsafe', hold_delivery)
        future = submit(resolver, method, 'queued.example.')
        requests = await server.receive(1 if method == 'query_dns' else 2)
        for item in requests:
            server.answer(item)
        assert resolver._channel.wait(1)
    assert len(queued) == 1
    assert not future.done()
    assert future.cancel()
    await asyncio.sleep(0)
    for callback, args in queued:
        callback(*args)
    assert future.cancelled()
    assert resolver._channel.wait(0)


@pytest.mark.parametrize(
    'method', ['query_dns', 'gethostbyname', 'getaddrinfo']
)
async def test_synchronous_completion_keeps_future_and_native_state_consistent(
    integration: tuple[aiodns.DNSResolver, ControlledDNS], method: str
) -> None:
    resolver, _ = integration
    name = 'synchronous.onion' if method == 'query_dns' else '127.0.0.1'
    future = submit(resolver, method, name)
    cancelled = future.cancel()
    if cancelled:
        with pytest.raises(asyncio.CancelledError):
            await future
    elif method == 'query_dns':
        with pytest.raises(aiodns.error.DNSError) as exception:
            await future
        assert exception.value.args[0] == pycares.errno.ARES_ENOTFOUND
    else:
        assert await future
    await asyncio.sleep(0)
    assert resolver._channel.wait(0)
    assert not resolver._pending_queries


@pytest.mark.parametrize(
    'capability', ['query_cancellable', 'getaddrinfo_cancellable']
)
async def test_resolver_requires_patched_pycares_cancellation_capabilities(
    monkeypatch: pytest.MonkeyPatch, capability: str
) -> None:
    monkeypatch.setattr(pycares.Channel, capability, None)
    with pytest.raises(RuntimeError, match='patched pycares'):
        aiodns.DNSResolver(loop=asyncio.get_running_loop())
