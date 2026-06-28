"""Gate: the inference server accepts connections as soon as it is BOUND, before
it serves — so `mill start` no longer refuses connections while the model loads.
`pixi run test-server-bind`. Pure CPU (loopback sockets), no GPU / weights.

server.mojo's main() binds + listen()s up front, THEN loads weights, THEN serve()s;
a client that connects during the ~10-15s load is completed by the kernel into the
listen backlog instead of getting ConnectionRefused (its request just waits until
accept() runs). This pins that property: a bound, not-yet-accepting TcpListener
still completes client connects — exactly what HttpServer.bind() relies on.
"""

from flare.tcp.listener import TcpListener
from flare.tcp.stream import TcpStream
from flare.net.address import SocketAddr


def main() raises:
    var all_ok = True

    # Bind to an ephemeral port (0 -> OS-assigned). TcpListener.bind() calls
    # listen(), exactly like HttpServer.bind() does in server.mojo's main().
    var lis = TcpListener.bind(SocketAddr.localhost(0))
    var port = lis.local_addr().port
    print(
        "  bound a listener on 127.0.0.1:",
        port,
        " (never accept()/serve() below)",
        sep="",
    )

    # No accept()/serve() is ever called — yet clients can connect, because the
    # kernel completes the handshake into the listen backlog. Queue several to show
    # they're accepted (not refused) before the "server" is ready.
    var n_ok = 0
    var i = 0
    while i < 4:
        try:
            var s = TcpStream.connect(lis.local_addr())
            n_ok += 1
            _ = (
                s^
            )  # close immediately; the connection was already accepted by the kernel
        except e:
            if i == 0:
                print("  (connect error: ", e, ")", sep="")
        i += 1
    var connects_ok = n_ok == 4
    print(
        "["
        + ("PASS" if connects_ok else "FAIL")
        + "] "
        + String(n_ok)
        + "/4 clients connect to a bound, not-yet-serving listener"
    )
    all_ok = all_ok and connects_ok

    print()
    if all_ok:
        print("ALL CHECKS PASSED")
    else:
        print("CHECKS FAILED")
        raise Error(
            "test-server-bind: a connection to a bound listener was refused"
        )
