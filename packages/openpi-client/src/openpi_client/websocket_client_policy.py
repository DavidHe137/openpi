import asyncio
import logging
import time
from typing import Any, Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client
from websockets.asyncio.client import connect

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        pass


class AsyncWebsocketClientPolicy:
    """Async version of WebsocketClientPolicy for high-performance concurrent requests.

    This class uses async websockets with a fixed-size connection pool to enable true concurrent
    requests without blocking. Each request gets its own connection from the pool.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        num_connections: int = 100,
    ) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._server_metadata = None
        self._connection_pool: list[Any] = []
        self._pool_lock = asyncio.Lock()
        self._num_connections = num_connections

    async def connect(self) -> Dict:
        """Connect to the server and retrieve metadata."""
        results = await asyncio.gather(*[self._create_connection() for _ in range(self._num_connections)])
        self._connection_pool = [conn for conn, _ in results]
        self._server_metadata = results[0][1]
        return self._server_metadata

    async def _create_connection(self) -> Tuple[Any, Dict[str, Any]]:
        """Create a new websocket connection and retrieve metadata."""
        headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
        extra_headers = headers if headers else {}
        conn = await connect(self._uri, compression=None, max_size=None, additional_headers=extra_headers)
        metadata_bytes = await conn.recv()
        metadata = msgpack_numpy.unpackb(metadata_bytes)
        return conn, metadata

    async def _get_connection(self) -> Any:
        """Get a connection from the pool."""
        async with self._pool_lock:
            assert self._connection_pool, (
                "No connections left in pool. Either allocate more connections or reduce the number of concurrent requests."
            )
            return self._connection_pool.pop()

    async def _return_connection(self, conn: Any) -> None:
        """Return a connection to the pool."""
        async with self._pool_lock:
            self._connection_pool.append(conn)

    async def infer(self, obs: Dict) -> Dict:
        """Send an observation and receive an action asynchronously.

        Each request uses its own connection from the pool to avoid
        concurrent recv() conflicts.
        """
        if self._server_metadata is None:
            raise RuntimeError("Client not connected. Call connect() first.")

        conn = await self._get_connection()
        try:
            data = self._packer.pack(obs)
            await conn.send(data)
            response = await conn.recv()
            if isinstance(response, str):
                # we're expecting bytes; if the server sends a string, it's an error.
                raise RuntimeError(f"Error in inference server:\n{response}")
            result = msgpack_numpy.unpackb(response)
            await self._return_connection(conn)
            return result
        except Exception:
            # Don't return broken connections to pool
            await conn.close()
            raise

    async def close(self) -> None:
        """Close all connections in the pool."""
        async with self._pool_lock:
            for conn in self._connection_pool:
                await conn.close()
            self._connection_pool.clear()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
