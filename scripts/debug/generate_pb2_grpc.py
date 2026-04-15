# Backward compatibility shim — re-exports all gRPC stubs from the split modules.
# New code should import from engine_pb2_grpc, resource_pb2_grpc, admin_pb2_grpc directly.
from engine_pb2_grpc import *  # noqa: F401,F403
from resource_pb2_grpc import *  # noqa: F401,F403
from admin_pb2_grpc import *  # noqa: F401,F403

from engine_pb2_grpc import EngineServiceStub
from resource_pb2_grpc import ResourceServiceStub
from admin_pb2_grpc import AdminServiceStub


class TextGenerationServiceStub:
    """Backward-compatible composite stub wrapping 3 service stubs."""

    def __init__(self, channel):
        self._engine = EngineServiceStub(channel)
        self._resource = ResourceServiceStub(channel)
        self._admin = AdminServiceStub(channel)

    def __getattr__(self, name):
        for stub in (self._engine, self._resource, self._admin):
            try:
                return getattr(stub, name)
            except AttributeError:
                continue
        raise AttributeError(f"No RPC method '{name}' found in any service")
