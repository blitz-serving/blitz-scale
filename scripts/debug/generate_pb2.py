# Backward compatibility shim — re-exports all proto types from the split modules.
# New code should import from common_pb2, engine_pb2, resource_pb2, admin_pb2 directly.
from common_pb2 import *  # noqa: F401,F403
from engine_pb2 import *  # noqa: F401,F403
from resource_pb2 import *  # noqa: F401,F403
from admin_pb2 import *  # noqa: F401,F403
