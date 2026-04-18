import sys
import types


def _install_bcrypt_stub() -> None:
    if "bcrypt" in sys.modules:
        return

    bcrypt_stub = types.ModuleType("bcrypt")
    bcrypt_stub.__version__ = "4.1.3"
    bcrypt_stub.__about__ = types.SimpleNamespace(__version__="4.1.3")
    bcrypt_stub.gensalt = lambda rounds=12: b"stub-salt"
    bcrypt_stub.hashpw = lambda password, salt: password + b"." + salt
    bcrypt_stub.checkpw = lambda password, hashed: True
    bcrypt_stub.kdf = lambda password, salt, desired_key_bytes, rounds, ignore_few_rounds=False: b"k" * desired_key_bytes
    sys.modules["bcrypt"] = bcrypt_stub


# Work around local Windows wheel issues where bcrypt may fail during import.
try:
    import bcrypt  # type: ignore # noqa: F401
except Exception:
    _install_bcrypt_stub()
