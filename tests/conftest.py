"""Keep the suite away from the real ~/.config/dictdact and ~/.local/share/dictdact.

Without this, test_server's Server.start() overwrites the RUNNING app's ui-url file with a
port/token that die when the test ends — `dictdact ui` (and anything else reading that file)
then points at a dead server until the app is relaunched.
"""
import atexit
import os
import shutil
import tempfile

_tmp = tempfile.mkdtemp(prefix="dictdact-tests-")
atexit.register(shutil.rmtree, _tmp, True)
# Hard assignment, not setdefault: a dev with these exported (pointing at the live app)
# is exactly who needs the isolation.
os.environ["DICTDACT_CONFIG_DIR"] = os.path.join(_tmp, "config")
os.environ["DICTDACT_DATA_DIR"] = os.path.join(_tmp, "data")
