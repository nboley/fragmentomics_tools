import os
import shutil
import subprocess
import tempfile
import random
from contextlib import contextmanager

import pytest


@contextmanager
def environment_variables(**kwargs):
    old_env_vars = {key: os.environ.get(key) for key in kwargs if key in os.environ}
    os.environ.update(kwargs)
    yield
    # delete all of the new env variables
    for key in kwargs:
        del os.environ[key]
    # re-add the old variables
    os.environ.update(old_env_vars)


def random_string(length):
    return "".join([random.choice(string.ascii_letters + string.digits) for n in range(length)])


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")
    parser.addoption("--run-extra-slow", action="store_true", default=False, help="run extra slow tests")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    if not config.getoption("--run-extra-slow"):
        # --runslow given in cli: do not skip slow tests
        skip_extra_slow = pytest.mark.skip(reason="need --run-extra-slow option to run")
        for item in items:
            if "extra_slow" in item.keywords:
                item.add_marker(skip_extra_slow)


@pytest.fixture()
def cleandir():
    oldpath = os.getcwd()
    newpath = tempfile.mkdtemp()
    os.chdir(newpath)
    yield newpath

    os.chdir(oldpath)
    shutil.rmtree(newpath)


@contextmanager
def gpu_test_env():
    ld_library_path = os.environ.get("LD_LIBRARY_PATH")
    with environment_variables(
        CUDA_VISIBLE_DEVICES="2",
        LD_LIBRARY_PATH=f"/usr/local/cuda/extras/CUPTI/lib64:{ld_library_path}",
        KERAS_BACKEND="tensorflow",
    ):
        yield


# from fbio.constants import FBIO_S3_TEST_BUCKET
#@pytest.fixture()
#def clean_s3_prefix():
#    remote_datastore_prefix = f"s3://{FBIO_S3_TEST_BUCKET}/{random_string(16)}"
#    yield remote_datastore_prefix
#    subprocess.check_call(f"aws s3 rm --recursive {remote_datastore_prefix}".split())
