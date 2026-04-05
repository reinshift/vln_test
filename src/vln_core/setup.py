from distutils.core import setup

from catkin_pkg.python_setup import generate_distutils_setup


setup_args = generate_distutils_setup(
    packages=[
        "vln_core",
        "vln_core.config",
        "vln_core.eval",
        "vln_core.mission",
        "vln_core.planning",
        "vln_core.runtime",
        "vln_core.safety",
        "vln_core.world_model",
    ],
    package_dir={"": "src"},
)

setup(**setup_args)
