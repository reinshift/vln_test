from distutils.core import setup

from catkin_pkg.python_setup import generate_distutils_setup


setup_args = generate_distutils_setup(
    packages=["vln_sim", "vln_sim.rover"],
    package_dir={"": "src"},
)

setup(**setup_args)

