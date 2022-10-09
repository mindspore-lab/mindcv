# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import shutil
import sphinx_rtd_theme
import mindcv

sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------

project = 'mindcv'
copyright = '2022, mindcv contributors'
author = 'mindcv contributors'

version_file = '../../mindcv/version.py'
with open(version_file) as f:
    exec(compile(f.read(), version_file, 'exec'))
__version__ = locals()['__version__']
# The short X.Y version
version = __version__
# The full version, including alpha/beta/rc tags
release = __version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosectionlabel',
    'sphinx_markdown_tables',
    'myst_parser',
    'sphinx_copybutton',
    'sphinx.ext.autodoc.typehints',
]  # yapf: disable
autodoc_typehints = 'description'
myst_heading_anchors = 4

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# copy markdown files from outer directory
if not os.path.exists('./tutorials'):
    os.makedirs('./tutorials')
shutil.copy('../../tutorials/deployment.md', './tutorials/deployment.md')
shutil.copy('../../tutorials/finetune.md', './tutorials/finetune.md')
shutil.copy('../../tutorials/Inference.md', './tutorials/Inference.md')
shutil.copy('../../tutorials/learn_about_config.md', './tutorials/learn_about_config.md')
shutil.copy('../../tutorials/output_8_0.png', './tutorials/output_8_0.png')
shutil.copy('../../tutorials/output_11_0.png', './tutorials/output_11_0.png')
shutil.copy('../../tutorials/output_23_0.png', './tutorials/output_23_0.png')
shutil.copy('../../tutorials/output_30_0.png', './tutorials/output_30_0.png')
if not os.path.exists('./quick_start'):
    os.makedirs('./quick_start')
shutil.copy('../../quick_start.md', './quick_start/quick_start.md')

os.system('cp -R %s %s'% ('../../configs', './'))

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['css/readthedocs.css']

# -- Extension configuration -------------------------------------------------
# Ignore >>> when copying code
copybutton_prompt_text = r'>>> |\.\.\. '
copybutton_prompt_is_regexp = True
