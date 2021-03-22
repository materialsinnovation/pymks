import sys
import os
import alabaster

html_theme_path = [alabaster.get_path()]
extensions = ['alabaster']
html_theme = 'alabaster'

sys.path.insert(0, os.path.abspath('..'))

root_doc = 'index'

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.mathjax',
              'sphinx.ext.autosummary',
              'sphinx.ext.napoleon',
              'nbsphinx',
              'm2r2',
              'sphinxcontrib.jinja',
              'IPython.sphinxext.ipython_console_highlighting'
]

templates_path = ['_templates']

master_doc = 'contents'

project = u'pymks'
copyright = u'2021, Daniel Wheeler'


import pymks
version = pymks.__version__
release = pymks.__version__

exclude_patterns = ['_build', '**.ipynb_checkpoints']
autoclass_content = 'both'

pygments_style = 'alabaster.support.Alabaster'

html_sidebars = {
    '**': [
        'about.html',
        'release.html',
        'navigation.html',
    ]
}

html_theme_options = {
    'logo': 'pymks_logo.svg',
    'github_user': 'materialsinnovation',
    'github_repo': 'pymks',
    'github_button': True,
    'sidebar_collapse': True,
    'sidebar_width': '200px',
    'page_width': '1050px',
    'fixed_sidebar': True
}

html_title = "PyMKS"
html_short_title = "PyMKS"
html_favicon = '_static/pymks_logo.ico'
html_static_path = ['_static']
html_css_files = [
    'pymks.css'
]
html_additional_pages = {}
html_show_copyright = False
htmlhelp_basename = 'pymksdoc'

source_suffix = ['.rst', '.md']

def url_resolver(url):
    """Resolve url for both documentation and Github online.

    If the url is an IPython notebook links to the correct path.

    Args:
      url: the path to the link (not always a full url)

    Returns:
      a local url to either the documentation or the Github

    """
    if url[-6:] == '.ipynb':
        return url[4:-6] + '.html'
    else:
        return url

import shutil, os, glob

rst_directory = 'rst'
notebook_directory = os.path.join(rst_directory, 'notebooks')

for directory in [rst_directory, notebook_directory]:
    if not os.path.exists(directory):
        os.makedirs(directory)

files_to_copy = (
    'README.md',
    'notebooks/*.ipynb'
)

for fpath in files_to_copy:
    for fpath_glob in glob.glob(os.path.join('..', fpath)):
        fpath_glob_ = '/'.join(fpath_glob.split('/')[1:])
        shutil.copy(fpath_glob, os.path.join(rst_directory, fpath_glob_))
