# Administata

Basic administrative procedures.

## Merging

 * Push all branches for code review before merging to master.

 * The reviewer should test [code integrity](#code-integrity).

 * The author of the branch should not merge.

## Code Integrity

 * Use Pylint

 * Run autopep8

 * Use the package
   ["coverage"](https://pypi.python.org/pypi/coverage/3.7.1) to check
   test coverage

## Docs

 * Use Sphinx

 * All documentation with
   [Google Python doc styling](http://google-styleguide.googlecode.com/svn/trunk/pyguide.html#Comments).

Steps to install

    $ pip install sphinx
    $ pip install sphinx_bootstrap_theme
    $ pip install sphinxcontrib-napoleon
    $ cd doc/
    $ make

The documentation should be in `_build/html`.

## Branching

 * In general make an issue before a major branch and call the branch
   "issueXX-my_branch".

 * Use `Fix #XX`, when merging the branch if issues is fixed and
   `address #XX` for all commits.

## Commit Messages

Run tests before comitting. Use the following template:

```
# Header, 50 characters or less
#
# Links to tickets, fixes etc
#
# Main Message
#
# List of other changes
#
# Answer the following questions:
#
#   * Why is this change necessary?
#
#   * How does it address the issue?
#
#   * What side effects does this change have?
```
