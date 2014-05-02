# Administata

Basic administrative procedures.

## Merging

 * Push all branches for code review before merging to master.

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

## Branching

 * In general make an issue before a major branch and call the branch
   "issueXX-my_branch".

 * Use `Fix #XX`, when merging the branch if issues is fixed and
   `address #XX` for all commits.

## Commit Messages

Use the following template:

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
