.PHONY: test
test:
	cd test; pytest test*.py

.PHONY: reformat
reformat:
	black *.py depanyzed/*.py test/*.py
