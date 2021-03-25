.SILENT:
.DEFAULT_GOAL := list

AUTHOR := Sebastia Agramunt Puig
PROJECT := OpenMined's Cryptography Lesson (Author ${AUTHOR})

CODE_PATH=src

COLOR_RESET = \033[0m
COLOR_COMMAND = \033[36m
COLOR_YELLOW = \033[33m
COLOR_GREEN = \033[32m
COLOR_RED = \033[31m

.PHONY: bootstrap # : install all requirements needed
bootstrap:
	pip install --upgrade pip
	pip install -r ./requirements.txt

.PHONY: run # : run jupyter lab
run:
	jupyter lab --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token='' --NotebookApp.password=''


.PHONY: list # : Makefile command list
list:
	printf "\n${COLOR_YELLOW}${PROJECT}\n-------------------------------------------------------------------\n${COLOR_RESET}"
	@grep '^.PHONY: .* #' Makefile | sed 's/\.PHONY: \(.*\) # \(.*\)/\1 \2/' | expand -t20