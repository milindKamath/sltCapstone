################################################################################
# Makefile - in order to automate execution and params
# It is just like bash script.
# To execute this makefile, cmd: make from the home directory
################################################################################

# python path used for execution
PYTHON	= /home/nrk1787/miniconda3/envs/sl/bin/python
# PYTHON	= /home/nrk1787/miniconda3/envs/sl2/bin/python

BIN 	= ./bin
CONFIG	= ./configs/features.yaml
DEVICES = 0

################################################################################
.PHONY:	all add generate of mp
################################################################################

add:
	$(PYTHON) $(BIN)/add_path.py $(CONFIG)

generate:
	CUDA_VISIBLE_DEVICES=$(DEVICES) $(PYTHON) $(BIN)/generate.py $(CONFIG)