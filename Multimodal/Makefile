################################################################################
# Makefile - in order to automate execution and params
# It is just like bash script.
# To execute this makefile, cmd: make from the home directory
################################################################################

# python path used for execution
PYTHON	= /home/nrk1787/miniconda3/envs/sl/bin/python
# PYTHON	= /home/nrk1787/miniconda3/envs/sl2/bin/python

BIN 	= ./bin
CONFIG = ./configs/gsl/mult_joeynmtenc.yaml
DEVICES = 0,1
WANDB = dryrun

VIZ	= ./visualize

################################################################################
.PHONY:	all train debug plot
################################################################################

train:
	CUDA_VISIBLE_DEVICES=$(DEVICES) $(PYTHON) $(BIN)/main.py $(CONFIG)

debug:
	$(PYTHON) $(BIN)/main.py $(CONFIG)

test:
	CUDA_VISIBLE_DEVICES=$(DEVICES) $(PYTHON) $(BIN)/inference.py $(CONFIG)

plot:
	$(PYTHON) $(VIZ)/app.py $(CONFIG)