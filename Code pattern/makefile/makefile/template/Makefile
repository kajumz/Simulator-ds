# Makefile for mock ML project

#SHELL := /bin/bash
#.PHONY: predict preprocess build-docker


# Make predictions
predict:
    #source /root/miniconda3/etc/profile.d/conda.sh && \
    #conda activate eta && \
	python src/predict.py
# run preprocess
preprocess:
    #source /root/miniconda3/etc/profile.d/conda.sh && \
    #conda activate eta-test && \
	python src/preprocess.py

#build-docker
build-docker:
	docker build -t mock_ml_image .