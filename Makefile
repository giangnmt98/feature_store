SHELL = /bin/bash
PYTHON := python3
VENV_NAME = rec_featurestore_env
FEATURESTORE_FOLDER = featurestore
TEST_FOLDER = tests

# Environment
venv:
	${PYTHON} -m venv ${VENV_NAME} && \
	source ${VENV_NAME}/bin/activate && \
	${PYTHON} -m pip install pip setuptools wheel && \
	${PYTHON} -m pip install -e .[dev] && \
	pre-commit install

# Style
style:
	black ./${FEATURESTORE_FOLDER}/
	flake8 ./${FEATURESTORE_FOLDER}/
	${PYTHON} -m isort -rc ./${FEATURESTORE_FOLDER}/
	pylint --disable=R0913,R0903,R0902,R0914,W0718,R0917,R0801,C0411,R0915,R1710 ./${FEATURESTORE_FOLDER}/

test:
	${PYTHON} -m flake8 ./${FEATURESTORE_FOLDER}/
	${PYTHON} -m mypy ./${FEATURESTORE_FOLDER}/
	${PYTHON} -m pylint --disable=R0913,R0903,R0902,R0914,W0718,R0917,R0801,C0411,R0915,R1710 ./${FEATURESTORE_FOLDER}/
	CUDA_VISIBLE_DEVICES=""  ${PYTHON} -m pytest -s --durations=0 --disable-warnings ${TEST_FOLDER}/
