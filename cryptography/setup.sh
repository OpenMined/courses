#!/bin/sh

[ $# -eq 0 ] && { VENV_DIR=venv; }
[ $# -eq 1 ] && { VENV_DIR=$1; }


if [ ! -d "${VENV_DIR}" ]; then
  echo "creating virtual environment ${VENV_DIR}"
  python3 -m venv $VENV_DIR
else
  echo "${VENV_DIR} created previously"
fi

echo "activating virtual environment ${VENV_DIR}"
. $VENV_DIR/bin/activate
