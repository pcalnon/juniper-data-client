#!/usr/bin/env bash

# pytest -vvv --cov ./tests/*

export JUNIPER_DATA_BENCHMARK_FLAG=1
export JUNIPER_DATA_URL_PARAM="http://localhost:8100"

echo "JUNIPER_DATA_BENCHMARK=\"${JUNIPER_DATA_BENCHMARK_FLAG}\" JUNIPER_DATA_URL=\"${JUNIPER_DATA_URL_PARAM}\" pytest -vvv --cov ./tests/*"
JUNIPER_DATA_BENCHMARK="${JUNIPER_DATA_BENCHMARK_FLAG}" JUNIPER_DATA_URL="${JUNIPER_DATA_URL_PARAM}" pytest -vvv --cov ./tests/*


