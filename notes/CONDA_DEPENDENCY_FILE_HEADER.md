#####################################################################################################################################################################################################
# Project:       Juniper
# Application:   juniper-data-client
# Purpose:       JuniperData Python HTTP Client Library
#
# Author:        Paul Calnon
# Version:       <X.Y.Z  Major, Minor, Point Version for juniper-data-client>
# Config Name:   conda_environment_ci.yaml
# Config Path:   Juniper/juniper-data-client/conf/
#
# Date:          2025-12-06
# Last Modified: <YYYY-MM-dd for current date>
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-<YYYY for current year> Paul Calnon
#
# Description:
#     This config file contains an automatically generated list of environment dependencies managed by conda / mamba for the juniper-data-client application.
#
#####################################################################################################################################################################################################
# Notes:
#     created-by: conda <YYYY.MM.dd for current date>
#     platform: linux-64
#     python: <Python Version>
#
#####################################################################################################################################################################################################
# References:
#
#     This file may be used to create the Juniper Project, juniper-data-client Application environment
#         with conda and miniforge3 using:
#     create env: conda create --name [env] --file [filename]
#         e.g., $ conda create --name JuniperPython --file juniper-data-client/conf/conda_environment_ci.yaml
#     Update env: conda env update --name [env] --file [filename]
#         e.g., $ conda env update --name JuniperPython --file juniper-data-client/conf/conda_environment_ci.yaml
#     Generate deps: conda list --explicit > [filename]
#         e.g., $ conda list -e >> juniper-data-client/conf/conda_environment_ci.yaml
#
#####################################################################################################################################################################################################
# TODO :
#
#####################################################################################################################################################################################################
# COMPLETED:
#
#####################################################################################################################################################################################################
name: juniper-data-client
channels:
  - conda-forge
  - pytorch
  - nvidia
  - plotly
  - RMG
  - numba
  - jasonb857
  - ehmoussi
  - konstantin-orangeqs
dependencies:
