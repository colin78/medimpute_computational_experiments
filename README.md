# Overview
This is the codebase for the computational experiments for the paper "Imputation of Clinical Covariates in Time Series" by Dimitris Bertsimas, Agni Orfanoudaki, and Colin Pawlowski.  The purpose of this method, MedImpute, is to fill in missing values in healthcare datasets with a longitudinal time series structure.  This code is compatible with Julia version 1.0.5, available for download [here](https://julialang.org/downloads/).

# Datasets

In this paper, three real-world clinical datasets were used for the computational experiments to compare imputation methods: Framingham Heart Study, Parkinson's Progressive Markers Index, and electronic health record data from the Dana Farber Cancer Institute.  Access to the first two datasets can be requested from the following sources:

1. [Framingham Heart Study](https://framinghamheartstudy.org/fhs-for-researchers/data-available-overview/)
2. [Parkinson's Progressive Markers Index](https://www.ppmi-info.org/access-data-specimens/download-data/)

# Academic License and Installation

You must obtain an academic license and precompiled Julia system image in order to install and run the MedImpute software package.  Please email {agniorf,cpawlows}@mit.edu with the subject line "Request for MedImpute License" in order to request access.

# Documentation

Documentation for the MedImpute package is available [here](https://interpretableai.gitlab.io/DocumentationStaging/MedImpute/master/dev/).  In addition, [here](https://interpretableai.gitlab.io/DocumentationStaging/MedImpute/master/dev/example/) is an example of MedImpute being used to impute missing values in a synthetic EHR dataset.  

