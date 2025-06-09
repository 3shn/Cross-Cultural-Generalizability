# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository implements the methodology from "A Causal Framework for Cross-Cultural Generalizability" (Deffner, Rohrer, & McElreath, 2022). It contains R scripts and Stan models for demographic standardization and causal effect transport across populations using multilevel regression with poststratification (MRP).

## Software Requirements

- R 4.0.3+ with packages: rstan (2.21.2+), rethinking (2.12)
- Stan MCMC engine with C++ compiler
- Installation: https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started

## Common Development Commands

```bash
# Run R scripts (from repository root)
Rscript "Example 1 Generalizing Description/DemoStandHouse.R"
Rscript "Example 1 Generalizing Description/DemoStandSim.R" 
Rscript "Example 2 Transportability/TransportHouse.R"

# Check Stan model syntax
rstan::stan_model(file = "path/to/model.stan", model_name = "test")
```

## Architecture

### Two Main Examples

**Example 1 - Generalizing Description**: Demographic standardization for trait comparisons
- `DemoStandHouse.R`: Real data analysis (Berlin vs Vanuatu prosocial behavior)
- `DemoStandSim.R`: Simulation study demonstrating sampling bias effects
- `model_empirical.stan`: Simple Bernoulli baseline model
- `model_MRpoststratification.stan`: MRP with Gaussian processes

**Example 2 - Transportability**: Causal effect transport across populations  
- `TransportHouse.R`: Experimental intervention transport analysis
- `model_basic.stan`: Fixed effects baseline model
- `model_transport.stan`: GP-based causal effect transport

### Statistical Framework

All models use hierarchical Bayesian inference with:
- **Individual level**: Bernoulli/binomial outcomes
- **Demographic strata**: Age × gender with Gaussian process smoothing
- **Population level**: Poststratification to target demographics

### Data Structure

- `data/*.csv`: Contains demographic data (Berlin-2020.csv, Vanuatu-2019.csv) and experimental data (Model_*.csv from House et al. 2020)
- Demographic files provide population age×gender distributions for poststratification weights
- Experimental files contain behavioral outcomes across multiple field sites

### Workflow Pattern

1. **Data prep**: Load behavioral + demographic data, create age bins
2. **Model fitting**: Compare empirical vs MRP/transport models in Stan
3. **Poststratification**: Weight predictions by target population demographics  
4. **Analysis**: Compare naive vs demographically-adjusted estimates

## Key Stan Model Features

- **Gaussian processes**: Handle smooth age effects with `cov_exp_quad()` 
- **Demographic indexing**: Age bins and gender combinations for stratification
- **Generated quantities**: Compute poststratified estimates and effect transport
- **Hyperpriors**: Careful priors on GP length scales and variance parameters

## Working with Models

- Stan files use array syntax (updated from deprecated format)
- Models expect specific data structure: N (observations), age_index, gender_index, population_index
- GP models require age_grid for prediction and demographic weight matrices