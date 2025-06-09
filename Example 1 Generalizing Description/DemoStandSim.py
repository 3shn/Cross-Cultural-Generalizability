# %% [markdown]
# # Generalizing description: Cross-cultural comparisons and demographic standardization
# ## Simulation Example
# 
# Python conversion of DemoStandSim.R using cmdstanpy and arviz
# 
# The aim here is to simulate and analyze exemplary data sets generated from different processes:
# a) Disparities arise only from real demographic differences among populations (e.g. one is older)
# b) Disparities arise only from differences in sampling procedure (e.g. gender of researcher influences gender of volunteer participants)

# %%
import numpy as np
import pandas as pd
import cmdstanpy
import arviz as az
from scipy import stats
from scipy.special import expit  # equivalent to inv_logit
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# %% [markdown]
# ## Parameters and Setup

# %%
# Parameters
N = 500  # Sample size
age_range = np.arange(1, 91)  # Age range of participants

# Create "Real" cultural differences: (Demography-independent) Probabilities in both populations on logit scale
p_logit_culture = np.array([-3, -1.5])

# Effects of age and gender on prosociality
b_age = 0.04
b_gender = 2

# %% [markdown]
# ## a) Real population differences
# Create populations with different age distributions but same sampling probabilities

# %%

# Create population array [pop, sex, age]
D_popdiff = np.zeros((2, 2, len(age_range)))

# Exponential distribution (similar to many growing populations)
exp_dist = stats.expon.pdf(np.arange(0, 101, 1), scale=1/0.04)[age_range-1]
for i in range(2):
    D_popdiff[0, i, :] = exp_dist

# Skewed normal distribution (similar to many shrinking populations)
skew_dist = stats.skewnorm.pdf(np.arange(1, 101), a=1, loc=30, scale=50)[age_range-1]
for i in range(2):
    D_popdiff[1, i, :] = skew_dist

# Generate data
d = pd.DataFrame({
    'id': range(2*N),
    'soc_id': [1]*N + [2]*N,
    'age': np.nan,
    'gender': np.nan,
    'outcome': np.nan
})

# Simulate ages from above distributions
for pop_id in [1, 2]:
    mask = d['soc_id'] == pop_id
    d.loc[mask, 'age'] = np.random.choice(
        age_range, 
        size=N, 
        replace=True, 
        p=D_popdiff[pop_id-1, 0, :] / np.sum(D_popdiff[pop_id-1, 0, :])
    )

# Simulate Genders
p_male = 0.5
for pop_id in [1, 2]:
    mask = d['soc_id'] == pop_id
    d.loc[mask, 'gender'] = np.random.choice([1, 2], size=N, replace=True, p=[p_male, 1-p_male])

# Convert to integers
d['age'] = d['age'].astype(int)
d['gender'] = d['gender'].astype(int)

# Record demographic statistics of samples
SampleD_popdiff = np.zeros((2, 2, len(age_range)))
for pop_id in [1, 2]:
    for gender in [1, 2]:
        for i, age in enumerate(age_range):
            count = len(d[(d['soc_id'] == pop_id) & (d['age'] == age) & (d['gender'] == gender)])
            SampleD_popdiff[pop_id-1, gender-1, i] = count

# Generate observations
for i in range(2*N):
    soc_id = d.loc[i, 'soc_id']
    age = d.loc[i, 'age']
    gender = d.loc[i, 'gender']
    prob = expit(p_logit_culture[soc_id-2] + b_age*age + b_gender*(gender-1))
    d.loc[i, 'outcome'] = np.random.binomial(1, prob)

# Compute true expected prosociality in both populations
True_Value_popdiff = np.zeros(2)
for pop in range(2):
    expect_pos = 0
    total = 0
    for a in range(2):
        for b in range(len(age_range)):
            total += D_popdiff[pop, a, b]
            prob = expit(p_logit_culture[pop] + b_gender*a + b_age*(b+1))
            expect_pos += D_popdiff[pop, a, b] * prob
    True_Value_popdiff[pop] = expect_pos / total

# Create data lists for both populations
d1_popdiff = {
    'N': N,
    'MA': len(age_range),
    'gender': d[d['soc_id'] == 1]['gender'].values,
    'age': d[d['soc_id'] == 1]['age'].values,
    'outcome': d[d['soc_id'] == 1]['outcome'].values.astype(int)
}

d2_popdiff = {
    'N': N,
    'MA': len(age_range),
    'gender': d[d['soc_id'] == 2]['gender'].values,
    'age': d[d['soc_id'] == 2]['age'].values,
    'outcome': d[d['soc_id'] == 2]['outcome'].values.astype(int)
}

# Population demography of both sites: multiply by large number and convert to int
d1_popdiff['P_Pop'] = (D_popdiff[0, :, :] * 1e9).astype(int)
d1_popdiff['P_other'] = (D_popdiff[1, :, :] * 1e9).astype(int)

d2_popdiff['P_Pop'] = (D_popdiff[1, :, :] * 1e9).astype(int)
d2_popdiff['P_other'] = (D_popdiff[0, :, :] * 1e9).astype(int)

# %% [markdown]
# ## b) Differences in sampling of genders
# Create populations with same age distributions but different gender sampling probabilities

# %%

# Create population array [pop, sex, age] - same for both populations now
D_samplediff = np.zeros((2, 2, len(age_range)))

# Exponential distribution for both populations and genders
exp_dist = stats.expon.pdf(np.arange(0, 101, 1), scale=1/0.04)[age_range-1]
for i in range(2):
    for j in range(2):
        D_samplediff[i, j, :] = exp_dist

# Generate data with different gender sampling probabilities
d_sample = pd.DataFrame({
    'id': range(2*N),
    'soc_id': [1]*N + [2]*N,
    'age': np.nan,
    'gender': np.nan,
    'outcome': np.nan
})

# Simulate ages from above distributions
for pop_id in [1, 2]:
    mask = d_sample['soc_id'] == pop_id
    d_sample.loc[mask, 'age'] = np.random.choice(
        age_range, 
        size=N, 
        replace=True, 
        p=D_samplediff[pop_id-1, 0, :] / np.sum(D_samplediff[pop_id-1, 0, :])
    )

# Simulate Genders with different sampling probabilities
mask1 = d_sample['soc_id'] == 1
mask2 = d_sample['soc_id'] == 2
d_sample.loc[mask1, 'gender'] = np.random.choice([1, 2], size=N, replace=True, p=[0.3, 0.7])
d_sample.loc[mask2, 'gender'] = np.random.choice([1, 2], size=N, replace=True, p=[0.8, 0.2])

# Convert to integers
d_sample['age'] = d_sample['age'].astype(int)
d_sample['gender'] = d_sample['gender'].astype(int)

# Create demographic statistics of samples
SampleD_samplediff = np.zeros((2, 2, len(age_range)))
for pop_id in [1, 2]:
    for gender in [1, 2]:
        for i, age in enumerate(age_range):
            count = len(d_sample[(d_sample['soc_id'] == pop_id) & (d_sample['age'] == age) & (d_sample['gender'] == gender)])
            SampleD_samplediff[pop_id-1, gender-1, i] = count

# Generate observations
for i in range(2*N):
    soc_id = d_sample.loc[i, 'soc_id']
    age = d_sample.loc[i, 'age']
    gender = d_sample.loc[i, 'gender']
    prob = expit(p_logit_culture[soc_id-1] + b_age*age + b_gender*(gender-1))
    d_sample.loc[i, 'outcome'] = np.random.binomial(1, prob)

# Compute true expected prosociality in both populations
True_Value_samplediff = np.zeros(2)
for pop in range(2):
    expect_pos = 0
    total = 0
    for a in range(2):
        for b in range(len(age_range)):
            total += D_samplediff[pop, a, b]
            prob = expit(p_logit_culture[pop] + b_gender*a + b_age*(b+1))
            expect_pos += D_samplediff[pop, a, b] * prob
    True_Value_samplediff[pop] = expect_pos / total

# Prepare lists for stan
d1_samplediff = {
    'N': N,
    'MA': len(age_range),
    'gender': d_sample[d_sample['soc_id'] == 1]['gender'].values,
    'age': d_sample[d_sample['soc_id'] == 1]['age'].values,
    'outcome': d_sample[d_sample['soc_id'] == 1]['outcome'].values.astype(int)
}

d2_samplediff = {
    'N': N,
    'MA': len(age_range),
    'gender': d_sample[d_sample['soc_id'] == 2]['gender'].values,
    'age': d_sample[d_sample['soc_id'] == 2]['age'].values,
    'outcome': d_sample[d_sample['soc_id'] == 2]['outcome'].values.astype(int)
}

# Population demography
d1_samplediff['P_Pop'] = (D_samplediff[0, :, :] * 1e9).astype(int)
d1_samplediff['P_other'] = (D_samplediff[1, :, :] * 1e9).astype(int)

d2_samplediff['P_Pop'] = (D_samplediff[1, :, :] * 1e9).astype(int)
d2_samplediff['P_other'] = (D_samplediff[0, :, :] * 1e9).astype(int)

# %% [markdown]
# ## Fit Stan models
# Compile and fit both empirical and MRP models for both scenarios

# %%
print("Compiling Stan models...")

# Compile models
model_empirical = cmdstanpy.CmdStanModel(stan_file='Example 1 Generalizing Description/model_empirical.stan')
model_mrp = cmdstanpy.CmdStanModel(stan_file='Example 1 Generalizing Description/model_MRpoststratification.stan')

print("Fitting models for population differences scenario...")

# Fit empirical models for population differences
m_popdiff1_basic = model_empirical.sample(
    data=d1_popdiff, 
    iter_sampling=2500, 
    iter_warmup=2500,
    chains=4, 
    parallel_chains=4,
    seed=1,
    adapt_delta=0.99,
    max_treedepth=13
)

m_popdiff2_basic = model_empirical.sample(
    data=d2_popdiff, 
    iter_sampling=2500, 
    iter_warmup=2500,
    chains=4, 
    parallel_chains=4,
    seed=1,
    adapt_delta=0.99,
    max_treedepth=13
)

# Fit MRP models for population differences
m_popdiff1 = model_mrp.sample(
    data=d1_popdiff, 
    iter_sampling=2500, 
    iter_warmup=2500,
    chains=4, 
    parallel_chains=4,
    seed=1,
    adapt_delta=0.99,
    max_treedepth=13
)

m_popdiff2 = model_mrp.sample(
    data=d2_popdiff, 
    iter_sampling=2500, 
    iter_warmup=2500,
    chains=4, 
    parallel_chains=4,
    seed=1,
    adapt_delta=0.99,
    max_treedepth=13
)

print("Fitting models for sampling differences scenario...")

# Fit empirical models for sampling differences
m_samplediff1_basic = model_empirical.sample(
    data=d1_samplediff, 
    iter_sampling=2500, 
    iter_warmup=2500,
    chains=4, 
    parallel_chains=4,
    seed=4,
    adapt_delta=0.99,
    max_treedepth=13
)

m_samplediff2_basic = model_empirical.sample(
    data=d2_samplediff, 
    iter_sampling=2500, 
    iter_warmup=2500,
    chains=4, 
    parallel_chains=4,
    seed=4,
    adapt_delta=0.99,
    max_treedepth=13
)

# Fit MRP models for sampling differences
m_samplediff1 = model_mrp.sample(
    data=d1_samplediff, 
    iter_sampling=2500, 
    iter_warmup=2500,
    chains=4, 
    parallel_chains=4,
    seed=4,
    adapt_delta=0.99,
    max_treedepth=13
)

m_samplediff2 = model_mrp.sample(
    data=d2_samplediff, 
    iter_sampling=2500, 
    iter_warmup=2500,
    chains=4, 
    parallel_chains=4,
    seed=4,
    adapt_delta=0.99,
    max_treedepth=13
)

print("Model fitting complete!")

# %% [markdown]
# ## Convert to ArviZ InferenceData objects

# %%

def cmdstan_to_arviz(fit, prior=None):
    """Convert cmdstanpy fit to arviz InferenceData"""
    return az.from_cmdstanpy(fit, prior=prior)

# Convert all fits to ArviZ
idata_popdiff1_basic = cmdstan_to_arviz(m_popdiff1_basic)
idata_popdiff2_basic = cmdstan_to_arviz(m_popdiff2_basic)
idata_popdiff1 = cmdstan_to_arviz(m_popdiff1)
idata_popdiff2 = cmdstan_to_arviz(m_popdiff2)

idata_samplediff1_basic = cmdstan_to_arviz(m_samplediff1_basic)
idata_samplediff2_basic = cmdstan_to_arviz(m_samplediff2_basic)
idata_samplediff1 = cmdstan_to_arviz(m_samplediff1)
idata_samplediff2 = cmdstan_to_arviz(m_samplediff2)

# %% [markdown]
# ## Plotting using ArviZ
# Create publication-quality plots showing empirical vs poststratified estimates

# %%
print("Creating plots...")

# Plot posterior comparisons for population differences
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Population Differences: Empirical vs Poststratified Estimates', fontsize=16)

# Population 1 - empirical vs poststratified to Pop 2
ax = axes[0, 0]
az.plot_posterior(
    idata_popdiff1_basic.posterior['p'].values.flatten(),
    transform=expit,
    ax=ax,
    color='blue',
    alpha=0.7,
    label='Empirical'
)
az.plot_posterior(
    idata_popdiff1.posterior['p_other'].values.flatten(),
    ax=ax,
    color='lightblue',
    alpha=0.7,
    label='Poststr. to Pop II'
)
ax.axvline(True_Value_popdiff[0], color='black', linestyle='--', label='True Value')
ax.set_title('Population I')
ax.legend()

# Population 2 - empirical vs poststratified to Pop 1
ax = axes[0, 1]
az.plot_posterior(
    idata_popdiff2_basic.posterior['p'].values.flatten(),
    transform=expit,
    ax=ax,
    color='red',
    alpha=0.7,
    label='Empirical'
)
az.plot_posterior(
    idata_popdiff2.posterior['p_other'].values.flatten(),
    ax=ax,
    color='lightcoral',
    alpha=0.7,
    label='Poststr. to Pop I'
)
ax.axvline(True_Value_popdiff[1], color='black', linestyle='--', label='True Value')
ax.set_title('Population II')
ax.legend()

# Sampling differences plots
ax = axes[1, 0]
az.plot_posterior(
    idata_samplediff1_basic.posterior['p'].values.flatten(),
    transform=expit,
    ax=ax,
    color='green',
    alpha=0.7,
    label='Empirical'
)
az.plot_posterior(
    idata_samplediff1.posterior['p_pop'].values.flatten(),
    ax=ax,
    color='lightgreen',
    alpha=0.7,
    label='Poststr. to Pop I'
)
ax.axvline(True_Value_samplediff[0], color='black', linestyle='--', label='True Value')
ax.set_title('Sampling Diff - Population I')
ax.legend()

ax = axes[1, 1]
az.plot_posterior(
    idata_samplediff2_basic.posterior['p'].values.flatten(),
    transform=expit,
    ax=ax,
    color='orange',
    alpha=0.7,
    label='Empirical'
)
az.plot_posterior(
    idata_samplediff2.posterior['p_pop'].values.flatten(),
    ax=ax,
    color='moccasin',
    alpha=0.7,
    label='Poststr. to Pop II'
)
ax.axvline(True_Value_samplediff[1], color='black', linestyle='--', label='True Value')
ax.set_title('Sampling Diff - Population II')
ax.legend()

plt.tight_layout()
plt.savefig('demographic_standardization_results.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### Age-specific predictions

# %%
# Plot age curves
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Age-Specific Predictions', fontsize=16)

# Function to plot age curves
def plot_age_curves(idata, ax, title, d_dict):
    pred_p_m = idata.posterior['pred_p_m'].values
    pred_p_f = idata.posterior['pred_p_f'].values
    
    ages = np.arange(1, d_dict['MA'] + 1)
    
    # Male predictions
    m_median = np.median(pred_p_m, axis=(0, 1))
    m_lower = np.percentile(pred_p_m, 5, axis=(0, 1))
    m_upper = np.percentile(pred_p_m, 95, axis=(0, 1))
    
    # Female predictions
    f_median = np.median(pred_p_f, axis=(0, 1))
    f_lower = np.percentile(pred_p_f, 5, axis=(0, 1))
    f_upper = np.percentile(pred_p_f, 95, axis=(0, 1))
    
    # Check which ages were observed
    observed_ages_m = np.unique(d_dict['age'][d_dict['gender'] == 1])
    observed_ages_f = np.unique(d_dict['age'][d_dict['gender'] == 2])
    
    # Plot male data
    for i, age in enumerate(ages):
        alpha = 1.0 if age in observed_ages_m else 0.3
        ax.plot(age, m_median[i], 'o', color='blue', alpha=alpha, markersize=4)
        ax.plot([age, age], [m_lower[i], m_upper[i]], '-', color='blue', alpha=alpha, linewidth=2)
    
    # Plot female data
    for i, age in enumerate(ages):
        alpha = 1.0 if age in observed_ages_f else 0.3
        ax.plot(age, f_median[i], '^', color='red', alpha=alpha, markersize=4)
        ax.plot([age, age], [f_lower[i], f_upper[i]], '-', color='red', alpha=alpha, linewidth=2)
    
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    if title.endswith('I'):
        ax.legend(['Male', 'Female'], loc='upper left')

# Plot age curves for all scenarios
plot_age_curves(idata_popdiff1, axes[0, 0], 'Population Differences - Population I', d1_popdiff)
plot_age_curves(idata_popdiff2, axes[0, 1], 'Population Differences - Population II', d2_popdiff)
plot_age_curves(idata_samplediff1, axes[1, 0], 'Sampling Differences - Population I', d1_samplediff)
plot_age_curves(idata_samplediff2, axes[1, 1], 'Sampling Differences - Population II', d2_samplediff)

plt.tight_layout()
plt.savefig('age_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print("Analysis complete! Plots saved as 'demographic_standardization_results.png' and 'age_curves.png'")

# %% [markdown]
# ## Summary Results

# %%
# Print summary statistics
print("\n" + "="*50)
print("SUMMARY RESULTS")
print("="*50)

print("\nPopulation Differences:")
print(f"True values: Pop I = {True_Value_popdiff[0]:.3f}, Pop II = {True_Value_popdiff[1]:.3f}")

emp1 = np.median(expit(idata_popdiff1_basic.posterior['p'].values))
post1 = np.median(idata_popdiff1.posterior['p_other'].values)
emp2 = np.median(expit(idata_popdiff2_basic.posterior['p'].values))
post2 = np.median(idata_popdiff2.posterior['p_other'].values)

print(f"Pop I - Empirical: {emp1:.3f}, Poststratified: {post1:.3f}")
print(f"Pop II - Empirical: {emp2:.3f}, Poststratified: {post2:.3f}")

print("\nSampling Differences:")
print(f"True values: Pop I = {True_Value_samplediff[0]:.3f}, Pop II = {True_Value_samplediff[1]:.3f}")

emp1_s = np.median(expit(idata_samplediff1_basic.posterior['p'].values))
post1_s = np.median(idata_samplediff1.posterior['p_pop'].values)
emp2_s = np.median(expit(idata_samplediff2_basic.posterior['p'].values))
post2_s = np.median(idata_samplediff2.posterior['p_pop'].values)

print(f"Pop I - Empirical: {emp1_s:.3f}, Poststratified: {post1_s:.3f}")
print(f"Pop II - Empirical: {emp2_s:.3f}, Poststratified: {post2_s:.3f}")