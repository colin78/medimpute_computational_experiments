using DataFrames, CSV
data = CSV.read("syntheticPatientData.csv", missingstring="NA")

categorical!(data, [:Gender, :Race])

mp = IAI.get_mp(data)

# Proportion of the missing data to add to the dataset
pct = 0.1

mpValid = IAI.generate_mp(data, pct)
dataMissing = IAI.set_mp(data, mpValid)

# 1. Mean Impute
dataImputed = IAI.meanImpute(dataMissing)
meanMAE = IAI.get_mae(dataImputed, data, mpValid)

# 2. Linear Interpolation Impute
dataLinearImputed = IAI.linearImpute(dataMissing)
linearMAE = IAI.get_mae(dataLinearImputed, data, mpValid)

# 3. MedKNN
dataMedKNN, num_iter = IAI.medknnImpute(dataMissing; seed=1, max_x1_sum=100)
medknnMAE = IAI.get_mae(dataMedKNN, data, mpValid)

using Statistics
println("Mean Impute MAE: ", round(mean(meanMAE), digits=2))
println("Linear Impute MAE: ", round(mean(linearMAE), digits=2))
println("MedKNN Impute MAE: ", round(mean(medknnMAE), digits=2))