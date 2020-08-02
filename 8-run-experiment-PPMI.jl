function runExperiment(dataSetName, pct, seed, method, gamma, NAind; 
    m = 5, rerun = false, NMAR = (gamma > 0))

	println("Parkinson Disease Study")
    println("Missing %: ", pct)
    println("Method: ", method)
    println("Dataset : ", dataSetName)
    println("Gamma: ", gamma)
    println("Seed: ", seed)
    println("Missing Data Indicators: ", NAind)
    println("NMAR: ", NMAR)

    Random.seed!(seed)

    fileName = "../Parkinson_data/$dataSetName.csv"
    data = readPDData(fileName);
    deletecols!(data,:updrs_totscore_on);
    deletecols!(data,:symptom6);

    features = 4:ncol(data) # indices of the features in the data
    indexPatientID = 1 # index of the patient ID in the data
    indexTime = 2 # index of time in the data
    indexOutcome = 3 # index of outcome variable in the data
    
    if NMAR 
        NMARstring = "_NMAR_$gamma"
    else
        NMARstring = ""
    end

    fileMedParams = "../PD_results/MedParams_$(dataSetName)_$(pct)_seed_$(seed)$(NMARstring).csv"
    fileMAE = "../PD_results/MAE_$(dataSetName)_$(method)_$(pct)_seed_$(seed)$(NMARstring).csv"
    fileRMSE = "../PD_results/RMSE_$(dataSetName)_$(method)_$(pct)_seed_$(seed)$(NMARstring).csv"
    fileMAE_pred = "../PD_results/MAE_pred_$(dataSetName)_$(method)_$(pct)_seed_$(seed)$(NMARstring).csv"

    # Step 2: Add in some percentage of missing data
    # for each feature column
    if NMAR 
        println("Not Missing At Random Pattern")
        dataMissing, mpValid = getMissingPatternNMAR(data;
                    features = features,
                     pct = pct, γ = gamma, seed = seed,
                    NAind = NAind);
    else
        dataMissing, mpValid = addMissingPercentage(data;
            pct = pct, features = features, seed = seed, NAind = NAind);
    end

    # Step 3: Impute the full data set and save MedParams
    dataImputed = imputation(dataMissing, method;
        features = features, indexPatientID = indexPatientID,
        indexTime = indexTime,
        fileMedParams = fileMedParams,
        m = m);

    # dataImputed = X_imputed;
    if method == "mice"
        dataSingleImputed = dataImputed[1];
    else
        dataSingleImputed = dataImputed;
    end

    mostRecentObservations = findall(data[:,:Times] .> 900);
    dataSubset = data[mostRecentObservations,:];
    if method == "mice"
        dataImputedSubset = [dataImputed[i][mostRecentObservations,:] for i in 1:m]
    else
        dataImputedSubset = dataImputed[mostRecentObservations,:]
    end
    dataSingleImputedSubset = dataSingleImputed[mostRecentObservations,:]
    mpValidSubset = BitArray(mpValid[mostRecentObservations,:])
    
    # Step 4: Compute the MAE on the data subset
    # and save values for each column
    resultΜΑΕ = getMAE(dataSubset, dataSingleImputedSubset, mpValidSubset;
        features = features,
        fileMAE = fileMAE);
    resultRMSE = getRMSE(dataSubset, dataSingleImputedSubset, mpValidSubset;
        features = features,
        fileRMSE = fileRMSE);
    println("Total MAE: ", round(resultΜΑΕ[1,:Total],digits=4))
    println("Total RMSE: ", round(resultRMSE[1,:Total],digits=4))


    if method == "mice"
        # Step 5: Convert data frame to numeric tensor
        n, p = size(dataSubset[:,features])
        X = [Matrix{Float64}(undef,n,p) for _ in 1:m];
        for i in 1:m
            # i = 1
            # dataImputed = dataImputedSubset[i];
            X[i] = getMatrix(dataImputedSubset[i], features);
        end
        y = Vector(dataSubset[:,indexOutcome])

        # Step 6: Fit a regularized logistic regression model,
        # and save the results
        trainMAE, testMAE  = fitMultiRegLinReg(X, y; seed = seed)
        println("Train MAE: ", round(trainMAE, digits=4))
        println("Test MAE: ", round(testMAE, digits=4))
        CSV.write(fileMAE_pred, DataFrame(trainMAE=trainMAE, testMAE=testMAE))
    else
        # Step 5: Convert data frame to numeric matrix
        X = getMatrix(dataImputedSubset, features);
        y = Vector(dataImputedSubset[:,indexOutcome])

        # Step 6: Fit a regularized logistic regression model,
        # and save the results
        trainMAE, testMAE = fitRegLinReg(X, y; seed = seed)
        println("Train MAE: ", round(trainMAE, digits=4))
        println("Test MAE: ", round(testMAE, digits=4))
        CSV.write(fileMAE_pred, DataFrame(trainMAE=trainMAE, testMAE=testMAE))
    end
end
