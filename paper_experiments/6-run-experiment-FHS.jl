function runExperiment(dataSetName, pct, seed, method, gamma, NAind; 
    m = 5, rerun = false, NMAR = (gamma > 0))

		println("Framingham Heart Study")
	    println("Missing %: ", pct)
	    println("Method: ", method)
	    println("Dataset : ", dataSetName)
	    println("Gamma: ", gamma)
	    println("Seed: ", seed)
	    println("Missing Data Indicators: ", NAind)
	    println("NMAR: ", NMAR)

	    Random.seed!(seed)

	    fileName = "../FHS_data/$dataSetName.csv"
	    data = readFHSData(fileName);
	    features = 6:ncol(data) # indices of the features in the data
	    indexPatientID = 1 # index of the patient ID in the data
	    indexTime = 3 # index of time in the data
	    indexOutcome = 4 # index of outcome variable in the data
	    examNumber = 5 # index of outcome variable in the data
	    
	    if NMAR 
	        NMARstring = "_NMAR_$gamma"
	    else
	        NMARstring = ""
	    end

	    fileMedParams = "../FHS_results/MedParams_$(dataSetName)_$(pct)_seed_$(seed)$(NMARstring).csv"
	    fileMAE = "../FHS_results/MAE_$(dataSetName)_$(method)_$(pct)_seed_$(seed)$(NMARstring).csv"
	    fileRMSE = "../FHS_results/RMSE_$(dataSetName)_$(method)_$(pct)_seed_$(seed)$(NMARstring).csv"
	    fileAUC = "../FHS_results/AUC_$(dataSetName)_$(method)_$(pct)_seed_$(seed)$(NMARstring).csv"

	    # Step 2: Add in some percentage of missing data
	    # for each feature column
	    if NMAR 
	        println("Not Missing At Random Pattern")
	        dataMissing, mpValid = addNMARMissingPercentage_FHS(data;
	            gamma = gamma, features = features, seed = seed, NAind = NAind, percent = pct);    
	    else
	        dataMissing, mpValid = addMissingPercentage(data;
	            pct = pct, features = features, seed = seed, NAind = NAind);
	    end
	    # Step 3: Impute the full data set and save MedParams
	    # TODO: add functionality to reuse the MedParams
	    # if this seed has already been run.
	    # [sum(isna.(dataImputed[:,d])) for d in 1:ncol(dataImputed)]  
	    # [sum(isa(dataImputed[:,d], DataArrays.DataArray) && isnan.(dataImputed[:,d])) for d in 1:ncol(dataImputed)]  
	    # [typeof(dataImputed[:,d]) for d in 1:ncol(dataImputed)]
	    # x = dataMissing;
	    dataImputed = imputation(dataMissing, method;
	        features = features, indexPatientID = indexPatientID,
	        indexTime = indexTime,
	        fileMedParams = fileMedParams,
	        m = m);

    if method == "mice"
        dataSingleImputed = dataImputed[1];
    else
        dataSingleImputed = dataImputed;
    end

    mostRecentObservations = findall(data[:,:time_obs] .== 1);
    dataSubset = data[mostRecentObservations,:];
    if method == "mice"
        dataImputedSubset = [dataImputed[i][mostRecentObservations,:] for i in 1:m]
    else
        dataImputedSubset = dataImputed[mostRecentObservations,:]
    end
    dataSingleImputedSubset = dataSingleImputed[mostRecentObservations,:]
    mpValidSubset = mpValid[mostRecentObservations,:]
    

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
        trainAUC, testAUC = fitMultiRegLogReg(X, y; seed = seed)
        println("Train AUC: ", round(trainAUC, digits=4))
        println("Test AUC: ", round(testAUC, digits=4))
        CSV.write(fileAUC, DataFrame(trainAUC=trainAUC, testAUC=testAUC))
    else
        # Step 5: Convert data frame to numeric matrix
        X = getMatrix(dataImputedSubset, features);
        y = Vector(dataImputedSubset[:,indexOutcome])

        # Step 6: Fit a regularized logistic regression model,
        # and save the results
        trainAUC, testAUC = fitRegLogReg(X, y; seed = seed)
        println("Train AUC: ", round(trainAUC, digits=4))
        println("Test AUC: ", round(testAUC, digits=4))
        CSV.write(fileAUC, DataFrame(trainAUC=trainAUC, testAUC=testAUC))
    end
end
