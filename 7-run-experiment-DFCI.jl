function runExperimentDanaFarber(pct, method, maxCategory, gamma, seed;
    m = 5, rerun = false, NAind = true, NMAR = (gamma > 0))

    println("Dana Farber Experiment: Breast Cancer")
    println("Missing %: ", pct)
    println("Method: ", method)
    println("Max Category: ", maxCategory)
    println("Gamma: ", gamma)
    println("Seed: ", seed)
    println("Missing Data Indicators: ", NAind)
    println("NMAR: ", NMAR)

    Random.seed!(seed)

    # Step 1: Read in the data
    cancerType = "BREAST"
    fileName = "../Dana_Farber_data_processed/$(cancerType)_OPP.csv"
    # data = readDanaFarberData(fileName, NAind)[1:600,:];
    data = readDanaFarberData(fileName, NAind);
    data = data[data[:,:category] .<= maxCategory,:];

    features = 11:ncol(data) # indices of the features in the data
    indexPatientID = 1 # index of the patient ID in the data
    indexTime = 2 # index of time in the data
    indexOutcome = 3 # index of outcome variable in the data (3 => :case60 for mortality)
    fileMedParams = "../Dana_Farber_results_final/MedParams_$(pct)_maxCategory_$(maxCategory)_gamma_$(gamma)_seed_$(seed).csv"
    fileMAE = "../Dana_Farber_results_final/MAE_$(method)_$(pct)_maxCategory_$(maxCategory)_gamma_$(gamma)_seed_$(seed).csv"
    fileRMSE = "../Dana_Farber_results_final/RMSE_$(method)_$(pct)_maxCategory_$(maxCategory)_gamma_$(gamma)_seed_$(seed).csv"
    fileAUC = "../Dana_Farber_results_final/AUC_$(method)_$(pct)_maxCategory_$(maxCategory)_gamma_$(gamma)_seed_$(seed).csv"

    # Remove rows with missing Y-variable
    data = data[.!ismissing.(data[:,indexOutcome]),:];


    # Convert Y variable to 0, 1
    data[:,(end+1)] = 1.0 .- (1.0 * (data[:,indexOutcome] .== data[1,indexOutcome]));
    data = data[:,[1:2;(end);4:(end-1)]];

    # Step 2: Add in some percentage of missing data
    # for each feature column
    dataMissing, mpValid = getMissingPatternNMAR(data;
        features = features,
        pct = pct, γ = gamma, seed = seed,
        NAind = NAind);

    # Step 3: Impute the full data set and save MedParams
    dataImputed = imputation(dataMissing[:,1:110], method;
        features = 11:110, indexPatientID = indexPatientID,
        indexTime = indexTime,
        fileMedParams = fileMedParams,
        m = m);
    if NAind
        NAindFeatureNames = names(dataMissing)[111:end]
        if method == "mice"
            for i in 1:m
                # dataImputed[i][:,NAindFeatureNames] = dataMissing[:,NAindFeatureNames]
                dataImputed[i] = DataFrame(hcat(dataImputed[i],dataMissing[:,NAindFeatureNames]))
            end
        else
            dataImputed = DataFrame(hcat(dataImputed,dataMissing[:,NAindFeatureNames]))
        end
    end

    if method == "mice"
        dataSingleImputed = dataImputed[1];
    else
        dataSingleImputed = dataImputed;
    end

    dataSubset = data;
    dataImputedSubset = dataImputed;
    dataSingleImputedSubset = dataSingleImputed;
    mpValidSubset = mpValid;

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
        X = [Matrix{Float64}(undef, n,p) for _ in 1:m];
        for i in 1:m
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
        y = Vector(dataImputedSubset[:,indexOutcome]);

        # Step 6: Fit a regularized logistic regression model,
        # and save the results
        trainAUC, testAUC = fitRegLogReg(X, y; seed = seed)
        println("Train AUC: ", round(trainAUC, digits=4))
        println("Test AUC: ", round(testAUC, digits=4))
        CSV.write(fileAUC, DataFrame(trainAUC=trainAUC, testAUC=testAUC))
    end
end
