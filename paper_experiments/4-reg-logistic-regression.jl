using RCall

function getMatrix(dataImputed, features)
    numericFeatures = Int[]
    categoricFeatures = Int[]

    for d in features
        if isa(dataImputed[1,d], Number)
            push!(numericFeatures, d)
        else
            push!(categoricFeatures, d)
        end
    end

    dataNumeric = deepcopy(dataImputed[:,numericFeatures]);
    dataNumericMatrix = Matrix(dataNumeric)
    numCategoricFeatures = 0
    for d in categoricFeatures
        columnValues = dataImputed[:,d]
        uniqueValues = unique(columnValues)
        if length(uniqueValues) >= 2
            numCategoricFeatures += length(uniqueValues) - 1
        end
    end

    dataCategoricMatrix = zeros(nrow(dataImputed), numCategoricFeatures)

    i = 1
    # d = 27
    for d in categoricFeatures
        columnValues = dataImputed[:,d]
        uniqueValues = unique(columnValues)
        if length(uniqueValues) < 2
            # Do not add this column to the categoric matrix
        else
            # Add in K-1 columns to the categoric matrix
            # for each column, where K is the number of unique
            # classes in that column
            # val = uniqueValues[1]
            for val in uniqueValues[1:(end-1)]
                # println(i)
                dataCategoricMatrix[:,i] = 1.0 * (columnValues .== val)
                i += 1
            end
        end
    end

    dataMatrix = [dataNumericMatrix dataCategoricMatrix]

    return(dataMatrix)
end


R"library(glmnet)"
R"library(ROCR)"

function fitRegLogReg(X, y; seed = 1)
    for j in 1:size(X,2)
        if size(X[ismissing.(X[:,j]),j], 1)>0
            X[ismissing.(X[:,j]),j] .= mean(filter(!ismissing, X[:,j]))
        end
        if size(X[isnan.(X[:,j]),j], 1)>0
            X[isnan.(X[:,j]),j] .= mean(filter(!isnan, X[:,j]))
        end
    end

    train, test = splitData(y, SplitRatio = 0.75, seed = seed);
    xTrain = X[train,:]
    yTrain = y[train]
    xTest = X[test,:]
    yTest = y[test]

    # R"library(pROC)"
   
    R"set.seed($(seed))"
    R"lasso <- cv.glmnet($xTrain, $yTrain, family = 'binomial')"
    R"pred_train <- predict(lasso, newx = $xTrain, s = 'lambda.1se')"
    R"pred_test <- predict(lasso, newx = $xTest, s = 'lambda.1se')"

    # Training AUC
    R"pred_obj <- prediction(pred_train, $yTrain)";
    R"perf <- performance(pred_obj, 'tpr', 'fpr')";
    trainAUC = rcopy(R"performance(pred_obj, 'auc')@y.values[[1]]")

    # Testing AUC
    R"pred_obj <- prediction(pred_test, $yTest)";
    R"perf <- performance(pred_obj, 'tpr', 'fpr')";
    testAUC = rcopy(R"performance(pred_obj, 'auc')@y.values[[1]]")

    if trainAUC<0.5
        # Training AUC
        #R"roc_obj <- roc($yTrain, pred_train)"
        #trainAUC = rcopy(R"auc(roc_obj)")
    
        # Testing AUC
        #R"roc_obj <- roc($yTest, pred_test)"
        #testAUC = rcopy(R"auc(roc_obj)")
        trainAUC = 1 - trainAUC
        testAUC = 1 - testAUC
    end

    return(trainAUC, testAUC)
end

function fitMultiRegLogReg(X, y; seed = 1)

    for i in 1:size(X,1)
        for j in 1:size(X,2)
            if size(X[i][isnan.(X[i][:,j]),j], 1)>0
                X[i][isnan.(X[i][:,j]),j] .= mean(filter(!isnan, X[i][:,j]))
            end
        end
    end
    m = length(X)
    n, p = size(X[1])
    train, test = splitData(y, SplitRatio = 0.75, seed = seed);
    yTrain = y[train]
    yTest = y[test]
    predTrain = [Vector{Float64}() for _ in 1:m]
    predTest = [Vector{Float64}() for _ in 1:m]

    for i in 1:m
        xTrain = X[i][train,:]
        xTest = X[i][test,:]
        R"set.seed($(seed))"
        R"lasso <- cv.glmnet($xTrain, $yTrain, family = 'binomial')"
        predTrain[i] = rcopy(R"predict(lasso, newx = $xTrain, s = 'lambda.1se')")[:]
        predTest[i] = rcopy(R"predict(lasso, newx = $xTest, s = 'lambda.1se')")[:]
    end
    
    predTrainAvg = mean(hcat(predTrain...), dims=2)[:]
    predTestAvg = mean(hcat(predTest...), dims=2)[:]

    # Training AUC
    R"set.seed($(seed))"
    R"pred_obj <- prediction($predTrainAvg, $yTrain)";
    R"perf <- performance(pred_obj, 'tpr', 'fpr')";
    trainAUC = rcopy(R"performance(pred_obj, 'auc')@y.values[[1]]")

    # Testing AUC
    R"pred_obj <- prediction($predTestAvg, $yTest)";
    R"perf <- performance(pred_obj, 'tpr', 'fpr')";
    testAUC = rcopy(R"performance(pred_obj, 'auc')@y.values[[1]]")

    return(trainAUC, testAUC)
end
