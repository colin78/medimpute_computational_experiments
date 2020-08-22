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

function fitRegLinReg(X, y; seed = 1)
    for j in 1:size(X,2)
        if size(X[isnan.(X[:,j]),j], 1)>0
            X[isnan.(X[:,j]),j] .= mean(filter(!isnan, X[:,j]))
        end
    end

    train, test = splitData(y, SplitRatio = 0.75, seed = seed);
    xTrain = X[train,:]
    yTrain = y[train]
    xTest = X[test,:]
    yTest = y[test]

    R"set.seed($(seed))"
    R"lasso <- cv.glmnet($xTrain, $yTrain)"

    pred_train = rcopy(R"predict(lasso, newx = $xTrain, s = 'lambda.1se')")
    pred_test = rcopy(R"predict(lasso, newx = $xTest, s = 'lambda.1se')")
    
    # pred_train = rcopy(R"predict.cv.glmnet(lasso, newx = $xTrain, s = 'lambda.1se')")
    # pred_test = rcopy(R"predict.cv.glmnet(lasso, newx = $xTest, s = 'lambda.1se')")

    # Training MAE
    trainMAE = getMAE_pred(pred_train[:,1], Array{Union{Missing, Float64},1}(yTrain))
    # Testing MAE
    testMAE = getMAE_pred(pred_test[:,1], Array{Union{Missing, Float64},1}(yTest))
    return(trainMAE, testMAE)
end

function fitMultiRegLinReg(X, y; seed = 1)

    m = length(X)
    n, p = size(X[1])
    train, test = splitData(y, SplitRatio = 0.75, seed = seed);
    yTrain = Array{Union{Missing, Float64},1}(y[train])
    yTest = Array{Union{Missing, Float64},1}(y[test])
    predTrain = [Vector{Float64}() for _ in 1:m]
    predTest = [Vector{Float64}() for _ in 1:m]

    for i in 1:m
        xTrain = X[i][train,:]
        xTest = X[i][test,:]
        R"set.seed($(seed))"
        R"lasso <- cv.glmnet($xTrain, $yTrain)"
        predTrain[i] = rcopy(R"predict(lasso, newx = $xTrain, s = 'lambda.1se')")[:]
        predTest[i] = rcopy(R"predict(lasso, newx = $xTest, s = 'lambda.1se')")[:]
    end
    
    predTrainAvg = mean(hcat(predTrain...), dims=2)[:]
    predTestAvg = mean(hcat(predTest...), dims=2)[:]

    #Training MAE
    trainMAE = getMAE_pred(predTrainAvg, yTrain)

    # Testing MAE
    testMAE = getMAE_pred(predTestAvg, yTest)

    return(trainMAE, testMAE)
end
