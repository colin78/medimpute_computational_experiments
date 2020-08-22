using DataFrames, RDatasets, MLDataUtils

function getMAE(data::DataFrame,
                dataImputed::DataFrame,
                mpValid::BitArray{2};
                features = 1:ncol(data),
                fileMAE = "")
    featureMAE = zeros(ncol(data))
    totalAbsError = 0.0
    for d in features
        trueValues = data[mpValid[:,d],d]
        imputedValues = dataImputed[mpValid[:,d],d]
        if size(trueValues)!=(0,)
            if isa(trueValues[1], Number)
                if size(imputedValues[ismissing.(imputedValues)],1)>0
                    imputedValues[ismissing.(imputedValues)].=mean(skipmissing(imputedValues))
                end
                absoluteDiff = abs.(trueValues - imputedValues)
            else
                if size(imputedValues[ismissing.(imputedValues)],1)>0
                    imputedValues[ismissing.(imputedValues)].=mode(skipmissing(imputedValues))
                end
                absoluteDiff = trueValues .== imputedValues
            end
            if size(absoluteDiff[isnan.(absoluteDiff)], 1)>0
                absoluteDiff[isnan.(absoluteDiff)] .= mean(filter(!isnan, absoluteDiff))
            end
            featureMAE[d] = mean(absoluteDiff)
            totalAbsError += sum(absoluteDiff)
        end
    end

    totalMAE = totalAbsError / sum(mpValid)
    result = DataFrame(Total = totalMAE)

    for d in features
        featureName = names(data)[d]
        result[featureName] = featureMAE[d]
    end

    if fileMAE != ""
        CSV.write(fileMAE, result)
    end

    return(result)
end

function getMAE_pred(imputedValues:: Array{Float64,1},
                trueValues:: Array{Union{Missing, Float64},1})
    totalAbsError = 0.0
    absoluteDiff = abs.(trueValues - imputedValues)    
    MAE = mean(absoluteDiff)
    return(MAE)
end

function getRMSE(data::DataFrame,
                dataImputed::DataFrame,
                mpValid::BitArray{2};
                features = 1:ncol(data),
                fileRMSE = "")
    featureRMSE = zeros(ncol(data))
    totalError = 0.0
    n = size(data,1)

    for d in features
        trueValues = data[mpValid[:,d],d]
        imputedValues = dataImputed[mpValid[:,d],d]

        if size(trueValues)!=(0,)
            if isa(trueValues[1], Number) 
                if size(imputedValues[ismissing.(imputedValues)],1)>0
                    imputedValues[ismissing.(imputedValues)].=mean(skipmissing(imputedValues))
                end
                absoluteDiff = (trueValues - imputedValues).^2
            else
                if size(imputedValues[ismissing.(imputedValues)],1)>0
                    imputedValues[ismissing.(imputedValues)].=mode(skipmissing(imputedValues))
                end
                absoluteDiff = trueValues .== imputedValues
            end
            if size(absoluteDiff[isnan.(absoluteDiff)], 1)>0
                absoluteDiff[isnan.(absoluteDiff)] .= mean(filter(!isnan, absoluteDiff))
            end
            
            featureRMSE[d] =  sqrt(sum(skipmissing(absoluteDiff))/n)
            totalError += sum(absoluteDiff)
        end
    end

    totalRSMSE = sqrt(totalError / sum(mpValid))
    result = DataFrame(Total = totalRSMSE)

    for d in features
        featureName = names(data)[d]
        result[featureName] = featureRMSE[d]
    end

    if fileRMSE != ""
        CSV.write(fileRMSE, result)
    end

    return(result)
end

cancerTypeList = ["BREAST","LUNG","GASTRO","ESOPHA","KIDNEY",
  "PROSTATE","COLORECTAL","SKIN","PANCREATIC","OVARIAN",
  "LYMPH","LEUK","CARCI","SARCO"]
cancerTypeList =   cancerTypeList.*"_meanAdj"
experimentsFHS = ["FHD_exp1_time_$i" for i in 1:10]

function aggregateResults(dataSetName)
    cancerTypeList = ["BREAST","LUNG","GASTRO","ESOPHA","KIDNEY",
         "PROSTATE","COLORECTAL","SKIN","PANCREATIC","OVARIAN",
         "LYMPH","LEUK","CARCI","SARCO","BREAST_CARCI_PROSTATE"]
    #cancerTypeList =   cancerTypeList.*"_meanAdj"
    experimentsFHS = ["FHD_exp1_time_$i" for i in 1:10]


    if dataSetName in cancerTypeList
        folder_results = "../Dana_Farber_results_final/"
        folder_aggregated = "../aggregated_results_final/"
    else
        folder_results = "../FHS_results"
        folder_aggregated = "../aggregated_results"
    end

    missingPctList = [30;]
    seedSplitList = [1:5;]
    gammaList = [0:0.1:1;]
    methodList = ["mean", "linear", "opt.knn", "med.knn", "med.knn.params", "mice", "bpca"]
    paramList = collect(Iterators.product(missingPctList, seedSplitList, methodList,gammaList))

    # Aggregate MAE results
    dataFrameMAE = DataFrame()
    for (pct, seed, method, gamma) in paramList
        #fileMAE = "$folder_results/MAE\_$dataSetName\_$method\_$pct\_seed\_$seed$maxCategoryString.csv"
        fileMAE = "$folder_results/MAE_$(dataSetName)_$(method)_$(pct)_seed_$(seed)_NMAR_$(gamma).csv"

        # println(fileMAE)
        if isfile(fileMAE)
            nextDF = readtable(fileMAE)
            nextDF[:Method] = method
            nextDF[:pct] = pct
            nextDF[:seed] = seed
            nextDF[:gamma] = gamma
            dataFrameMAE = [dataFrameMAE; nextDF]
        end
    end
    dataFrameMAE = dataFrameMAE[:,[(end-3):end;1:(end-4)]]
    #writetable("$folder_aggregated/MAE_$dataSetName$maxCategoryString.csv", dataFrameMAE)
    writetable("$folder_aggregated/MAE_$(dataSetName_)$(gamma).csv", dataFrameMAE)

    # Aggregate AUC results
    dataFrameAUC = DataFrame()
    for (pct, seed, method, gamma) in paramList
        #fileAUC = "$folder_results/AUC\_$dataSetName\_$method\_$pct\_seed\_$seed$maxCategoryString.csv"
        fileAUC = "$folder_results/AUC_$(dataSetName)_$(method)_$(pct)_seed_$(seed)_NMAR_$(gamma).csv"
        if isfile(fileAUC)
            nextDF = readtable(fileAUC)
            nextDF[:Method] = method
            nextDF[:pct] = pct
            nextDF[:seed] = seed
            nextDF[:gamma] = gamma
            dataFrameAUC = [dataFrameAUC; nextDF]
        end
    end
    dataFrameAUC = dataFrameAUC[:,[(end-3):end;1:(end-4)]]
    #writetable("$folder_aggregated/AUC_$dataSetName$maxCategoryString.csv", dataFrameAUC)
    writetable("$folder_aggregated/AUC_$dataSetName_$(gamma).csv", dataFrameAUC)

    dataFrameMedParams = DataFrame()
    for (pct, seed, method, gamma) in collect(Iterators.product(missingPctList, seedSplitList, methodList,gammaList))
        fileMedParams = "$folder_results/MedParams_$(dataSetName)_$(pct)_seed_$(seed)_NMAR_$(gamma).csv"
        if isfile(fileMedParams)
            nextDF = readtable(fileMedParams)
            nextDF[:pct] = pct
            nextDF[:seed] = seed
            nextDF[:Method] = method
            nextDF[:gamma] = gamma
            dataFrameMedParams = [dataFrameMedParams; nextDF]
        end
    end
    if size(dataFrameMedParams, 2) > 0
        dataFrameMedParams = dataFrameMedParams[:,[(end-1):end;1:(end-2)]]
        writetable("$folder_aggregated/MedParams_$(dataSetName)_$(gamma).csv", dataFrameMedParams)
    end
end
