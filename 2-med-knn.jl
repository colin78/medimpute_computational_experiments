include("MedKNN/MedKNN.jl") # Note: The MedKNN module
# requires an academic license in order to run.
# Please contact agniorf@mit.edu or cpawlows@mit.edu
# in order to obtain.  

# Convert the columns to the correct datatypes
function convertColumnTypes(dataTemp)
    dataResult = deepcopy(dataTemp)
    n, p = size(dataTemp)

    for d in 1:p
        typeValue = eltype(dataTemp[:,d])
        if typeValue in [Float64, CategoricalString{UInt32}]
            dataResult[:,d] = Vector{Union{Missing, typeValue}}(dataTemp[:,d])
        end
    end

    return(dataResult)
end

# Handle the columns which have only 0 or 1 value
function handleMissingColumns(dataTemp)
    dataResult = deepcopy(dataTemp)
    n, p = size(dataTemp)

    for d in 1:p
        num_non_missing = sum(.!ismissing.(dataTemp[:,d]))
        if num_non_missing == 0
            typeValue = eltype(dataTemp[:,d])
            typeValue = setdiff([typeValue.a, typeValue.b],
                [Missing])[1]
            if typeValue in [Float64, Int64]
                dataResult[:,d] = 0
            else
                dataResult[:,d] = "Missing"
            end
        end
        if num_non_missing == 1
            val = dataTemp[.!ismissing.(dataTemp[:,d]),d][1]
            dataResult[:,d] = val
        end
    end

    return(dataResult)
end

function wrapperMedKNN(dataFeatures, Time, PatientID)
    dataTemp = convertColumnTypes(dataFeatures)
    dataTemp = handleMissingColumns(dataTemp)

    dataTemp[:PatientID] = PatientID
    dataTemp[:Date] = Time
    dataMedKNN, params = MedKNN.medknnImpute(dataTemp)
    variableNames = setdiff(names(dataMedKNN),
        [:PatientID, :Date])
    dataImputed = dataMedKNN[variableNames]

    return(dataImputed, params)
end

function wrapperMovingAverage(dataFeatures, Time, PatientID)
    dataTemp = convertColumnTypes(dataFeatures)
    dataTemp = handleMissingColumns(dataTemp)

    dataTemp[:PatientID] = PatientID
    dataTemp[:Date] = Time
    dataMovingAvgImputed = MedKNN.movingAverageImpute(dataTemp)
    variableNames = setdiff(names(dataMovingAvgImputed),
        [:PatientID, :Date])
    dataImputed = dataMovingAvgImputed[variableNames]

    return(dataImputed)
end
