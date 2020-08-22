using DataFrames, RDatasets, MLDataUtils, CSV
using StatsBase, Distributions

function readDanaFarberData(fileName, NAind; minEntries = 25)
    data = readtable(fileName, makefactors = true);
    data = data[:,[1:2;(end-2):end;3:(end-3)]];
    data = dropSparseColumns(data, minEntries = minEntries)

    if NAind
        return addMissingDataIndicators(data)
    else 
        return data
    end
end

function readSyntheaData(fileName,n)
    data = readtable(fileName, makefactors = true);

    unique_parts = size(unique(data[:PatientID]),1)

    if unique_parts>n
        limit = round(n/2,digits=0)
        obs_per_person =2

        obs_inds = []

        for i in unique(data[:PatientID])[1:Int64(limit)]
            allinds = findall(data[:PatientID].==i)
            if size(allinds,1)<obs_per_person
                append!(obs_inds, allinds)
            else
                append!(obs_inds, allinds[1:obs_per_person])
            end
        end

        if size(obs_inds,1)<n
            new_inds = n-size(obs_inds,1)
            append!(obs_inds, size(data,1)-new_inds+1:size(data,1))
        end
        data2 = data[obs_inds,:]
    else
        obs_per_person = round(n/size(unique(data[:PatientID]),1),RoundUp)
        limit = round(n/obs_per_person,digits=0)

        obs_inds = []

        for i in unique(data[:PatientID])[1:Int64(limit)]
            allinds = findall(data[:PatientID].==i)
            if size(allinds,1)<obs_per_person
                append!(obs_inds, allinds)
            else
                append!(obs_inds, allinds[1:Int64(obs_per_person)])
            end
        end

        if size(obs_inds,1)<n
            new_inds = n-size(obs_inds,1)
            append!(obs_inds, size(data,1)-new_inds+1:size(data,1))
        end
        if size(obs_inds,1)>n
           obs_inds = obs_inds[1:n]
        end

        data2 = data[obs_inds,:]
    end
    return data2
end

function addMissingDataIndicators(data)

    df = DataFrame(zeros(size(data,1), size(data[:,6:end],2)))
    for d in 1:ncol(data[:,6:end])
        df[ismissing.(data[:,d+5]),d] = 1
    end
    extDf = hcat(data,df);
    return extDf
end

function readFHSData(fileName)
    data = readtable(fileName, makefactors = true);
    delete!(data, [:x, :Cohort,:strokedate]);
    data = data[:,[1;(end-3):end;2:(end-4)]];
    return data
end

function readPDData(fileName)
    data = readtable(fileName, makefactors = true);
    data = data[:,[1;2;63;3:62;64:end]];
    return data
end

# Drop columns in a data frame which have fewer
# than "minEntries" non-NA's
function dropSparseColumns(x; minEntries = 25)
    sparseColumns = [sum(.!ismissing.(x[:,d])) < minEntries for d in 1:ncol(x)]
    x2 = deepcopy(x)
    return(x2[:,.!sparseColumns])
end

# Adds in a fixed percentage of missing values
# to a set of columns in a data frame.  Returns the new
# data frame and missing pattern that was created.
# Does not modify the original data frame.
function addMissingPercentage(data::DataFrame;
    pct = 10, features = 1:ncol(data), seed = 1, NAind=false)
    Random.seed!(seed)
    dataCopy = deepcopy(data);
    mp = falses(nrow(data), ncol(data));

    if NAind
        lim = Int64((ncol(data)-5)/2)+5
        features = 6:lim
    end

    for d in features
        x = findall(.!ismissing.(dataCopy[:,d]))
        missing_cols = sample(x, Int(floor(pct * length(x) / 100)), replace = false)
        dataCopy[missing_cols,d] .= missing
        for j in missing_cols
            dataCopy[j,d] = missing
        end
        mp[missing_cols,d] .= true
    end

    return(dataCopy, mp)
end

function addNMARMissingPercentage_PD(data::DataFrame;
    gamma = 0.1, features = 1:ncol(data), seed = 1, NAind=false, percent = 30)
    
    Random.seed!(seed)
    p = length(features)
    featureOrder = sample(features,p,replace=false)
    TimePerfeatureLimit =  rand(2:4, p)

    dataCopy = deepcopy(data);
    insert!(dataCopy, size(dataCopy,2)+1, zeros(size(dataCopy,1)), :time_obs)

    for i in 1:size(dataCopy,1)
        if dataCopy[i,:Times] > 900
            dataCopy[i,:time_obs] = 4
        elseif dataCopy[i,:Times] > 600
            dataCopy[i,:time_obs] = 3
        elseif dataCopy[i,:Times] > 200
            dataCopy[i,:time_obs] = 2
        else   
            dataCopy[i,:time_obs] = 1
        end
    end

    mp = falses(nrow(data), ncol(data));

    nonMissingValues = 0
    for d in features
       nonMissingValues = nonMissingValues + size(findall(.!ismissing.(dataCopy[:,d])),1)
    end

    numOfNMARvalues = Int(floor(gamma * (percent/100) * nonMissingValues))
    numOfMCARvalues = Int(floor((1-gamma) * (percent/100) * nonMissingValues))

    featList = zeros(0,3)
    counter = 1
    sumNMAR = 0
    sumMCAR = 0

    while sumNMAR < numOfNMARvalues && counter<=p
        
        d = featureOrder[counter]
        timeLim = TimePerfeatureLimit[counter]

        x = findall(.!ismissing.(dataCopy[:,d]))
        caplim =  Int(floor(percent * length(x) / 100))

        if ((sumNMAR +  sum(dataCopy[:,:time_obs].>timeLim)) <= numOfNMARvalues) && (sum(dataCopy[:,:time_obs].>timeLim)<=caplim)
            dataCopy[dataCopy[:,:time_obs].>timeLim ,d] .= missing;
            mp[dataCopy[:,:time_obs].>timeLim ,d] .= true
            sumNMAR = sumNMAR +  sum(dataCopy[:,:time_obs].>timeLim)
            println("Feature : $d , Limit at Number of Observations : $(timeLim)")
            featList = vcat(featList,[d timeLim sum(dataCopy[:,:time_obs].>timeLim)])
        else
            included = false
            while !(included)
                if timeLim <  9 
                    timeLim = timeLim + 1
                    if (sumNMAR +  sum(dataCopy[:time_obs].>timeLim)) <= numOfNMARvalues && (sum(dataCopy[:time_obs].>timeLim)<=caplim)
                         dataCopy[dataCopy[:time_obs].>timeLim ,d] .= missing
                         mp[dataCopy[:time_obs].>timeLim ,d] .= true
                         sumNMAR = sumNMAR +  sum(dataCopy[:time_obs].>timeLim)
                         included = true
                         println("Feature : $d , Limit at Number of Observations : $(timeLim)")
                         featList = vcat(featList,[d timeLim sum(dataCopy[:time_obs].>timeLim)])
                    end
                else
                    numRedValues = numOfNMARvalues - sumNMAR
                    res = min(caplim, numRedValues)
                    candInd = findall(dataCopy[:time_obs].>timeLim)
                    randInd =  rand(candInd, res)                
                    dataCopy[randInd ,d] .= missing
                    mp[randInd ,d] .= true
                    sumNMAR = sumNMAR +  numRedValues
                    included = true
                    println("Feature : $d , Limit at Number of Observations : $(timeLim)")
                    featList = vcat(featList,[d timeLim sum(dataCopy[:time_obs].>timeLim)])
                end
            end
        end
        counter +=1       
    end

    featList = convert(Array{Int64,2},featList)
    numOfMCARvaluesPerColumn = Int(floor(((1-gamma) * (percent/100) * nonMissingValues)/(p)))

    for d in features

        k = -size(findall(.!ismissing.(dataCopy[:,d])),1)+size(findall(.!ismissing.(data[:,d])),1)
        additonalPerc = percent - 100*k/size(findall(.!ismissing.(data[:,d])),1)
        x = findall(.!ismissing.(dataCopy[:,d]))
        missing_cols = sample(x, Int(floor(additonalPerc * length(x) / 100)), replace = false)
        dataCopy[missing_cols,d] .= missing
        for j in missing_cols
            dataCopy[j,d] = missing
        end
        mp[missing_cols,d] .= true
        
    end

    return(dataCopy, mp)
end

function addNMARMissingPercentage_FHS(data::DataFrame;
    gamma = 0.1, features = 1:ncol(data), seed = 1, NAind=false, percent = 30)
    
    Random.seed!(seed)
    p = length(features)
    featureOrder = sample(features,p,replace=false)
    TimePerfeatureLimit =  rand(3:9, p)

    dataCopy = deepcopy(data);
    mp = falses(nrow(data), ncol(data));

    nonMissingValues = 0
    for d in features
       nonMissingValues = nonMissingValues + size(findall(.!ismissing.(dataCopy[:,d])),1)
    end

    numOfNMARvalues = Int(floor(gamma * (percent/100) * nonMissingValues))
    numOfMCARvalues = Int(floor((1-gamma) * (percent/100) * nonMissingValues))

    featList = zeros(0,3)
    counter = 1
    sumNMAR = 0
    sumMCAR = 0

    while sumNMAR < numOfNMARvalues && counter<=p
        
        d = featureOrder[counter]
        timeLim = TimePerfeatureLimit[counter]

        x = findall(.!ismissing.(dataCopy[:,d]))
        caplim =  Int(floor(percent * length(x) / 100))

        if ((sumNMAR +  sum(dataCopy[:,:time_obs].>timeLim)) <= numOfNMARvalues) && (sum(dataCopy[:,:time_obs].>timeLim)<=caplim)
            dataCopy[dataCopy[:,:time_obs].>timeLim ,d] .= missing;
            mp[dataCopy[:,:time_obs].>timeLim ,d] .= true
            sumNMAR = sumNMAR +  sum(dataCopy[:,:time_obs].>timeLim)
            println("Feature : $d , Limit at Number of Observations : $(timeLim)")
            featList = vcat(featList,[d timeLim sum(dataCopy[:,:time_obs].>timeLim)])
        else
            included = false
            while !(included)
                if timeLim <  9 
                    timeLim = timeLim + 1
                    if (sumNMAR +  sum(dataCopy[:time_obs].>timeLim)) <= numOfNMARvalues && (sum(dataCopy[:time_obs].>timeLim)<=caplim)
                         dataCopy[dataCopy[:time_obs].>timeLim ,d] .= missing
                         mp[dataCopy[:time_obs].>timeLim ,d] .= true
                         sumNMAR = sumNMAR +  sum(dataCopy[:time_obs].>timeLim)
                         included = true
                         println("Feature : $d , Limit at Number of Observations : $(timeLim)")
                         featList = vcat(featList,[d timeLim sum(dataCopy[:time_obs].>timeLim)])
                    end
                else
                    numRedValues = numOfNMARvalues - sumNMAR
                    res = min(caplim, numRedValues)
                    candInd = findall(dataCopy[:time_obs].>timeLim)
                    randInd =  rand(candInd, res)                
                    dataCopy[randInd ,d] .= missing
                    mp[randInd ,d] .= true
                    sumNMAR = sumNMAR +  numRedValues
                    included = true
                    println("Feature : $d , Limit at Number of Observations : $(timeLim)")
                    featList = vcat(featList,[d timeLim sum(dataCopy[:time_obs].>timeLim)])
                end
            end
        end
        counter +=1       
    end

    featList = convert(Array{Int64,2},featList)
    numOfMCARvaluesPerColumn = Int(floor(((1-gamma) * (percent/100) * nonMissingValues)/(p)))

    for d in features

        k = -size(findall(.!ismissing.(dataCopy[:,d])),1)+size(findall(.!ismissing.(data[:,d])),1)
        additonalPerc = percent - 100*k/size(findall(.!ismissing.(data[:,d])),1)
        x = findall(.!ismissing.(dataCopy[:,d]))
        missing_cols = sample(x, Int(floor(additonalPerc * length(x) / 100)), replace = false)
        dataCopy[missing_cols,d] .= missing
        for j in missing_cols
            dataCopy[j,d] = missing
        end
        mp[missing_cols,d] .= true
        
    end

    return(dataCopy, mp)
end

function getMissingPatternColumnNMAR(x, a, b, seed)
    observed = findall(.!ismissing.(x))
    σ = std(x[observed])
    n = length(observed)
    
    Random.seed!(seed);
    if b == 0
        indicesNMAR = Int[]
    else
        noise = rand(Normal(0, σ), n)
        index = Int(floor(n * (1 - b) + 1))
        valueNMAR = x[observed] + noise
        threshold = sort(valueNMAR)[index]
        indicesNMAR = observed[valueNMAR .>= threshold]
    end

    numberMCAR = Int(floor(n * a))

    if size(setdiff(observed, indicesNMAR),1)>0
        indicesMCAR = sample(setdiff(observed, indicesNMAR),
             numberMCAR, replace = false)
    else
        indicesMCAR = []
    end

    return indicesNMAR, indicesMCAR
end

function getMissingPatternNMAR(data; features = 1:ncol(data),
    pct = 30, γ = 0, seed = 1,
    NAind = true)

    dataCopy = deepcopy(data);
    mp = falses(size(data));

    if NAind
        features = 11:110
    end
    
    a = (pct / 100) * (1 - γ)
    b = (pct / 100) * γ
    println("$(100 * a) % MCAR")
    println("$(100 * b) % NMAR")

    for d in features
        x = deepcopy(data[:,d])
        if isa(x, CategoricalArray)
            temp = zeros(length(x))
            temp = Array{Union{Missing, Float64},1}(temp)
            temp[ismissing.(x)] .= missing 
            for i in 1:length(levels(x))
                temp[(x .== levels(x)[i]) .& (.!ismissing.(x))] .= i
            end
            x = temp
        end

        indicesNMAR, indicesMCAR = getMissingPatternColumnNMAR(x, a, b, seed)
        mp[indicesNMAR, d] .= true
        mp[indicesMCAR, d] .= true

        for j in indicesMCAR
            dataCopy[j, d] = missing
        end

        for j in indicesNMAR
            dataCopy[j, d] = missing
        end
        
    end

    return(dataCopy, mp)
end


function splitData(y; SplitRatio = 0.75, seed = 1)
    Random.seed!(seed)
    allIndices = findall(.!ismissing.(y))
    (train, yTrain), (test, yTest) = stratifiedobs((allIndices, y[allIndices]), p = SplitRatio)
    train = Vector{Int}(train)
    test = Vector{Int}(test)
    return train, test
end
