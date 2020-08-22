using MedImpute
using JLD

function imputation(dataMissing, method;
  features = 1:ncol(dataMissing),
  indexPatientID::Int = 0,
  indexTime::Int = 0,
  fileMedParams::String = "",
  seed::Int = 1,
  m::Int = 5)

  Random.seed!(seed)

  x_imputed = deepcopy(dataMissing);
  PatientID = Vector{Int}(dataMissing[:,indexPatientID]);
  Time = Vector{Float64}(dataMissing[:,indexTime]);

  for d in 1:ncol(x_imputed)
    if isa(x_imputed[:,d], Array{Union{Missing, Int64},1})
      x_imputed[:,d] = Array{Union{Missing, Float64},1}(x_imputed[:,d])
    end
  end

  if method == "mean"
    X_imputed = MedImpute.opt_impute(dataMissing[:,features], lnr=MedImpute.Learner(:mean))
    X_imputed = DataFrame(hcat(dataMissing[1:10], X_imputed))
    return(X_imputed)
  elseif method == "linear"
    X_imputed = MedImpute.med_lin_interpolation(dataMissing[:,features], PatientID, Time, verbose=false)
    X_imputed = DataFrame(hcat(dataMissing[1:10], X_imputed))
    return(X_imputed)
  elseif method == "opt.knn"
    lnr = MedImpute.Learner(:opt_knn)
    X_imputed = MedImpute.opt_impute(dataMissing[:,features], lnr=lnr)
    X_imputed = DataFrame(hcat(dataMissing[1:10], X_imputed))
    return(X_imputed)
  elseif method == "med.knn"
    X_imputed, medParams = wrapperMedKNN(dataMissing[:,features],
        Time, PatientID)
    X_imputed = DataFrame(hcat(dataMissing[1:10], X_imputed))
    CSV.write(fileMedParams, medParams)
    return(X_imputed)
  elseif method == "mice"
    x_imputed_mi = mice(dataMissing, features = features, m = m)
    return(x_imputed_mi)
  elseif method == "bpca"
    X_imputed = MedImpute.opt_impute(dataMissing[:,features], lnr=MedImpute.Learner(:bpca))
    X_imputed = DataFrame(hcat(dataMissing[1:10], X_imputed))
    return(X_imputed)
  elseif method == "amelia"
    X_imputed = amelia_imputation(dataMissing, Time, PatientID, features = features, m = m)
    X_imputed = DataFrame(hcat(dataMissing[1:10], X_imputed))
    return(X_imputed)
  elseif method == "moving.avg"
    X_imputed = wrapperMovingAverage(dataMissing[:,features],
        Time, PatientID)
    X_imputed = DataFrame(hcat(dataMissing[1:10], X_imputed))
    return(X_imputed)
  else
    error("method must be one of: 'mean', 'linear', 'opt.knn', 'med.knn', 'mice', 'bpca'.")
  end
end

using RCall
R"library(mice)"

function mice(dataMissing::DataFrame;
              features = 1:ncol(dataMissing),
              m::Int64 = 5,
              maxit::Int64 = 2,
              seed::Int64 = 1)
    R"tempData <- mice($(dataMissing[:,features]), m=$m, maxit=$maxit, seed=$seed,
                         defaultMethod = c('cart','cart', 'cart', 'cart'))";

    X_imputed = [deepcopy(dataMissing) for _ in 1:m];
    for i=1:m
        completeImputation = rcopy(R"complete(tempData, $i)");
        # Run mean impute finally, because MICE algorithm results in NAs
        # if there are collinear columns
        p = ncol(completeImputation)
        # [d for d=1:p if sum(isna.(completeImputation[:,d])) > 0]
        for d in 1:p
          missingBool = getMissing(completeImputation[:,d])
          missingInd = findall(missingBool)
          knownInd = findall(.!missingBool)

          if isa(completeImputation[knownInd[1],d], Number)
            # X_imputed[i][:,features[d]] = DataArrays.DataArray{Float64, 1}(X_imputed[i][:,features[d]])
            # completeImputation[:,d] = DataArrays.DataArray{Float64, 1}(completeImputation[:,d])
            X_imputed[i][:,features[d]] = (X_imputed[i][:,features[d]])
            completeImputation[:,d] = (completeImputation[:,d])   
            fillFunction = mean
          else
            # X_imputed[i][:,features[d]] = DataArrays.PooledDataArray{String,UInt32,1}(X_imputed[i][:,features[d]])
            X_imputed[i][:,features[d]] = (X_imputed[i][:,features[d]])
            fillFunction = mode
          end

          if sum(missingBool) > 0
            if isa(completeImputation[knownInd[1],d], Number) & size(unique(completeImputation[knownInd,d]),1)<10
              completeImputation[missingInd,d] = mode(completeImputation[knownInd,d])
            else
              completeImputation[missingInd,d] = fillFunction(completeImputation[knownInd,d])
            end
          end
        end
        
        # completeImputation = opt_impute(completeImputation, lnr=Learner(:mean));
        X_imputed[i][:,features] = completeImputation
    end

    return(X_imputed)
end

function getMissing(x)
    n = length(x)

    missingNA = try
        isna.(x)
    catch
        falses(n)
    end

    missingNaN = try
        isnan.(x)
    catch
        falses(n)
    end

    missingNULL = try
        isnull.(x)
    catch
        falses(n)
    end

    missingMissing = try
        ismissing.(x)
    catch
        falses(n)
    end

    missingResult = missingNA .| missingNaN .| missingNULL .| missingMissing

    return(missingResult)
end

function isCategoric(col)
  option1 = isa(col, Vector{Union{Missing,CategoricalString{UInt8}}})
  option2 = isa(col, CategoricalArray{Union{Missing, String},1,UInt32})
  option3 = isa(col, CategoricalString{UInt8})

  return(option1 | option2 | option3)
end

function amelia_imputation(dataMissing::DataFrame,Time::Array{Float64,1}, 
                      PatientID::Array{Int64,1};
                      features = 1:ncol(dataMissing),
                      m::Int64 = 5,
                      maxit::Int64 = 2,
                      seed::Int64 = 1)
  X_imputed = deepcopy(dataMissing[:,features]);
  X_imputed = DataFrame(hcat(X_imputed,PatientID, Time, makeunique=true));
  rename!(X_imputed, Dict(:x1 => :PatientID));
  rename!(X_imputed, Dict(:x1_1 => :Time));
  
  col_list = [];
  for i in 1:ncol(X_imputed)
    if length(unique(skipmissing(X_imputed[:,i])))==1
      append!(col_list, i)
    end
  end
  
  right_ord_names = names(X_imputed);
  
  mean_names = names(X_imputed)[col_list]
  append!(mean_names, [:HIV,:Dementia])
  amelia_names = names(X_imputed)[.![x in mean_names for x in names(X_imputed)]]
  
  X_mean = deepcopy(X_imputed[:,unique(mean_names)]);
  X_amelia = deepcopy(X_imputed[:,amelia_names]);
  
  a = Int64[];
  for i in 1:(ncol(X_amelia))
    if isCategoric(X_amelia[:,i])
      push!(a, i)
    end
  end
  
  cat_n = names(X_amelia)[a]
  filter!(x->x≠:HIV,cat_n)
  filter!(x->x≠:Dementia,cat_n)
  
  cols_to_keep = deepcopy(cat_n)
  left_names = names(X_amelia)[.![x in cols_to_keep for x in names(X_amelia)]]
  
  filter!(x->x≠:Time,left_names)
  filter!(x->x≠:PatientID,left_names)
  
  x_corr = MedImpute.opt_impute(X_imputed[:, left_names], lnr=MedImpute.Learner(:mean))
  
  R"library(caret)"
  indexesToDrop = Array{Int64,1}(rcopy(R"findCorrelation(cor($(x_corr)), cutoff = 0.2)"))
  
  append!(mean_names, left_names[indexesToDrop])
  amelia_names = names(X_imputed)[.![x in mean_names for x in names(X_imputed)]]
  
  X_mean = deepcopy(X_imputed[:,unique(mean_names)])
  X_amelia = deepcopy(X_imputed[:,amelia_names])
  
  a = Int64[]
  for i in 1:(ncol(X_amelia))
    if isCategoric(X_amelia[:,i])
      push!(a, i)
    end
  end
  
  cat_n = names(X_amelia)[a]
    
  amelia_imputed = rcopy(R"Amelia::amelia($(X_amelia), m = 5, p2s = 1,ts = c('Time'),ords= as.character(unlist($cat_n)), idvars=c('PatientID'))[1]")
  
  k = []
  for j in [:imp1,:imp2,:imp3,:imp4,:imp5]
    if typeof(amelia_imputed[:imputations][j]) ==DataFrame
      push!(k, j)
    end
  end

  if size(k,1) == 0 
     x_imputed = MedImpute.opt_impute(dataMissing[:,features], lnr=MedImpute.Learner(:mean))
     return x_imputed
  else
     amelia_imputed_df = amelia_imputed[:imputations][k[length(k)]]
     mean_imputed = MedImpute.opt_impute(X_mean, lnr=MedImpute.Learner(:mean))
  
     x_imputed = hcat(amelia_imputed_df, mean_imputed)
     x_imputed = x_imputed[:,right_ord_names]
     return(x_imputed[:,1:(ncol(x_imputed)-2)])
  end

end

