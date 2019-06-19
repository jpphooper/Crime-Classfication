
clc; clear all;
% For our poster we use two algorithms (NAIVE BAYES AND RANDOM FOREST) to
% investigate areas of high/low crime in London

%loading dataset of all plausible census 2001 variables that could effect crime rates in 2011
crime = readtable('LSOA_datasetvFinal.csv');

%SECTION 1: DATA EXPLORATION AND DATA PREPARATION
%==================================================================================================%
% creating the high/low crime variable dependent variable 
Y = crime.CrimeRate  > 150;
X = crime{:,2:12};


% summary statistics on continuous variables
% averages
AvX = mean(X);
AvXY1 = mean(X(Y,:));
AvXY2 = mean(X(~Y,:));

% standard deviations
StdX = std(X);
StdXY1 = std(X(Y,:));
StdXY2 = std(X(~Y,:));

% kurtosis/skew
SkewX = kurtosis(X);
SkewXY1 = kurtosis(X(Y,:));
SkewXY2 = kurtosis(X(~Y,:));

% normalising all continuous predictors
X = zscore(X);


% only two variables have a strong covariance above 0.8
Covariance = cov(X);

% pca dimension reduction of continouos predictors shows heavy loadings for all variables in the first component 
% we have therefore decided to keep all continuous variables.
[coeff,~,~,~,explained,~] = pca(X);

%initial histograms of continuous variables
% two variables (Fig(2) Population Density and House Prices) are particularly skewed.
% The histograms used in the poster are i = 5 and i = 6, which refer to the
% columns x_NOTUK and HousePrice respectively.
for i=1:11
figure(i)
histogram(X(:,i),'FaceColor','blue') 
hold on 
histogram(X(Y,i),'FaceColor','red')
histogram(X(~Y,i),'FaceColor','black')
end

% recombining dataset into a single table for pre-processing
X=[Y X];
X = array2table(X);
new_crime =[X crime.Borough];
%===================================================================================================%
%SECTION 2 NAIVE BAYES


%PART 1 PARTITION DATASET
% create training dataset and test set
cvpt = cvpartition(new_crime{:,1},'Holdout',0.2);
        dataTrain = new_crime(training(cvpt),:);
        dataTest = new_crime(test(cvpt),:);

%PART 2 CROSS VALIDATION WITH BASELINE AND DIFFERENT VARIANTS OF
%PRE-PROCESSING

% In this second part, for  cross-validation we looked at 3 variants of pre-processing tuning. Logging variables, equal
% width-binning and equal frequency binning on all of our variables. 
% None of our tuning showed a significant improvement on the standard model

%Start cross validation by initialising final results matrices for each parameterisation (ie each of the 4 rows)
%and each continuous variable (each of the 11 columns)
TotalErrNB = zeros(4,11);
TotStdModelErrNB = zeros(4,11);
TotAvSpecificityNB = zeros(4,11);
TotAvSensitivityNB = zeros(4,11);



%In this for loop, we loop through each continouos variable in the dataset
%and look at the impact of changing the pre-processing of a single variable
%on Naive bayes

% there are two stages to train NB to do this:

% A) PREPROCESSING FOR EACH CONTINUOUS VARIABLE
% B) CROSSVALIDATION FOR EACH TYPE OF PREPROCESSING



for p = 2:12
    
 % A) PREPROCESSING FOR EACH CONTINUOUS VARIABLE   
Parameter= zeros(size(dataTrain,1),4);

% Baseline variable with no preprocessing 
Parameter(:,1) = dataTrain{:,p};

%logged version of the variable
Parameter(:,2) = log(dataTrain{:,p});

% equal width binned version of the variable (note: we also looked at
% changing the number of intervals but this did not impact performance
% much. In the main 4 intervals provided the best binning)
Parameter(:,3) = discretize(dataTrain{:,p},4,'categorical',{'cat1','cat2','cat3','cat4'});

% equal frequency  binned version of the variable
% I use quantiles to split up the data by edges with equal frequencies.
quant0 = quantile(dataTrain{:,p},0);
quant1 = quantile(dataTrain{:,p},0.25);
quant2= quantile(dataTrain{:,p},0.5);
quant3 = quantile(dataTrain{:,p},0.75);
quant4 = quantile(dataTrain{:,12},1);
edges = [quant0 quant1 quant2 quant3 quant4];
Parameter(:,4) = discretize(dataTrain{:,p},edges,'categorical',{'cat1','cat2','cat3','cat4'});




% Training, Testing and validation
% in this section we mainly the data into training/testing and undertake cross validation. 
% However it also includes some parameter tuning. (see plotting model with different distributions example). 

% start by setting measures of performance to 0 
 AvModelErrNB = zeros(4,1);
 StdModelErrNB = zeros(4,1);
 AvSensitivityNB = zeros(4,1);
 AvSpecificityNB = zeros(4,1);

% MODEL CROSS VALIDATION USING EACH PARAMETERISATION
    for j = 1:4

        dataTrain{:,p} = Parameter(:,j);
        % split into training and testing 
        % we chose an 80:20 split
        

        % initalize error arrays
        ValidationError = zeros(10,1);
        cvspecificity = zeros(10,1);
        cvsensitivity = zeros(10,1);


        
        % we used 10 Kfold cross validation. This gives less protection against bias than other techniques such as leave one out but it has low error variance.  
        cvpt = cvpartition(new_crime{:,1},'KFold',10);
        for i = 1:10
            %vof
            cvdataTrain = new_crime(training(cvpt,i),:);
            cvdataTest = new_crime(test(cvpt, i),:);

            nbModel = fitcnb(cvdataTrain(:,2:end),cvdataTrain(:,1));
            finaltestvalues = cvdataTest{:,1};
            predictions = predict(nbModel,cvdataTest(:,2:end));

            confusion = confusionmat(finaltestvalues,predictions);
            cvsensitivity(i) = confusion(1,1)/(confusion(1,1)+confusion(2,1));
            cvspecificity(i) = confusion(2,2)/(confusion(1,2)+confusion(2,2));

            ValidationError(i) = loss(nbModel, cvdataTest(:,2:end),cvdataTest(:,1));
        end
    % average model Validation error for a single variable for all parameter tuning variants    
    AvModelErrNB(j) = mean(ValidationError);
    StdModelErrNB(j) = std(ValidationError)/sqrt(10);
    AvSensitivityNB(j) = mean(cvsensitivity);
    AvSpecificityNB(j) = mean(cvspecificity);
   
    end
   %Final results Matrix initialize
TotalErrNB(:,p-1) = AvModelErrNB;
TotStdModelErrNB(:,p-1) = StdModelErrNB;
TotAvSensitivityNB(:,p-1) = AvSensitivityNB;
TotAvSpecificityNB(:,p-1)= AvSpecificityNB;

  
    
    
end


%PART 3 CROSS VALIDATION WITH KERNEL PARAMETER TUNING

%re-initializie average model error arrays
 AvModelErrNB = zeros(11,1);
 StdModelErrNB = zeros(11,1);
 AvSensitivityNB = zeros(11,1);
 AvSpecificityNB = zeros(11,1);


for j = 1:11
% initalize error arrays
        cvError = zeros(10,1);
        cvspecificity = zeros(10,1);
        cvsensitivity = zeros(10,1);

        NBParameter = {'normal','normal','normal','normal','normal','normal','normal','normal','normal','normal','normal','mvmn'}
        NBParameter{j} = 'kernel'
        % cross-validation error
        % we used 10 Kfold cross validation. This gives less protection against bias than other techniques such as leave one out but it has low error variance.  
        cvpt = cvpartition(dataTrain{:,1},'KFold',10);
            for i = 1:10
                cvdataTrain = new_crime(training(cvpt,i),:);
                cvdataTest = new_crime(test(cvpt, i),:);

                nbModel = fitcnb(cvdataTrain(:,2:end),cvdataTrain(:,1),'dist',NBParameter);

                finaltestvaluesRF = cvdataTest{:,1};
                predictions = predict(nbModel,cvdataTest(:,2:end));

                confusion = confusionmat(finaltestvaluesRF,predictions);
                cvsensitivity(i) = confusion(1,1)/(confusion(1,1)+confusion(2,1));
                cvspecificity(i) = confusion(2,2)/(confusion(1,2)+confusion(2,2));

                cvError(i) = loss(nbModel, dataTest(:,2:end),dataTest(:,1));
            end
    % average model error for a single variable for all parameter tuning variants    
    AvModelErrNB(j) = mean(cvError);
    StdModelErrNB(j) = std(cvError)/sqrt(10);
    AvSensitivityNB(j) = mean(cvsensitivity);
    AvSpecificityNB(j) = mean(cvspecificity);
end

% add to other cross-validation preprocessing/parameter tuning
% the kernel paramter tuning variables are added to the arrays for other
% pre-processing parameter tuning
%TOTAL ERRNB is the data used in the figure in the poster
%AND the first row ofTotStdModelErrNB the standard error for the baseline  
TotalErrNB(5,:) = AvModelErrNB';
TotStdModelErrNB(5,:) = StdModelErrNB';
TotAvSensitivityNB(5,:) = AvSensitivityNB';
TotAvSpecificityNB(5,:)= AvSpecificityNB';




% PART4 TRAINING AND TESTING NAIVE BAYES USING BASELINE (THE BEST
% PERFORMING ALGORITHM ON THE DATASET)
nbModel = fitcnb(dataTrain(:,2:end),dataTrain(:,1));
finaltestvalues = cvdataTest{:,1};
finaltestpredictions = predict(nbModel,cvdataTest(:,2:end));
finalconfusion = confusionmat(finaltestvalues,finaltestpredictions);

% test sensitity reported in results table in the poster
testsensitivityNB = confusion(1,1)/(confusion(1,1)+confusion(2,1));
% test specificity reported in results table in the poster
testspecificityNB = confusion(2,2)/(confusion(1,2)+confusion(2,2));

% test accuracy of the model
testError = loss(nbModel, dataTest(:,2:end),dataTest(:,1));
% training accuracy of the model
trainError = loss(nbModel, dataTrain(:,2:end),dataTrain(:,1));




%================================================================================================%

% SECTION 3: RANDOM FOREST

%loading data
crime = readtable('LSOA_datasetvFinal.csv');

%creating dependent variable for high and low crime (above 150 is high)
Y = crime.CrimeRate > 150;

% normalising all the predictors
X = crime{:,2:12};
X = zscore(X);
X = [Y X];

%PART 1 PARTITION DATASET
% split into training and testing 
cvpt = cvpartition(X(:,1),'Holdout',0.2);
dataTrainfinal = X(training(cvpt),:);
dataTest = X(test(cvpt),:);

% split into training and validation set (to be used to find optimum
% hyperparameters).
cvpt = cvpartition(dataTrainfinal(:,1),'Holdout',0.1);
dataTrain = X(training(cvpt),:);
dataVal = X(test(cvpt),:);

% PART 2: GRID SEARCH FOR HYPERPARAMETERS
% First we will find the best value for the number of trees by searching
% through the range set out below in NumTrees from 50 to 150.
% Then Using that optimum value for number of trees we will perform a
% search for minleaves and number of predictors to sample.

% initialising search range
NumTrees = linspace(50,150,21);
TreeErrorrate = zeros(1,numel(NumTrees));

% Looking for Optimum Number of trees by finding error for each value of
% NumTrees.
% K-Fold was not used for validating and finding hyperparameters for Random
% Forests as there was a large increase in computational time
for i = 1:numel(NumTrees)
    MdlSearch = TreeBagger(NumTrees(i),dataTrain(:,2:12),dataTrain(:,1),'OOBPrediction','On','Method','classification');
    predictions = predict(MdlSearch,dataVal(:,2:12));
    predictions = str2double(predictions);
    finaltestvaluesRF = dataVal(:,1);
    errors = finaltestvaluesRF ~= predictions;
    TreeErrorrate(i) = nnz(errors)/numel(errors);
end

% Visualising the best number of trees (Graph not used in poster)
figure(12)
plot(NumTrees,TreeErrorrate)
title('Optimum Number of Trees')
xlabel('Number of Trees')
ylabel('Error')

% Finiding Optimum Tree
[OptErr, e] = min(TreeErrorrate);
OptTree = NumTrees(e);

NumPredictor = linspace(1,11,11);
MinLeaf = linspace(1,11,11);
errorrate = zeros(numel(MinLeaf),numel(NumPredictor));

% Looking for optimum number of predictors to sample and minimum leaf size
% Again K-fold was not used due to large increase in computational time.
for i = 1:numel(NumPredictor)
    for j = 1:numel(MinLeaf)
        MdlSearch = TreeBagger(OptTree,dataTrain(:,2:12),dataTrain(:,1),'OOBPrediction','On','Method','classification','NumPredictorsToSample',NumPredictor(i),'MinLeafSize',MinLeaf(i));

        predictions = predict(MdlSearch,dataVal(:,2:12));
        predictions = str2double(predictions);
        finaltestvaluesRF = dataVal(:,1);
        errors = finaltestvaluesRF ~= predictions;
        errorrate(j,i) = nnz(errors)/numel(errors);   
    end
end

%Visualising
figure(13)
pcolor(errorrate)
title('Hyperparamter Search')
xlabel('Minimum Leaf Size')
ylabel('Number of Predictors Sampled')

% Find optimal number of tress and predictors to sample

[a, b] = min(errorrate);
[c,d] = min(a);
OptLeafSize = NumPredictor(d);
OptPredictor = MinLeaf(b(d));

% PART3: TRAINING AND TESTING RANDOM FOREST USING OPTTREE, OPTLEAFSIZE AND
% OPTPREDICTOR AS HYPERPARAMETERS.
MdlFinalRF = TreeBagger(OptTree,dataTrainfinal(:,2:12),dataTrainfinal(:,1),'OOBPrediction','On','Method','classification','NumPredictorsToSample',OptPredictor,'MinLeafSize',OptLeafSize)
oobErrorBaggedEnsembleFinal = oobError(MdlFinalRF);

finaltestpredictionsRF = predict(MdlFinalRF,dataTest(:,2:12));
finaltestpredictionsRF = str2double(finaltestpredictionsRF);
finaltestvaluesRF = dataTest(:,1);
errors = finaltestvaluesRF ~= finaltestpredictionsRF;
% Test Accuracy of the final model used in poster
ErrorRF = nnz(errors)/numel(errors);
TestErrorRF = ErrorRF

finalconfusionRF = confusionmat(finaltestvaluesRF,finaltestpredictionsRF);
% test sensitity reported in results table in the poster
testsensitivityRF = finalconfusionRF(1,1)/(finalconfusionRF(1,1)+finalconfusionRF(2,1));
% test specificity reported in results table in the poster
testspecificityRF = finalconfusionRF(2,2)/(finalconfusionRF(1,2)+finalconfusionRF(2,2));

% Training Accuracy of the final model used in poster
finaltrainpredictionsRF = predict(MdlFinalRF,dataTrain(:,2:12));
finaltrainpredictionsRF = str2double(finaltrainpredictionsRF);
finaltrainvaluesRF = dataTrain(:,1);
Trainerrors = finaltrainvaluesRF ~= finaltrainpredictionsRF;
% Training Accuracy of the final model used in poster
TrainErrorRF = nnz(Trainerrors)/numel(Trainerrors);


% Plotting the OOBError (Not used in poster)
figure(14)
plot(1:1:OptTree,oobErrorBaggedEnsembleFinal,'c')
title('OOB Error')
xlabel('Number of grown trees')
ylabel('Out-of-bag classification error')









