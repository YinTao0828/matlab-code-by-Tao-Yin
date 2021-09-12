%% Feature selection and SVM classification

% First, the subjects were randomly divided into a cross-validation set 
% (80% of subjects) and an independent validation set (20% of subjects). 
% Then the feature selection was performed at the cross-validation set to
% find the valuable features for classification. Based on these selected 
% features, the RBF SVM classifier between patients and HS was established
% at the cross-validation set. The established classifier was then applied
% to classify FD patients and HS of the independent validation set, to 
% further evaluate the generalization of the classifier.
% by Tao Yin, Chengdu University of Traditional Chinese Medicine
% Email:605499143@qq.com
% The LIBSVM toolbox is needed to add to the folder.
% Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for
% support vector machines, 2001. Software available at
% http://www.csie.ntu.edu.tw/~cjlin/libsvm

%% Load the functional brain network of FD patients and HS
clc; clear all;
NumROI  = 35; % Number of nodes in the functional brain network
conn_msk = ones(NumROI);
path_FD = 'F:\phd_original_STAT\FC_FD_SVM\DATA_matrix\FD'; 
file1 = dir([path_FD,filesep, '*.mat']);
path_HS = 'F:\phd_original_STAT\FC_FD_SVM\DATA_matrix\HS';
file2 = dir([path_HS,filesep, '*.mat']);
Ind_01    = find(triu(ones(NumROI),1));
Ind_02    = find(conn_msk(Ind_01) ~= 0);

DATA_FD = zeros(length(file1), length(Ind_02));
for i = 1:length(file1)
    load([path_FD,filesep, file1(i).name])
    DATA_FD(i,:) = R(Ind_01(Ind_02)); % the name of matrix is 'R'
end
DATA_HS = zeros(length(file2), length(Ind_02));
for i = 1:length(file2)
    load([path_HS,filesep, file2(i).name])
    DATA_HS(i,:) = R(Ind_01(Ind_02));
end
data_all = [DATA_FD;DATA_HS];
label_all = [ones(size(DATA_FD,1),1); -1*ones(size(DATA_HS,1),1)];
clear conn_msk file1 file2 Ind_01 Ind_02 i path_FD path_HS R;

%% Feature selection
h = waitbar(0,'please wait..');
permut = 20; 
for mn = 1:permut
        waitbar(mn/permut,h,['repetition:',num2str(mn),'/',num2str(permut)]);
        sample = size(data_all,1);
        a = [1:sample];
        c = randperm(numel(a));
        TrainTest = a(c(1:fix(sample*0.8))); % 80% participants divided into a cross-validation set
        IndepVerif = a(c(fix(sample*0.8)+1:sample));
        DATA = data_all(TrainTest,:);
        label = label_all(TrainTest,:);
        clear a c sample;
        
        step = 1;
        x  = 0; % number of fearture
        k = 10;
        indices=crossvalind('Kfold',size(DATA,1),k);% 10-fold cross-validation
        for z = 1:ceil((size(DATA,2))/step)
            x = x+step;
            for i = 1:k % Perform 10-fold cross-validation with different number of features
                 test = (indices == i); train = ~test;
                 train_data = DATA(train,:);
                 train_label=label(train,:);
                 test_data = DATA(test,:);
                 test_label=label(test,:);
                 % Sort the features in descending order by t-value
                 [~,~,~,stat] = ttest2(train_data(train_label==1,:),train_data(train_label==-1,:));
                 F(i,:) = abs(stat.tstat);
                 clear stat 
                 [B,IX] = sort(F(i,:),'descend');
                 order(i,:) = IX;
                 
                 if x == 1 % Construct a classifier using the first-ranked feature
                   Feature(i,1) = order(i,1); 
                   Feature_num(1) = 1;
                   model = svmtrain(train_label,train_data(:,Feature(i,1)),'-t 2'); 
                   [predicted_label, accuracy, deci] = svmpredict(test_label,test_data(:,Feature(i,1)),model);
                   acc(i) = accuracy(1); 
                   clear train_data train_label test_label test_data model;
                 else % Construct a classifier using the multiple features
                    if x < size(DATA,2)
                        new_Feature = [Feature(i,:),order(i,x)];
                        model = svmtrain(train_label,train_data(:,new_Feature),'-t 2');
                        [predicted_label, accuracy, deci] = svmpredict(test_label,test_data(:,new_Feature),model);
                        acc(i) = accuracy(1); 
                        clear test train train_data train_label test_label test_data model;
                    end
                  end

            end
            mean_acc = mean(acc); % Calculate the average accuracy of 10-fold cross-validation with the current features

            if x == 1
                max_acc(1) = mean_acc;% Record the accuracy obtained with the first-ranked feature
            else
                if  max_acc(size(Feature,2)) <= mean_acc    % Compare the accuracy of classifier obtained with n+1 features to n features 
                    Feature = [Feature,order(:,x)];         % Retain the currently added feature
                    Feature_num = [Feature_num,x];          % Record the number of the feature
                    max_acc(size(Feature,2)) = mean(acc);   % Record the max accuracy
                end
            end

        end
        acc_Feature = [max_acc;Feature_num];
        clear acc accuracy B deci F i IX k max_acc mean_acc new_Feature order predicted_label step x z Feature_num;
        
   %% Selecte the top 10% as the classification features
        feature_occurrence = tabulate(Feature(:));
        y = 1;
        feature_identify = [];
        Sequence = sort(feature_occurrence(:,2),'descend'); % Order features by frequency of occurrence
        for n = 1:size(feature_occurrence,1)
            if feature_occurrence(n,2)> Sequence(60,1) % Selecte the top 10%£¨595*10% = 60£©features
               feature_identify(y,1) = feature_occurrence (n,1);
               y = y+1;
            end
        end
        clear y n feature_occurrence Feature Sequence;
        
    %% Construct classifiers for FD patients and HS based on selected features in cross-validation sets
        k2 = 10; 
        indices2 = crossvalind('Kfold',size(DATA,1),k2);
        for ij = 1:k2
            test = (indices2 == ij); train = ~test;
            train_data = DATA(train,feature_identify);
            train_label = label(train,:);
            test_data = DATA(test,feature_identify);
            test_label = label(test,:);
            model = svmtrain(train_label,train_data,'-t 2');    
            [predicted_label, accuracy, deci] = svmpredict(test_label,test_data,model);
            acc(ij,1) = accuracy(1);
            deci_value(test,1) = deci;
            predicted(test,1) = predicted_label;
        end
        acc_TrainTest = mean(acc);
        Sensitivty_TrainTest = sum((label==1)&(predicted==1))/sum(label==1);
        Specificity_TrainTest = sum((label==-1)&(predicted==-1))/sum(label==-1);
        [X_TrainTest,Y_TrainTest,~,AUC_TrainTest] = perfcurve(label,deci_value,1); % ROC curve
        clear k2 acc ij test train train_data train_label test_label test_data model predicted_label accuracy deci;
        
    %% Evaluate the generalization of the classifier in the independent validation set
        train_data = data_all(TrainTest,feature_identify);
        train_label = label_all(TrainTest,:);
        test_data = data_all(IndepVerif,feature_identify);
        test_label = label_all(IndepVerif,:);
        
        model = svmtrain(train_label,train_data,'-t 2');    
        [predicted_label, accuracy, deci] = svmpredict(test_label,test_data,model);
        acc_IndepVerif = accuracy(1); 
        Sensitivty_IndepVerif = sum((test_label==1)&(predicted_label==1))/sum(test_label==1);
        Specificity_IndepVerif = sum((test_label==-1)&(predicted_label==-1))/sum(test_label==-1);
        [X,Y,~,AUC] = perfcurve(test_label,deci,1); % ROC curve
        w = model.SVs'*model.sv_coef;
        weight_feature{:,mn} = [feature_identify,w];

        % Perform permutation tests on AUC (permutation times=1000)
        Nsloop = 1000;
        auc_rand = zeros(Nsloop,1);
        for i = 1:Nsloop
            label_rand = randperm(length(test_label));
            deci_value_rand = deci(label_rand);
            [~,~,~,auc_rand(i)] = perfcurve(test_label,deci_value_rand,1);
        end
        p_auc = mean(auc_rand >= AUC)
        clear label_rand deci_value_rand auc_rand i;
 
        % Perform permutation tests on accuracy (permutation times=1000)
        for i = 1:Nsloop
            label_rand = randperm(length(test_label));
            label_r  = test_label(label_rand); 
            train_data = data_all(TrainTest,feature_identify);
            train_label = label_all(TrainTest,:);
            test_data = data_all(IndepVerif,feature_identify);
            test_label = label_r;
            model = svmtrain(train_label,train_data,'-t 2');    
            [predicted_label, accuracy, deci] = svmpredict(test_label,test_data,model);
            acc_r(i) = accuracy(1);  
        end 
        p_acc = mean(abs(acc_r) >= abs(acc_IndepVerif));
        clear i label_rand train_data train_label test_label test_data model label_r acc_r predicted_label accuracy deci;
        
        %% Save the results in the 100 iterations  
        train_permut(mn,:) = TrainTest';
        test_permut(mn,:) = IndepVerif';
        indices_permut(mn,:) = indices;
        indices2_permut(mn,:) = indices2;
        ACC_TrainTest_permut(mn,:) = acc_TrainTest;
        Sensitivty_TrainTest_permut(mn,:) = Sensitivty_TrainTest;
        Specificity_TrainTest_permut(mn,:) = Specificity_TrainTest;
        AUC_TrainTest_permut(mn,:)  = AUC_TrainTest;
        ACC_permut(mn,:) = acc_IndepVerif;
        AUC_permut(mn,:) = AUC;
        Sensitivty_permut(mn,:) = Sensitivty_IndepVerif;
        Specificity_permut(mn,:) = Specificity_IndepVerif;
        p_acc_permut(mn,:) = p_acc;
        p_auc_permut(mn,:) = p_auc;
        feature_permut{mn,:} = feature_identify';
        acc_Feature_permut{mn,:} = acc_Feature;
        X_permut(mn,:) = X;
        Y_permut(mn,:) = Y;
        X_TrainTest_permut(mn,:) = X_TrainTest;
        Y_TrainTest_permut(mn,:) = Y_TrainTest;
        
        clearvars -except NumROI data_all label_all permut Nsloop mn h train_permut test_permut indices_permut indices2_permut ACC_TrainTest_permut ACC_permut AUC_permut Sensitivty_permut Specificity_permut p_acc_permut p_auc_permut feature_permut acc_Feature_permut X_permut Y_permut X_TrainTest_permut Y_TrainTest_permut weight_feature;
end
clear h mn;
save('FD_HS_classfication.mat');

%% Plot the feature selection curve over 100 iterations
figure;
for i = 1:permut       
    Feature_plot = acc_Feature_permut{i,:};
    M = Feature_plot(2,:); N = Feature_plot(1,:); 
    plot(M,N,'.-', 'Linewidth',1.5, 'MarkerSize',10);
    hold on;
end
xlabel('Number of features'); ylabel('Accuracy');

%% Plot the ROC curve over 100 iterations in cross-validation sets
figure;
for i = 1:permut       
    X = X_TrainTest_permut(i,:); Y = Y_TrainTest_permut(i,:);
    plot(X,Y,'.-', 'Linewidth', 2, 'color',[0.8 0.8 0.8]);
    hold on;
end
median = 1; % Select the corresponding X and Y of the median AUC
hold on; plot(X,X,'-','Linewidth', 2, 'color',[0 0 0]); 
plot(X_TrainTest_permut(median,:),Y_TrainTest_permut(median,:),'-','Linewidth', 2, 'color',[0 0 0])
xlabel('False positive rate'); ylabel('True positive rate');

%% Plot the ROC curve over 100 iterations in independent validation set
figure;
for i = 1:permut       
    X = X_permut(i,:); Y = Y_permut(i,:);
    plot(X,Y,'.-', 'Linewidth', 2, 'color',[0.8 0.8 0.8]);
    hold on; 
end
median = 1; % Select the corresponding X and Y of the median AUC
hold on; plot(X,X,'-','Linewidth', 2, 'color',[0 0 0]); 
plot(X_permut(median,:),Y_permut(median,:),'-','Linewidth', 2, 'color',[0 0 0])
xlabel('False positive rate'); ylabel('True positive rate');

%% Plot the consensus feature over 100 iterations
% identify the consensus feature (more than 30% of the maximum weight vector)
for j = 1:permut
weight_plot= weight_feature{:,j};
conn_msk = ones(NumROI);
Ind_01 = find(triu(ones(NumROI),1));
Ind_02 = weight_plot(:,1);
ROIMatrix = zeros(NumROI);
ROIMatrix(Ind_01(Ind_02)) = abs(weight_plot(:,2));
ROIMatrix = ROIMatrix+ROIMatrix';
mean_ROIMatrix(:,:,j)= ROIMatrix;
clear Ind_01 Ind_02;
end

weight = mean(mean_ROIMatrix,3);
max_weight = max(max(weight));
threshold = weight>0.3*max_weight; 
weight_matrix = weight.*double(threshold);
figure;
imagesc(weight_matrix);
hold on; plot([6.5,6.5],[0.5,35.5],'w','linewidth',2);
hold on; plot([16.5,16.5],[0.5,35.5],'w','linewidth',2);
hold on; plot([26.5,26.5],[0.5,35.5],'w','linewidth',2);
hold on; plot([0.5,35.5],[6.5,6.5],'w','linewidth',2);
hold on; plot([0.5,35.5],[16.5,16.5],'w','linewidth',2);
hold on; plot([0.5,35.5],[26.5,26.5],'w','linewidth',2);
axis off; 