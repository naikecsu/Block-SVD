clear all;
clc;
addpath(genpath('.\ksvdbox'));  % add K-SVD box
addpath(genpath('.\large_scale_svm'));  % add K-SVD box
addpath(genpath('.\OMPbox')); % add sparse coding algorithem OMP
load('.\trainingdata\featurevectorAR1.mat','training_feats', 'testing_feats','H_train','H_test','testing_cell','training_cell');

%% constant
sparsitythres=5; %block sparsity prior 
sparsitythres2=5; %sparsity prior for initialization 
dictsize=500;  % dictionary size        
iterations=10; % iteration number                  
iterations4ini=20; % iteration number for initialization  
% 
%% dictionary initialization   
fprintf('\n Block k-svd initialization... ');
[Dinit] = initializationbksvd(training_feats,H_train,dictsize,iterations4ini,sparsitythres2);
fprintf('done!');


%% dictionary learning process
fprintf('\nDictionary learning by Block k-svd...');
[D1 X] = DictionaryUpdate(training_feats,Dinit,H_train,iterations,sparsitythres,dictsize);  %正式学习字典
save('.\dictionarydata5-1.mat','D1','X');
fprintf('dictionary learning finished!\n');

% classification process
load('.\dictionarydata5-1');
n_correct1 = 0;  
test_num=6;      %the numbeer of test samples for each class
fprintf('\nClassification... ');
[prediction,accuracy,count,x]=classification(D1,testing_feats,H_test,sparsitythres); %分类测试样本
fprintf('\ndone!\n');
 for indTest=1:size(testing_feats,2)           
     [val ind]=max(H_test(:,indTest));
     if(prediction(indTest)==ind)
         n_correct1=n_correct1+1;
    end
 end
n_correct1
accuracy1= n_correct1/size(testing_feats,2);   
fprintf('\nFinal recognition rate for Block-ksvd is : %.03f ', accuracy1);


