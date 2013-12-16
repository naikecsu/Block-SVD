% ========================================================================
% Initialization for Block k-svd
% USAGE: [Dinit] = initializationbksvd(training_feats,....
%                               H_train,dictsize,iterations,sparsitythres)
% Inputs
%       training_feats  -training features
%       H_train         -label matrix for training feature 
%       dictsize        -number of dictionary items
%       iterations      -iterations
%       sparsitythres   -sparsity threshold
% Outputs
%       Dinit           -initialized dictionary
%     
% Author: Yixong Liang
% Date: 3-16-2013
% ========================================================================
function [Dinit]=initializationbksvd(training_feats,H_train,dictsize,iterations,sparsitythres)
numClass = size(H_train,1);
numPerClass = round(dictsize/numClass); 
Dinit = []; 
dictLabel = [];
for classid=1:numClass
    classid
    col_ids = find(H_train(classid,:)==1);
    data_ids = find(colnorms_squared_new(training_feats(:,col_ids)) > 1e-6);  
    perm = [1:length(data_ids)]; 
    Dpart = training_feats(:,col_ids(data_ids(perm(1:numPerClass))));   
    para.data = training_feats(:,col_ids(data_ids));                             
    para.Tdata = sparsitythres;
    para.iternum = iterations;
    para.dicnum=numPerClass;
    % normalization
    para.initdict = normcols(Dpart);
    % ksvd process
    Dpart=ksvdinit(para,'');

    Dinit = [Dinit Dpart];
    labelvector = zeros(numClass,1);
    labelvector(classid) = 1;
    dictLabel = [dictLabel repmat(labelvector,1,numPerClass)];
end

