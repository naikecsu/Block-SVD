% ========================================================================
% update dictionary atoms
% USAGE: [D X] =DictionaryUpdate(Y,Dinit,H_train,iterations,sparsitythres,dictsize)
% Inputs
%       Y               -training features
%       Dinit           -initialized dictionary
%       H_train         -label matrix for training feature 
%       dictsize        -number of dictionary items
%       iterations      -iterations
%       sparsitythres   -sparsity threshold
% Outputs
%       D               -learned dictionary
%       X               -sparsed codes
%     
% Author: Yixiong Liang
% Date: 3-16-2013
% ========================================================================
function [D X]=DictionaryUpdate(Y,Dinit,H_train,iterations,sparsitythres,dictsize)
    params.classnum = size(H_train,1);   
    params.subdicnum = round(dictsize/size(H_train,1)); 
    params.data =Y;                      
    params.Tdata = sparsitythres; 
    params.iternum = iterations;          
    D_ext2 = [Dinit];                      
    D_ext2=normcols(D_ext2); 
    params.initdict = D_ext2;
    params.dim=size(Dinit,1);
    params.label=H_train;
    % blocksvd process
    [D,X] = blocksvd(params,'');      
end
