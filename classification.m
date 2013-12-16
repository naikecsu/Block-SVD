% ========================================================================
% Classification 
% USAGE: [prediction, accuracy, count,x] = classification(D1,testing_feats,H_test,sparsity)
% Inputs
%       D               -learned dictionary
%       testing_feats   -learned classifier parameters
%       H_test          -labels matrix for testing feature 
%       sparsity        -group sparsity factor
% outputs
%       prediction      -predicted labels for testing features
%       accuracy        -classification accuracy
%       count           -Correct classification number
%       x               _sparse codes
%
% Author: Yixiong Liang
% Date: 3-16-2013
% ========================================================================

function [prediction,accuracy,count,x]=classification(D1,testing_feats,H_test,sparsity)
   testnum=size(testing_feats,2);
   dictsize=size(D1,2);
   cnum=size(H_test,1);
   subnum=dictsize/cnum;
   param.initdict=D1;
   param.classnum=cnum;
   param.data=testing_feats;
   param.Tdata=sparsity;
   param.subdicnum=subnum;
   x=zeros(size(D1,2),testnum);
   x=gomp(param);
   count=0;
   Q=zeros(dictsize,cnum);
   indd=0;
   kk=ones(subnum,1);
   for i=1:cnum
       Q(indd+1:indd+subnum,i)=kk;
       indd=indd+subnum;
   end    
   for k=1:testnum     
       ind = 0;
       err_Bsr1=zeros(cnum,1);
       for i = 1:cnum
          err_Bsr1(i) = norm(testing_feats(:,k)- D1(:,ind+1:ind+subnum) * x(ind+1:ind+subnum,k),'fro');
          ind = ind + subnum;
       end   
       [val label]=min(err_Bsr1);
       prediction(k)=label;
       [val2 label2]=max(H_test(:,k));
       if label==label2
          count=count+1;
       end
   end 
   accuracy=count/testnum;
end