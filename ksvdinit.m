% ========================================================================
% k-svd algorithm for block initialization
% [D]=ksvdinit(param,varargin)
% Inputs
%       param.initdict  -initialized dictionary
%       param.data      -training data 
%       param.dicnum -number of  atoms for each block
%       param.iternum    -the number of iteration 
% Outputs
%       D                 -learned dictionary
% Author: Yixiong Liang 
% Date: 3-16-2013
% ========================================================================

function D= ksvdinit(params,varargin)
   iter=params.iternum;
   tdata=params.data;
   dic=params.initdict;
   trainnum=size(tdata,2);
   dictnum=params.dicnum;
   x=zeros(dictnum,trainnum);
   for i=1:iter
       for j=1:trainnum
           x(:,j)=OMPerr(dic,tdata(:,j));
       end
       for j=1:dictnum
            index=find(x(j,:));%找到使用第一个原子的训练样本
            if length(index)<1
               ErrorMat=tdata-dic*x;
               ErrorNormVec = sum(ErrorMat.^2);
               [d,p] = max(ErrorNormVec);
               betterDictionaryElement =tdata(:,p);
               betterDictionaryElement = betterDictionaryElement./sqrt(betterDictionaryElement'*betterDictionaryElement);
               dic(:,j) = betterDictionaryElement;
               x(j,:) = 0;
            else
               tempx=x(:,index);
               tempx(j,:)=0;
               errors=tdata(:,index)-dic*tempx;
               [betterDictionaryElement,singularValue,betaVector] = svds(errors,1);
               dic(:,j) = betterDictionaryElement;
               x(j,index) = singularValue*betaVector';         
             end           
       end  
   end
   D=normcols(dic);
end