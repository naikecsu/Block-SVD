% ========================================================================
%  Block ksvd algorithm
% [D X]=blocksvd(param)
% Inputs
%       param.initdict  -dictionary
%       param.classnum  -class number
%       param.data      -training data 
%       param.subdicnum -number of  atoms for each block
%       param.iternum    -the number of iteration 
%       params.dim;      _the dimension of dictionary
% Outputs
%       D               _dictionary
%       X               -sparsed codes
% Author: Yixiong Liang 
% Date: 3-16-2013
% ========================================================================

function [D,X]= blocksvd(params,varargin)  
   c=params.classnum ;           
   cnum=params.subdicnum ;       
   iter=params.iternum;          
   tdata=params.data ;          
   dict=params.initdict;         
   dimen=params.dim;          
   dictnum=c*cnum;             
   trainnum=size(tdata,2);     
   X=zeros(dictnum,trainnum);    
   for i=1:iter   
       i
       X=sbomp(params);                       %using sbomp to perform sparse coding
       ind=1:cnum;                            %the indics for each block
       for j=1:c
           if j~=1
              ind=ind+cnum;     
           end
           index=find(sum(X(ind,:).^2));        %find the traing samples which use the jth block
           if length(index)<1                   %there is no training sample use the jth block
               ErrorMat=tdata-dict*X;           %using the error of all training samples to update 
               ErrorNormVec = sum(ErrorMat.^2);
               [d,p] = maxn(ErrorNormVec,cnum);
               betterDictionaryElement =tdata(:,p);
               betterDictionaryElement = betterDictionaryElement./repmat(sqrt(sum( betterDictionaryElement.^2)),size(betterDictionaryElement,1),1);
               dict(:,ind) = betterDictionaryElement;
               X(ind,:) = 0;
           else                       
               tempx=X(:,index);                 %puck up the training samples that use the jth block  
               tempx(ind,:)=0;                   %set the coefficient to zeros
               errors=tdata(:,index)-dict*tempx;  %obtain the error of training sample that discard the jth block
               [u,s,v] = svd(errors);
               summ=min(size(u,1),size(v,1));
               tempu=u(:,1:summ);
               temps=s(1:summ,1:summ);
               tempv=v(:,1:summ);
               pp=1:cnum;                        
               dict(:,ind) = tempu(:,pp);          %simultaneously the block 
               diagele=diag(temps);
               coeff=getcoeff(diagele,tempv,pp);   %simultaneously sparse codes
               X(ind,index)=coeff';    
           end   
       end
       params.initdict=normcols(dict);    
   end
   D=normcols(dict);    
end


function [d,p]=maxn(vec,cnum)   
    for i=1:cnum
        [val pos]=max(vec);
        d(i)=val;
        p(i)=pos;
        vec(pos)=0;
    end
end


function [d,p]=minn(vec,cnum) 
    for i=1:cnum
        [val pos]=min(vec);
        d(i)=val;
        p(i)=pos;
        vec(pos)=inf;
    end
end


function coeff=getcoeff(diagele,tempv,pp) 
    l=length(pp);
    for i=1:l
        coeff(:,i)=diagele(pp(i))*tempv(:,pp(i));
    end
end

