%=============================================
% Group Sparse coding of a group of signals based on a given 
% dictionary and specified number of atoms to use. 
% input arguments: A - the dictionary
%                  y - the signals to represent
%                  group - group indices, should be a vector with dimensionality of col of A
%                  err - the maximal allowed representation error for
%                  each siganl.
% output arguments: x - sparse coefficient .
%===============================================
function x=gomp(param)
% params.data = training_feats;
% params.Tdata = sparsitythres; % spasity term
% para.subdicnum=numPerClass;
% params.classnum =numClass;
% params.initdict = normcols(Dinit);
dic=param.initdict;
trainnum=size(param.data,2);
c=param.classnum;
x = zeros(size(dic,2),trainnum);
traindata=param.data;
sparsity=param.Tdata ;
cnum=param.subdicnum;
g = cell(c,1);
for i=1:trainnum
    i
    for j = 1:c  
       g{j} =(j-1)*cnum+[1:cnum];
    end
    r=traindata(:,i);
    L=[];   
    p=1;
    while p<=sparsity
          l=dic'*r; 
          lg = zeros(c,1);
          for j = 1:c
              if g{j}==-1
                  lg(j)=0;
              else
                  lg(j) = norm(abs(l(g{j})));
              end
          end
          [temp, idx] = sort(lg, 'descend');
          L=[L  g{idx(1)}];
          g{idx(1)}=-1;
          Psi = dic(:,L);
          x_bar =Psi\traindata(:,i);
          p=p+1;
          r = traindata(:,i) - Psi*x_bar;

    end
    x(L,i)=x_bar;
end
end