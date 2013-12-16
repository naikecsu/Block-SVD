% ========================================================================
% Simultaneous Block OMP algorithm
% [X]=sbomp(param)
% Inputs
%       param.initdict  -dictionary
%       param.classnum  -class number
%       param.data      -training data 
%       param.Tdata     -Block sparse factor
%       param.subdicnum -number of  atoms for each block
%       param.label     -the label of training sample
% Outputs
%       X               -sparsed codes
% Author: Yixiong Liang 
% Date: 3-16-2013
% ========================================================================
function x = sbomp(param)
  dic=param.initdict;                  
  trainnum=size(param.data,2);           
  c=param.classnum;                     
  x = zeros(size(dic,2),trainnum);     
  traindata=param.data;                
  sparsity=param.Tdata ;             
  cnum=param.subdicnum;   
  tlabel=param.label;
  g = cell(c,1);
  for j = 1:c     
       g{j} =(j-1)*cnum+[1:cnum];
  end
  for i=1:c     
      index=tlabel(i,:)==1;    
      y=traindata(:,index);    
      residual = y;            
      iter = 1;                
      select_idx = [];          
      while  iter <=sparsity   
           proj =dic'*residual;  
           val_i = zeros(c,1);
           for j=1:c 
               tmp=proj(g{j},:); 
               infnorm=max(abs(tmp));
               val_i(j)=norm(infnorm,1);
           end 
           [tmp, idx_g] = sort(val_i, 'descend');
           selectclass=idx_g(1);               
           select_idx=[select_idx g{selectclass}];
           x_i=dic(:,select_idx)\ y;           
           residual=y-dic(:,select_idx)*x_i;       
           iter = iter + 1;
      end
       x(select_idx,index)=x_i;
  end
end