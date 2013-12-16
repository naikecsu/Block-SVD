function x=OMPerr(Dic,img_new)
         r=img_new;        %残差        
         D=[];             %空字典
         coeff=[];          %去0系数
         pos_arr=[];         %位置
         Dic_Count=size(Dic,2);
         x=zeros(Dic_Count,1);%完整系数
         Dic2=Dic;
         e=0.05;  
         p=1;
         while sqrt(sum(r.^2))>e&&p<=Dic_Count                   
             for col=1:Dic_Count
                 product(col)=abs(Dic2(:,col)'*r);
             end 
                [val pos]=max(product);
                D=[D,Dic2(:,pos)];
                Dic2(:,pos)=zeros(length(img_new),1); 
                coeff=D\img_new;
                r=img_new-D*coeff;
                pos_arr=[pos_arr;pos];  
                p=p+1;
         end
         x(pos_arr)=coeff;
end