addpath('\DataSet');

Image = load('Indian_pines.mat');

            Image=cell2mat(struct2cell(Image));
          
            [r, c, b] = size(Image);
             newRow = r*c;
            data = reshape(Image,newRow,b);
         

   tic        
 DIST=distanceMatrix(data);
DIST(DIST==0)=inf;
DIST=min(DIST);
para=5*mean(DIST);
K0=kernel(data,'gaussian',para);

image=(data'*K0)';

meanImage = mean(image);
imageAd = bsxfun(@minus,double(image),meanImage);

covMat = cov(imageAd);
[eigVector,eigValue] = eig(covMat,'nobalance');
[eigValue,idx] = sort(diag(eigValue),'descend');
eigVector = eigVector(:,idx(1:1:end));

k=toc;
preFinal = eigVector'*imageAd';
finalData = preFinal';
Iapprox = bsxfun(@plus,meanImage,finalData);
[r1, c1]=size(Iapprox);
Ipca=zeros(newRow,b);
for i=1:c1
    
    Ipca(:,i)=((Iapprox(:,i)-min(Iapprox(:,i)))/(max(Iapprox(:,i))-(min(Iapprox(:,i))))*255);
end


Ipca=uint8(Ipca);
Ipca=reshape(Ipca, r, c, b);

 for i = 1:20
 figure,colormap(gray), imagesc(Ipca(:,:,i));
 end
multibandwrite(Ipca(:,:,1:4),'RKPCA.tif','bsq');


figure;subplot(2,2,1)
colormap(gray)
imagesc(Ipca(:,:,1));
title('Principal component-1');
hold on;
subplot(2,2,2)
colormap(gray)
imagesc(Ipca(:,:,2));
title('Principal component-2');
hold on;
subplot(2,2,3)
colormap(gray)
imagesc(Ipca(:,:,3));
title('Principal component-3');
hold on;
subplot(2,2,4)
colormap(gray)
imagesc(Ipca(:,:,4));
title('Principal component-4');





total = eigValue(1);
cum(1) = eigValue(1);
for i=2:20
    cum(i) = eigValue(i)+cum(i-1);
    total = total+eigValue(i);
end

for i=1:20
    per(i) = (eigValue(i)/total)*100;
    var(i) = (cum(i)/total)*100;
end


a=1:20;
plot(a,var,'r-*');
set(findall(gca, 'Type', 'Line'),'LineWidth',1);
xlabel('Bands');
ylabel('Cummulative variance');
title('Cummulative variance graph');
