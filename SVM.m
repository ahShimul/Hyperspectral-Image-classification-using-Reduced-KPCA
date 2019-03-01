
addpath('\DataSet');
resultPCA = [];

load KPCAtrain.txt;
train = KPCAtrain;
clear KPCAtrain;
load KPCAtest.txt;
test=KPCAtest;
clear KPCAtest;

label_train = (train(:,1))';
train(:,1:2)=[];
% train = train(:,1:10);
train = train(:,:);

label_test = (test(:,1))';
test(:,1:2)=[];
% train = train(:,1:10);
test = test(:,:);



TrainingSample=train(:,:);
TrainingLabel=label_train(:);
TrainingLabel= TrainingLabel';

TestSample=test(:,:);
TestLabels=label_test(:);
TestLabels=TestLabels';

numClass=max(grp2idx(label_train(:)));

DM=zeros(numClass,sum(label_test));

Total_Accuracy = 0;
acc=zeros(numClass,1);


for i=1:numClass
    
    sigma=2.^(-5:0.1:5);
    cost=2.^(-5:0.1:5);
    %%% For one-against-all convert other classes to single class
    TrainingLabelnew= TrainingLabel;
    TrainingLabelnew = grp2idx(TrainingLabelnew);
    for j=1:length(TrainingLabelnew)
        if(TrainingLabelnew(j)==i)
            TrainingLabelnew(j)=i;
        else
            TrainingLabelnew(j)=i+1;
        end
    end
    TestLabelsnew= TestLabels;
    TestLabelsnew= grp2idx(TestLabelsnew);
    for j=1:length(TestLabelsnew)
        if(TestLabelsnew(j)==i)
            TestLabelsnew(j)=i;
        else
            TestLabelsnew(j)=i+1;
        end
    end
    
    %%% find best sigma and C
    numFolds=10;
    indices = crossvalind('Kfold',TrainingLabelnew,numFolds);
    
    for f=1:length(cost)
        for l=1:length(sigma)
            for k=1:numFolds
                TestingFoldSample=TrainingSample(indices==k,:);
                TrainingFoldSample=TrainingSample(indices~=k,:);
                TrainingFoldLabel=TrainingLabelnew(indices~=k,:);
                svmStruct1=svmtrain(TrainingFoldSample,TrainingFoldLabel,'showplot',0,'kernel_function','rbf','rbf_sigma',sigma(l), 'boxconstraint',cost(f));
                outLabel1(indices==k,1)=svmclassify(svmStruct1,TestingFoldSample,'showplot',0);
            end
            n=length(TrainingLabelnew(:,1));
            accFold(f,l)=mean(grp2idx(outLabel1(1:n,1))==grp2idx(TrainingLabelnew));
        end
    end
    [maxCol, Icol]=max(accFold,[],1);
    [m, Irow]=max(maxCol,[],2);
    bestSigma=sigma(Irow);
    bestC=cost(Icol(Irow));
    %%% Train and classify with best sigma and C
    svmStruct=svmtrain(TrainingSample,TrainingLabelnew,'showplot',0,'kernel_function','rbf','rbf_sigma',bestSigma,'boxconstraint',bestC);
    outLabel=svmclassify(svmStruct, TestSample,'showplot',0);
    Accuracy=mean(grp2idx(outLabel)==grp2idx(TestLabelsnew))*100;
    Total_Accuracy = Total_Accuracy + Accuracy;
    acc(i)=Accuracy;
 
    clear TrainingLabelnew;
    clear TestLabelsnew;
end
Total_Accuracy/numClass

accRKPCA=acc;
accKPCA=acc;

accuracy_RKPCA= sum(accRKPCA)/10;
accuracy_KPCA= sum(accKPCA)/10;
