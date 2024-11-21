% ==============================建议分段执行代码，以方便调试==================
clear
clc;
close all;
timeUnit='cut number';
% 刀具编号，以第4把刀具为例，如需要测试其他几组数据，请改为"c3", "c5", "c6"
cutterNo='c4';
% 先将原始.csv格式数据用"csv_to_mat"函数转为.mat格式文件，再执行下面语句，路径名修改为.mat文件所在路径
datastore=fileEnsembleDatastore(...
    fullfile(['D:\Desktop\20241029\Experiments-using-PHM2010dataset-main\Bayesian-MCMC-Prognostics\', cutterNo '_mat']), ...
    '.mat');
% fx, fy, fz, vx, vy, vz, ae分别为x,y,z三个方向切削力、振动及声发射信号
datastore.DataVariables=["fx","fy","fz","vx","vy","vz","ae"];
datastore.IndependentVariables="Date";
datastore.SelectedVariables=["Date","fx","fy","fz","vx","vy","vz","ae"];
datastore.ReadFcn=@helperReadData;
datastore.WriteToMemberFcn=@helperWriteToHSBearing;
Ttable=tall(datastore);
head(Ttable);
fs=50000;

%% 数据展示，rawdataDisplay函数用于展示原始数据，注释该行代码可节约大约5分钟
%  rawdataDisplay(datastore, fs);
%% 特征提取，调用featureExtraction函数进行特征提取
% ============= fx 特征提取================
% fx feature extraction
% 选择要读取的信号通道的名称
SelectedVariables=["Date", "fx"];
features_fx=featureExtraction(datastore, SelectedVariables, fs);
% ============fy feature extraction============
% 选择要读取的信号通道的名称
SelectedVariables = ["Date", "fy"];
features_fy=featureExtraction(datastore, SelectedVariables, fs);
% fz feature extraction
% 选择要读取的信号通道的名称
SelectedVariables=["Date","fz"];
features_fz=featureExtraction(datastore, SelectedVariables, fs);
% vx feature extraction
SelectedVariables=["Date", "vx"];
features_vx=featureExtraction(datastore, SelectedVariables, fs);
% vy feature extraction
SelectedVariables=["Date", "vy"];
features_vy=featureExtraction(datastore, SelectedVariables, fs);
% vz feature extraction
SelectedVariables=["Date", "vz"];
features_vz=featureExtraction(datastore, SelectedVariables, fs);
% ae feature extraction
SelectedVariables=["Date", "ae"];
features_ae=featureExtraction(datastore, SelectedVariables, fs);
%% =====================创建特征列表==========================
for i=1:size(features_fx,2)
    features_fx.Properties.VariableNames{i}=['fx_' features_fx.Properties.VariableNames{i}];
    features_fy.Properties.VariableNames{i}=['fy_' features_fy.Properties.VariableNames{i}];
    features_fz.Properties.VariableNames{i}=['fz_' features_fz.Properties.VariableNames{i}];
    
    features_vx.Properties.VariableNames{i}=['vx_' features_vx.Properties.VariableNames{i}];
    features_vy.Properties.VariableNames{i}=['vy_' features_vy.Properties.VariableNames{i}];
    features_vz.Properties.VariableNames{i}=['vz_' features_vz.Properties.VariableNames{i}];
    
    features_ae.Properties.VariableNames{i}=['ae_' features_ae.Properties.VariableNames{i}];
end

featureTable=[features_fx, features_fy, features_fz, features_vx, features_vy, features_vz, features_ae];
% ================ add the "Date" in the first coloum of feature table=====
featureTableSmooth=varfun(@(x) movmean(x, [5,0]), featureTable);
datastore.SelectedVariables=["Date"];
Date=gather(tall(datastore));
featureTableSmooth=[Date, featureTableSmooth];
%% =======================特征筛选===========================
importscore=0.8;  % 重要性分数阈值，基于斯皮尔曼系数做单调性排序后，得分大于0.8的特征被筛选出来
breaktime=150;  % 首次预测时刻
breakpoint=find(featureTableSmooth.Date<=breaktime, 1, 'last');
trainData=featureTableSmooth(1: breakpoint, :);
trainData_withoutDate=trainData(:, 2:end);
% 特征单调程度排名
featureImportance=monotonicity(trainData, 'WindowSize', 0, 'method', 'rank');
helperSortedBarPlot(featureImportance, 'Monotonicity');
aa=featureImportance{:,:}>importscore;
featureTableSmooth_withoutDate=featureTableSmooth(:,2:end);
featureSelected=featureTableSmooth_withoutDate(1:end, aa);
trainDataSelected=trainData_withoutDate(:, aa);
%% =================基于PCA的特征融合=========================
%主成分分析 (pca) 用于降维和特征融合%
meanTrain = mean(trainDataSelected{:,:});
sdTrain = std(trainDataSelected{:,:});
trainDataNormalized = (trainDataSelected{:,:} - meanTrain)./sdTrain;
coef = pca(trainDataNormalized);

%利用均值、标准偏差和 pca 系数对整个数据集进行处理%
PCA1 = (featureSelected{:,:} - meanTrain) ./ sdTrain * coef(:, 1);
PCA2 = (featureSelected{:,:} - meanTrain) ./ sdTrain * coef(:, 2);

%在前6个主要组件的空间中可视化数据。%
figure
hold on
numData = size(featureSelected, 1);

plot(1:numData,PCA1)
plot(1:numData, PCA2)

hold off
xlabel('Cut Number')
ylabel('Feature Value')
% legend('PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5', 'PCA6')
title('PCA')

%在前两个主要组件的空间中可视化数据。%
figure
numData = size(featureSelected, 1);
scatter(PCA1, PCA2, [], 1:numData, 'filled')
xlabel('PCA 1')
ylabel('PCA 2')
cbar = colorbar;
ylabel(cbar, ['Time (' timeUnit ')'])

%% ======================三次一阶指数平滑构建健康因子===================
healthIndicator = PCA1;
%healthIndicator = movmean(healthIndicator, 10);

healthIndicator = PCA1;
m=size(featureSelected, 1);
alpha = 0.1;
s1 = zeros(m,1);
s1(1,1) = healthIndicator(1,1);
for i=2:m
s1(i) = alpha*healthIndicator(i,1)+(1-alpha)*s1(i-1);
end
healthIndicator = s1;

s2 = zeros(m,1);
s2(1,1) = healthIndicator(1,1);
for i=2:m
s2(i) = alpha*s1(i,1)+(1-alpha)*s2(i-1);
end
healthIndicator = s2;

m=size(featureSelected, 1);
% 平滑常数的范围为[0,1]
s3 = zeros(m,1);
s3(1,1) = healthIndicator(1,1);
for i=2:m
s3(i) = alpha*s2(i,1)+(1-alpha)*s3(i-1);
end
healthIndicator = s3;

% 可视化健康指标%
healthIndicator = healthIndicator - healthIndicator(1);
figure
numData = size(featureSelected, 1);
plot(1:numData, healthIndicator)
xlabel('cut number')
title('Health Indicator')
box
%% =======================RUL 预测====================
%===============Bayesian-MCMC计算框架，核心代码==============
global y ParamName ny np
y_total=healthIndicator;

para0=[4; 0.1]; % a, b的初值
weigh=[0.5; 0.5]; % 建议分布为均匀分布，采样宽度范围[-0.5, 0.5]
TimeUnit='Cycles';
time=[1:length(y_total)]';
threshold=healthIndicator(end);
 
 % define the name of parameters
 ParamName=['a'; 'b'];
 
 % define the start time from where to predict the RUL
 startime=breaktime;
 
 % parameters used in MCMC sampling
 ns=5e4;  % 采样数设置为50000个
 burnIn=0.2; % 烧掉前20%个不稳定采样
 
 % start the predict RUL from startime until 315
 ny=startime;
 np=size(ParamName,1);
 para_old=para0;
 RUL_cell=cell(length(time),1);
 while ny<time(end)
     y=y_total(1:ny);
     ny=length(y);
     samples(:,1)=para_old; %initial samples
     % MODEL函数中计算a,b参数的联合概率密度后验分布
     [~, jpdf_old]=MODEL(para_old, time(1:ny)); % initial joint pdf
     t=time(1:ny);
     % MCMC函数用于实现MCMC采样
     [thetaHat_unique, para_old, jpdf_old]=MCMC(para_old, jpdf_old, weigh, ns, burnIn, t);
    
     % predict RUL at current time ny
     for i=1:size(thetaHat_unique,2)
         k=ny+1;
         a=thetaHat_unique(1,i);
         b=thetaHat_unique(2,i);
         z=0;
         flag=0;
         while z<threshold
             z=-1+a.*exp(b.*k);
             k=k+1;
             if z<0
                 flag=1;
                 break;
             end
         end
         RUL(:,i)=(k-1)-ny;
     end
     RUL_cell{ny}=RUL'; 
     RUL=[];
     ny=ny+1;
     X=sprintf('finished the %d measurement data', ny);
     disp(X);
 end

%% ==============打印结果=========================
RUL_statis=[];
 for i=startime:length(time)-1
     rul=RUL_cell{i};
     RUL_statis=cat(1, RUL_statis, prctile(rul, [5 50 95]));
 end
RUL_statis(:,4)=[startime:length(time)-1]';
% plot the 5% and 95% condidence interval
FaceColor=[0.1010 0.7450 0.980];
downsample_interval=5;
Y=[RUL_statis(1:downsample_interval:end-1, 1), RUL_statis(1:downsample_interval:end-1, 3)-RUL_statis(1:downsample_interval:end-1,1)];
X=[RUL_statis(1:downsample_interval:end-1,4), RUL_statis(1:downsample_interval:end-1,4)];
figure
hold on
ar=area(X,Y);
ar1=ar(1); ar2=ar(2);
ar1.FaceAlpha=0;
ar1.EdgeAlpha=0;
ar2.EdgeAlpha=0;
ar2.FaceAlpha=0.4;
ar2.FaceColor=FaceColor;

linewidth=1.5;
 % plot the real RUL
 plot([0 length(time)], [length(time), 0], 'linewidth', linewidth)

 RUL_statis(:,4)=[startime:length(time)-1];
 % plot the 50% percent estimated RUL
 plot(RUL_statis(1:end-1, 4), RUL_statis(1:end-1, 2), 'color', 'black', 'linewidth', linewidth, 'marker', '*', 'markersize',5);
 
 accuracy_zone=0.2;
 true_rul=[length(time)-startime:-1:1];
 alpha_lamda=[true_rul+accuracy_zone*true_rul; true_rul-accuracy_zone*true_rul]';
 plot(startime:time(end-1), alpha_lamda, 'linewidth', linewidth, 'color', [0 0.5 0], 'linestyle', '--');
 
 box on;
 h=get(gca,'children');
 legend([h(3), h(1), h(4), h(5)], 'RUL预测值', '\alpha=\pm20%','RUL真实值', '95%置信区间');
 set(gca,'fontsize',12, 'fontweight','bold')
 xlim([100,315])
 xlabel('切削次数')
 ylabel('RUL')

 xlim([150, time(end)])
 ylim([0,200])

 % 计算rmse和mae, 从第200时刻开始计算
loc = find(true_rul == time(end)-200);
er=[true_rul(loc : end)]'-RUL_statis(loc:end, 2);
rmse = sqrt(sum((er).^2)/length(er));
mae=sum(abs(er))/length(er);
%% MCMC 采样函数
function [thetaHat_unique, para_output, jpdf_output]=MCMC(para_old, jpdf_old, weigh, ns, burnIn, time)
    for i=2:ns/(1-burnIn)
        % 从建议分布（均匀分布）中产生新的参数样本
        para_new=unifrnd(para_old-weigh, para_old+weigh);
        % 
        [~, jpdf_new]=MODEL(para_new, time);
        % 接受-拒绝准则(acceptance/rejection
        % criterion)，如果满足准则，则接受新样本，同时将f_po(theta_new | Y{1:T})赋给f_po(theta_old | Y{1:T})，进入下一循环
        % 否则，拒绝新样本，保留旧样本
        if rand<min(1, jpdf_new/jpdf_old)
            para_old=para_new;
            jpdf_old=jpdf_new;
        end
        samples(:, i)=para_old;
    end  
    para_output=para_old;
    jpdf_output=jpdf_old;
    nBurn=ns/(1-burnIn)-ns;
    thetaHat=samples(:, nBurn+1:end);
    thetaHat_unique=[unique(thetaHat(1,:), 'stable'); unique(thetaHat(2,:), 'stable')];
end

%% 计算贝叶斯后验概率密度函数
function [z, poste]=MODEL(param,t)
    global y ParamName ny np
    for j=1:np
        eval([ParamName(j,:) '=param(j,:);']);
    end
    s=2.9;
    %=========退化模型定义========
    z= -1+a.*exp(b.*t);   
    if length(y)~=length(t); post=[];
    else
        % 计算参数的先验联合概率密度值，参见论文中公式(4)
        prior=lognpdf(param(1), 1, 1e2)*normpdf(param(2), 1,1e2);
        % 计算似然函数，参见论文中公式(5)
        likel=(1./(sqrt(2.*pi).*s)).^ny.*...
            exp(-0.5./s.^2.*(y-z)' * (y-z));
        % 计算参数的后验联合概率密度值，参见论文中公式(7)
        poste=likel.*prior;  % compute the posterior value
    end
end









 