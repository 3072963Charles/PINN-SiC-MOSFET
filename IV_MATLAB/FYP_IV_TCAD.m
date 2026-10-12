
clc; clear all;

fid01=importdata('Silvaco_IV01.csv1.csv');
total_row01=size(fid01.data,1);
data01=fid01.data;
Vgs01=data01(:,1);
Vds01=data01(:,7);
Ids01=data01(:,9);

fid02=importdata('Silvaco_IV01.csv2.csv');
total_row02=size(fid02.data,1);
data02=fid02.data;
Vgs02=data02(:,1);
Vds02=data02(:,7);
Ids02=data02(:,9);

fid03=importdata('Silvaco_IV01.csv3.csv');
total_row03=size(fid03.data,1);
data03=fid03.data;
Vgs03=data03(:,1);
Vds03=data03(:,7);
Ids03=data03(:,9);

fid04=importdata('Silvaco_IV01.csv4.csv');
total_row04=size(fid04.data,1);
data04=fid04.data;
Vgs04=data04(:,1);
Vds04=data04(:,7);
Ids04=data04(:,9);


Vgs0 = cat(1,Vgs01,Vgs02,Vgs03,Vgs04);
Vds0 = cat(1,Vds01,Vds02,Vds03,Vds04);
Ids0 = cat(1,Ids01,Ids02,Ids03,Ids04)/(5.2e-8);


% data = xlsread("IV_TCAD.xlsx");
% sorted_data_IV_02 = data;
% 
% Vds0 = sorted_data_IV_02(:,1);
% Ids0 = sorted_data_IV_02(:,2);
% Vgs0 = sorted_data_IV_02(:,3);

Vdata0=[Vds0 Vgs0];
%beta0 = [3.443711928939403,5.686636594655933,3.956731131088150,0.267911877913861,9.762057131808618,9.592207989753323];
%beta0 = [3.443711928939403,5.686636594655933,3.956731131088150,0.267911877913861,9.762057131808618,9.592207989753323, 3.9, 4.02, 5, 0.009];
beta0 = [3.443711928939403,5.686636594655933,3.956731131088150,0.267911877913861,9.762057131808618,9.592207989753323, 2.01, 3.5, 1, 0.007,0.7];

%beta0 = [3.443711928939403,5.686636594655933,3.956731131088150,0.267911877913861,9.762057131808618,9.592207989753323, 2.01, 3.5, 1, 0.007,0.7,3,,5,5,5,5,5,5,5,5,5,5,6,4,4,3];
%beta0 = [3.443711928939403,5.686636594655933,3.956731131088150,0.267911877913861,9.762057131808618,9.592207989753323, 2.01, 3.5, 1, 0.007,0.71,60,1,40];
Ids0 = log10(Ids0);
[p0] = lsqcurvefit(@FYP_IV_03_func,beta0,Vdata0,Ids0);
disp('最优拟合参数：');
disp(p0);


Ids1 = FYP_IV_03_func(p0,Vdata0);
TEST=FYP_IV_03_func(p0,[10 10]);
Ids0 = 10.^(Ids0);
Ids1 = 10.^(Ids1);
RMSE0 = sqrt((Ids1-Ids0).^2);
RMSE = sqrt(mean(RMSE0.^2));

MAPE0 = abs((Ids0 - Ids1)./Ids0)*100;
MAPE = mean(MAPE0);
MAPE1 = max(MAPE0);

% 计算总体平方和（Total Sum of Squares, SST）
SST = sum((Ids0 - mean(Ids0)).^2);

% 计算残差平方和（Residual Sum of Squares, SSR）
SSR = sum((Ids0 - Ids1).^2);

% 计算R²
R2 = 1 - (SSR/SST);

% 显示R²值
disp(['R² = ', num2str(R2)]);

%disp(Vpk);
figure(1);
plot3(Vds0,Vgs0,Ids0,'ko');%real
hold on;
plot3(Vds0,Vgs0,Ids1,'c*');%fit
legend('IdsReal','IdsFit');


xlabel('Vds');
ylabel('Vgs');
zlabel('Ids');
grid on;
figure(2);
semilogy(Vds0,RMSE0,'*');%RMSE
figure(3);
semilogy(Vds0,MAPE0,'*');%MAPE

figure(4);
plot(Vds0,Ids0,'ko'); % 实际数据
hold on;
plot(Vds0,Ids1,'c*'); % 拟合数据
legend('IdsReal','IdsFit');
xlabel('Vds');
ylabel('Ids');
% 指定保存的文件名
%filename = 'fitting_results1.xlsx';

% 写入第一条曲线的拟合数据到Sheet1的A列
%sheet = 1;
%xlswrite(filename, Ids1, sheet, 'A1');



