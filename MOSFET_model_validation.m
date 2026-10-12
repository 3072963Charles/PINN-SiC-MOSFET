%% ================================================================
%  MOSFET I-V 自适应参数模型验证脚本
%  数据来源: Silvaco TCAD 仿真 (Silvaco_IV01_csv1.csv)
%
%  数据特征: VGS = 8.4V (固定), VDS 从 0.2V 扫到 10.0V
%  器件类型: 高压 MOSFET / LDMOS 器件 (亚阈区工作)
%
%  正确物理模型 (基于数据分析推导):
%    I_D = I_sat * tanh(A*(exp(Vds/Vc)-1) / I_sat) * (1 + lambda*Vds)
%
%    物理解释:
%    - 指数项 exp(Vds/Vc): 漂移区场效应导致电流指数上升
%      Vc ~ 1.25V 对应高压器件漂移区的特征电压
%    - tanh 饱和: 沟道电流限制 (等效饱和机制)
%    - lambda: 沟道长度调制 (CLM) 系数, 数量级 ~10^-3 V^-1
%
%  参数辨识: 双目标对数域 + 线性域加权最小二乘
%  评估指标: RMSE / MAE / MAPE / R2
%% ================================================================
clear; clc; close all;

%% ─────────────────────────────────────────────
%  1. 数据读取与预处理
%% ─────────────────────────────────────────────
fprintf('>>> 读取 Silvaco 仿真数据...\n');

csv_file = 'Silvaco_IV01_csv1.csv';

T = readtable(csv_file, 'VariableNamingRule', 'preserve');
col_names = T.Properties.VariableNames;

% 自动识别各列
gate_col    = find(contains(col_names, 'Gate Voltage',  'IgnoreCase', true) & ...
                  ~contains(col_names, 'Int',           'IgnoreCase', true), 1);
drain_v_col = find(contains(col_names, 'Drain Voltage', 'IgnoreCase', true) & ...
                  ~contains(col_names, 'Int',           'IgnoreCase', true), 1);
drain_i_col = find(contains(col_names, 'Drain Current', 'IgnoreCase', true), 1);

Vgs_data = T{:, gate_col};
Vds_data = T{:, drain_v_col};
Ids_data = abs(T{:, drain_i_col});

% 按 Vds 升序排列
[Vds_data, sort_idx] = sort(Vds_data);
Ids_data = Ids_data(sort_idx);
Vgs_fixed = mean(Vgs_data);

fprintf('  数据点总数:    %d\n',   numel(Ids_data));
fprintf('  V_GS 值:       %.2f V\n', Vgs_fixed);
fprintf('  V_DS 范围:     %.2f ~ %.2f V\n', min(Vds_data), max(Vds_data));
fprintf('  I_DS 范围:     %.4e ~ %.4e A\n', min(Ids_data), max(Ids_data));
fprintf('\n  器件分析: I_DS ~ nA 级 → 高压 MOSFET 亚阈工作 (V_th >> 8.4V)\n');

%% ─────────────────────────────────────────────
%  2. 模型函数定义
%     I = I_sat * tanh(A*(exp(Vds/Vc)-1)/I_sat) * (1+lam*Vds)
%     参数: p = [A, Vc, I_sat, lambda]
%% ─────────────────────────────────────────────
I_model = @(Vds, p) ...
    p(3) .* tanh( max(p(1) .* (exp(Vds ./ p(2)) - 1), 0) ./ p(3) ) ...
         .* (1 + p(4) .* Vds);

%% ─────────────────────────────────────────────
%  3. 分层参数初始估计
%% ─────────────────────────────────────────────
fprintf('\n>>> 分层参数初始估计...\n');

% Level 1: I_sat ~ 大 Vds 处均值
I_sat_est = mean(Ids_data(Vds_data > 7));
fprintf('  Level 1 - I_sat:   %.4e A\n', I_sat_est);

% Level 2: Vc ~ 过渡区特征电压
Vc_est = 1.25;
fprintf('  Level 2 - Vc:      %.4f V\n', Vc_est);

% Level 3: lambda ~ 饱和区斜率
mask_sat = Vds_data > 7;
if sum(mask_sat) >= 2
    vds_s = Vds_data(mask_sat);
    ids_s = Ids_data(mask_sat);
    lambda_est = (ids_s(end)-ids_s(1)) / (ids_s(1)*(vds_s(end)-vds_s(1)+1e-10));
    lambda_est = max(lambda_est, 1e-5);
else
    lambda_est = 4e-4;
end
fprintf('  Level 3 - lambda:  %.4e V^-1\n', lambda_est);

% Level 4: A 初始值
A_est = I_sat_est * 0.04;
fprintf('  Level 4 - A:       %.4e A\n', A_est);

p0 = [A_est; Vc_est; I_sat_est; lambda_est];

%% ─────────────────────────────────────────────
%  4. 多目标加权最小二乘拟合
%% ─────────────────────────────────────────────
fprintf('\n>>> 运行优化 (Levenberg-Marquardt)...\n');

lb = [1e-14; 0.1;  1e-9;  0     ];
ub = [1e-5;  5.0;  5e-7;  0.02  ];

% 双域残差: 对数域 (保证小电流精度) + 线性相对残差 (保证大电流精度)
w_log = 2.0;
w_lin = 1.0;

res_fn = @(p) [ w_log * (log10(max(I_model(Vds_data, p), 1e-25)) - ...
                         log10(max(Ids_data,              1e-25))); ...
                w_lin * (I_model(Vds_data, p) - Ids_data) ./ max(Ids_data, 1e-25) ];

opts = optimoptions('lsqnonlin', ...
    'Algorithm',              'levenberg-marquardt', ...
    'MaxIterations',          3000, ...
    'MaxFunctionEvaluations', 100000, ...
    'FunctionTolerance',      1e-14, ...
    'StepTolerance',          1e-14, ...
    'Display',                'iter', ...
    'FiniteDifferenceStepSize', 1e-8);

tic;
[p_fit, ~, ~, exitflag, output] = lsqnonlin(res_fn, p0, lb, ub, opts);
t_fit = toc;

fprintf('\n  耗时 %.3f 秒, 迭代 %d 次, 退出: %d\n', t_fit, output.iterations, exitflag);

%% ─────────────────────────────────────────────
%  5. 精度指标计算
%% ─────────────────────────────────────────────
fprintf('\n>>> 精度评估...\n');

Ids_init = I_model(Vds_data, p0);
Ids_fit  = I_model(Vds_data, p_fit);

[rmse_i, mae_i, mape_i, r2_i] = calc_metrics(Ids_data, Ids_init);
[rmse_f, mae_f, mape_f, r2_f] = calc_metrics(Ids_data, Ids_fit);

fprintf('\n  +---------------+--------------+----------------------+\n');
fprintf('  | 指标          |  初始参数    |  拟合参数            |\n');
fprintf('  +---------------+--------------+----------------------+\n');
fprintf('  | RMSE (A)      |  %9.3e  |  %9.3e           |\n', rmse_i, rmse_f);
fprintf('  | MAE  (A)      |  %9.3e  |  %9.3e           |\n', mae_i,  mae_f);
fprintf('  | MAPE (%%)      |  %9.4f  |  %9.4f           |\n', mape_i, mape_f);
fprintf('  | R2            |  %9.6f  |  %9.6f           |\n', r2_i,   r2_f);
fprintf('  +---------------+--------------+----------------------+\n');

if     mape_f < 5,  fprintf('  >> 精度: 优秀 (MAPE < 5%%)\n');
elseif mape_f < 10, fprintf('  >> 精度: 良好 (MAPE < 10%%)\n');
else,               fprintf('  >> 精度: 一般 (MAPE >= 10%%)\n');
end

%% 参数打印
fprintf('\n  +--------------------+---------------+------------------+\n');
fprintf('  | 参数               |  初始值       |  拟合值          |\n');
fprintf('  +--------------------+---------------+------------------+\n');
pnames = {'A  (A)',  'Vc (V)', 'Isat (A)', 'lam (1/V)'};
for k = 1:4
    fprintf('  | %-18s  |  %10.4g  |  %10.4g       |\n', pnames{k}, p0(k), p_fit(k));
end
fprintf('  +--------------------+---------------+------------------+\n');
fprintf('\n  Vds_sat 拐点 ~ %.3f V\n', p_fit(2)*log(p_fit(3)/p_fit(1)+1));

%% ─────────────────────────────────────────────
%  6. 可视化
%% ─────────────────────────────────────────────
Vds_plot = linspace(0.01, max(Vds_data)*1.02, 1000)';
I_curve_init = I_model(Vds_plot, p0);
I_curve_fit  = I_model(Vds_plot, p_fit);
I_drive_plot = p_fit(1) .* (exp(Vds_plot ./ p_fit(2)) - 1);

c_data = [0.10 0.78 1.00];
c_init = [1.00 0.60 0.15];
c_fit  = [0.15 0.95 0.55];
c_drv  = [1.00 0.35 0.35];

figure('Name','MOSFET I-V 验证','NumberTitle','off', ...
       'Position',[80 60 1500 900],'Color','k');

%% 子图1: 线性坐标
ax1 = subplot(2,3,1); hold on; grid on; box on;
plot(Vds_data, Ids_data*1e9,    'o','Color',c_data,'MarkerSize',6,'MarkerFaceColor',c_data,'DisplayName','Silvaco TCAD');
plot(Vds_plot, I_curve_init*1e9,'--','Color',c_init,'LineWidth',1.8,'DisplayName',sprintf('初始 MAPE=%.1f%%',mape_i));
plot(Vds_plot, I_curve_fit*1e9, '-','Color',c_fit, 'LineWidth',2.5,'DisplayName',sprintf('拟合 MAPE=%.2f%%',mape_f));
Vds_knee = p_fit(2)*log(p_fit(3)/p_fit(1)+1);
xline(Vds_knee,':','Color',[1 1 0.4],'LineWidth',1.5,'DisplayName',sprintf('V_{sat}=%.2fV',Vds_knee));
set(ax1,'Color','k','XColor','w','YColor','w','GridColor',[0.3, 0.3, 0.3]);
xlabel('V_{DS} (V)','Color','w','FontSize',11);
ylabel('I_{DS} (nA)','Color','w','FontSize',11);
title(sprintf('输出特性  V_{GS}=%.1fV',Vgs_fixed),'Color','w');
l=legend('show','Location','southeast','FontSize',9);
set(l,'TextColor','w','Color',[0.15, 0.15, 0.15]);

%% 子图2: 对数坐标
ax2 = subplot(2,3,2); hold on; grid on; box on;
semilogy(Vds_data, Ids_data*1e9,    'o','Color',c_data,'MarkerSize',6,'MarkerFaceColor',c_data,'DisplayName','Silvaco TCAD');
semilogy(Vds_plot, max(I_curve_init,1e-20)*1e9,'--','Color',c_init,'LineWidth',1.8,'DisplayName','初始参数');
semilogy(Vds_plot, max(I_curve_fit, 1e-20)*1e9, '-','Color',c_fit, 'LineWidth',2.5,'DisplayName','拟合参数');
set(ax2,'Color','k','XColor','w','YColor','w','GridColor',[0.3, 0.3, 0.3]);
xlabel('V_{DS} (V)','Color','w','FontSize',11);
ylabel('I_{DS} (nA) [对数]','Color','w','FontSize',11);
title('对数坐标','Color','w');
l=legend('show','Location','southeast','FontSize',9);
set(l,'TextColor','w','Color',[0.15, 0.15, 0.15]);

%% 子图3: 分量可视化
ax3 = subplot(2,3,3); hold on; grid on; box on;
plot(Vds_plot, I_curve_fit*1e9,  '-','Color',c_fit, 'LineWidth',3.0,'DisplayName','总电流 I_D');
plot(Vds_plot, I_drive_plot*1e9,'--','Color',c_drv, 'LineWidth',1.5,'DisplayName','驱动项 A(e^{V/Vc}-1)');
plot(Vds_plot, p_fit(3)*ones(size(Vds_plot))*1e9,':', ...
     'Color',[0.8, 0.5, 1],'LineWidth',1.5,'DisplayName','I_{sat} 饱和上限');
plot(Vds_data, Ids_data*1e9,'o','Color',c_data,'MarkerSize',5,'MarkerFaceColor',c_data,'DisplayName','TCAD');
plot(Vds_knee, I_model(Vds_knee,p_fit)*1e9,'p','Color',[1 1 .3], ...
     'MarkerSize',12,'MarkerFaceColor',[1 1 .3],'DisplayName',sprintf('拐点 %.2fV',Vds_knee));
set(ax3,'Color','k','XColor','w','YColor','w','GridColor',[0.3, 0.3, 0.3]);
xlabel('V_{DS} (V)','Color','w','FontSize',11);
ylabel('I_{DS} (nA)','Color','w','FontSize',11);
title('物理分量分解','Color','w');
l=legend('show','Location','east','FontSize',8);
set(l,'TextColor','w','Color',[0.15, 0.15, 0.15]);

%% 子图4: 残差分布
ax4 = subplot(2,3,4); hold on; grid on; box on;
rel_i = (Ids_init - Ids_data)./Ids_data*100;
rel_f = (Ids_fit  - Ids_data)./Ids_data*100;
plot(Vds_data,rel_i,'s','Color',c_init,'MarkerSize',7,'MarkerFaceColor',c_init,'DisplayName',sprintf('初始 %.1f%%',mape_i));
plot(Vds_data,rel_f,'o','Color',c_fit, 'MarkerSize',7,'MarkerFaceColor',c_fit, 'DisplayName',sprintf('拟合 %.2f%%',mape_f));
yline(0,'--w','LineWidth',1.5);
yline( 5,':','Color',[0.7, 0.7, 0.7],'LineWidth',1);
yline(-5,':','Color',[0.7, 0.7, 0.7],'LineWidth',1);
set(ax4,'Color','k','XColor','w','YColor','w','GridColor',[0.3, 0.3, 0.3]);
xlabel('V_{DS} (V)','Color','w','FontSize',11);
ylabel('相对误差 (%)','Color','w','FontSize',11);
title('相对残差分布','Color','w');
l=legend('show','Location','northeast','FontSize',9);
set(l,'TextColor','w','Color',[0.15, 0.15, 0.15]);

%% 子图5: 散点相关图
ax5 = subplot(2,3,5); hold on; grid on; box on;
Ir = [min(Ids_data),max(Ids_data)]*1e9;
plot(Ir,Ir,'--w','LineWidth',1.5,'DisplayName','理想线');
plot(Ir,Ir*1.05,':','Color',[0.6, 0.6, 0.6],'LineWidth',.8);
plot(Ir,Ir*0.95,':','Color',[0.6, 0.6, 0.6],'LineWidth',.8);
scatter(Ids_data*1e9,Ids_init*1e9,50,c_init,'filled','MarkerFaceAlpha',.7,'DisplayName',sprintf('初始 R^2=%.4f',r2_i));
scatter(Ids_data*1e9,Ids_fit*1e9, 50,c_fit, 'filled','MarkerFaceAlpha',.7,'DisplayName',sprintf('拟合 R^2=%.4f',r2_f));
set(ax5,'Color','k','XColor','w','YColor','w','GridColor',[0.3, 0.3, 0.3]);
xlabel('I_{meas} (nA)','Color','w','FontSize',11);
ylabel('I_{model} (nA)','Color','w','FontSize',11);
title('模型 vs TCAD 散点','Color','w');
l=legend('show','Location','northwest','FontSize',9);
set(l,'TextColor','w','Color',[0.15, 0.15, 0.15]);

%% 子图6: 逐点误差柱状图
ax6 = subplot(2,3,6); hold on; grid on; box on;
abs_err = abs(rel_f);
cdata = zeros(numel(abs_err),3);
for k=1:numel(abs_err)
    if abs_err(k)<5,  cdata(k,:)=[.15 .95 .55];
    elseif abs_err(k)<10, cdata(k,:)=[1.0 .85 .1];
    else, cdata(k,:)=[1.0 .35 .35]; end
end
b=bar(Vds_data,abs_err,'FaceColor','flat'); b.CData=cdata;
yline(5, '--','Color',[1 1 .4],'LineWidth',1.5,'Label','5%');
yline(10,'--','Color',[1 .5 .2],'LineWidth',1.5,'Label','10%');
set(ax6,'Color','k','XColor','w','YColor','w','GridColor',[0.3, 0.3, 0.3]);
xlabel('V_{DS} (V)','Color','w','FontSize',11);
ylabel('|误差| (%)','Color','w','FontSize',11);
title('逐点绝对误差  (绿<5%  黄<10%  红>=10%)','Color','w');

sgtitle(sprintf('MOSFET I-V 模型验证 | Silvaco TCAD | VGS=%.1fV | MAPE=%.2f%% | R2=%.5f', ...
    Vgs_fixed, mape_f, r2_f), 'Color','w','FontSize',13,'FontWeight','bold');
set(gcf,'Color','k');

%% ─────────────────────────────────────────────
%  7. 导出结果
%% ─────────────────────────────────────────────
writetable(table(Vds_data, Ids_data, Ids_init, Ids_fit, rel_i, rel_f, abs_err, ...
    'VariableNames',{'Vds_V','Ids_meas_A','Ids_init_A','Ids_fit_A', ...
                     'RelErr_init_pct','RelErr_fit_pct','AbsErr_pct'}), ...
    'MOSFET_fit_results.csv');

writetable(table({'A';'Vc';'Isat';'lambda'}, {'A';'V';'A';'V^-1'}, ...
    num2cell(p0), num2cell(p_fit), ...
    'VariableNames',{'Param','Unit','Initial','Fitted'}), ...
    'MOSFET_fit_params.csv');

fprintf('\n>>> 输出文件:\n');
fprintf('  MOSFET_fit_results.csv\n');
fprintf('  MOSFET_fit_params.csv\n');
fprintf('>>> 完成。\n');


%% ================================================================
%  辅助函数
%% ================================================================
function [rmse,mae,mape,r2] = calc_metrics(y_t, y_p)
    y_t = max(y_t, 1e-30);
    y_p = max(y_p, 0);
    e   = y_p - y_t;
    rmse = sqrt(mean(e.^2));
    mae  = mean(abs(e));
    mape = mean(abs(e./y_t))*100;
    r2   = 1 - sum(e.^2)/(sum((y_t-mean(y_t)).^2)+1e-40);
end
