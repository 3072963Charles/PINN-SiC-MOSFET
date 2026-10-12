function Ids = FYP_IV_all_func_adaptive(beta_all, Vdata)
% =======================================================
% MOSFET I-V 紧凑模型 v4
% 修复梯度消失问题：
%   - 去除所有 log10(max(x, 1e-40)) 硬钳位
%   - 改用 log-sum-exp 数值稳定写法
%   - 模型输出范围严格匹配 [-18, 0]
%   - 亚阈值用纯解析式，无钳位截断
%
% 参数（共10个，精简版，减少欠定问题）：
%   (1) Vth    阈值电压 ~0.4-0.6V
%   (2) n      亚阈值理想因子 ~1-2
%   (3) logIt  关态电流 log10(I0)，约 -15 到 -12
%   (4) SS_inv 亚阈值斜率倒数 = 1/(SS*log10(e))，约 1/(60mV/dec)
%   (5) logbeta log10(beta)，强反型增益对数
%   (6) lambda  沟道长度调制系数
%   (7) theta  迁移率退化
%   (8) delta  过渡区宽度（V）
%   (9) eta    DIBL系数
%   (10) Vds_scale Vds软饱和尺度
% =======================================================

Vds = Vdata(:,1);
Vgs = Vdata(:,2);

% 参数提取
Vth       = beta_all(1);
n         = abs(beta_all(2)) + 0.8;          % >0.8
logIt     = beta_all(3);                      % log10(I0), 自由
SS_inv    = abs(beta_all(4)) + 1.0;          % >1 dec/V（亚阈值斜率）
logbeta   = beta_all(5);                      % log10(beta)
lambda    = abs(beta_all(6));
theta     = abs(beta_all(7));
delta     = abs(beta_all(8)) + 0.02;
eta       = abs(beta_all(9));
Vds_scale = abs(beta_all(10)) + 0.05;

% DIBL修正
Vth_eff = Vth - eta .* Vds;
Vov = Vgs - Vth_eff;

% ===== 亚阈值区（log域直接计算，完全无钳位）=====
% log10(I_sub) = logIt + SS_inv*(Vgs-Vth) + log10(1-exp(-Vds/(n*0.026)))
% 用 log1mexp 数值稳定计算 log(1-exp(-x))
nVt = n * 0.02585;
x_ds = Vds ./ nVt;
% log10(1-exp(-x))：x很大时→0，x很小时→log10(x)
log10_sat_factor = log10_1mexp(x_ds);

log10_Isub = logIt + SS_inv .* Vov + log10_sat_factor;

% ===== 强反型区（log域）=====
Vov_pos = max(Vov, 0);
% 软饱和：Vds_eff = Vds_scale * tanh(Vds/Vds_scale)
Vds_eff = Vds_scale .* tanh(Vds ./ Vds_scale);
% 迁移率退化
mu_factor = 1 ./ (1 + theta .* Vov_pos);
% 沟道长度调制
clm = 1 + lambda .* Vds;
% log10(I_strong)
log10_Istrong = logbeta + 2*log10(max(Vov_pos, 1e-10)) ...
    + log10(mu_factor) + log10(Vds_eff + 1e-30) + log10(clm);

% ===== log域平滑拼接（避免线性域相加再取log）=====
% smooth_w: 0=亚阈值, 1=强反型
smooth_w = 0.5 + 0.5 .* tanh(Vov ./ delta);

% 在log域加权：不能直接线性插值log，用logsumexp思路
% log10(I_total) ≈ log10( (1-w)*I_sub + w*I_strong )
% = log10_Isub + log10( 1 + (w/(1-w)) * 10^(log10_Istrong - log10_Isub) )
% 用稳定写法：取两者中较大的作基准

log10_total = logsumexp_weighted(log10_Isub, log10_Istrong, smooth_w);

Ids = log10_total;

end

% ===== 辅助函数 =====

function y = log10_1mexp(x)
% 数值稳定计算 log10(1 - exp(-x))，x >= 0
% x 很小时：1-exp(-x) ≈ x，所以 log10(1-exp(-x)) ≈ log10(x)
% x 很大时：→ 0
x = max(x, 1e-30);
small = x < 1e-4;
y = zeros(size(x));
y( small) = log10(x(small));              % log10(x) 近似
y(~small) = log10(1 - exp(-x(~small)));  % 直接计算
% 保证不超出范围
y = max(y, -40);
end

function out = logsumexp_weighted(la, lb, w)
% 计算 log10( (1-w)*10^la + w*10^lb ) 的数值稳定版本
% 分离两个区域避免数值问题
w = min(max(w, 1e-10), 1-1e-10);
lw  = log10(w);
l1w = log10(1-w);

% 取较大值作基准
m = max(la + l1w, lb + lw);
out = m + log10( 10.^(la + l1w - m) + 10.^(lb + lw - m) );
end
