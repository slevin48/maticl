%% Day-ahead demand forecasting with prediction intervals
% Forecast an hourly electricity-demand series one day ahead with the TabICL
% regressor. Forecasting becomes tabular regression: each hour is a row
% whose features are calendar terms and values known when the forecast is
% issued (the previous day and the previous week). The regressor predicts
% a full distribution per hour, so the forecast comes with quantiles.

here = fileparts(mfilename('fullpath'));
if isempty(which('tabicl_fit'))              % not installed: use this clone
    addpath(fullfile(here, '..', 'src'));
end
rng(2);

%% Simulate 9 weeks of hourly demand
% Daily profile with morning and evening peaks, lower weekends, a slow
% trend, a temperature effect and autocorrelated noise.
H = 24;
days = 63;
T = H * days;
t = (0:T-1)';
hour = mod(t, H);
dow = mod(floor(t / H), 7);                % 0 = Monday
weekend = double(dow >= 5);
profile = 20 * exp(-((hour - 8) .^ 2) / 6) + 30 * exp(-((hour - 19) .^ 2) / 8);
temp = 12 + 8 * sin(2 * pi * (t / (H * 30))) + 3 * sin(2 * pi * (hour - 15) / H);
noise = zeros(T, 1);
for k = 2:T
    noise(k) = 0.8 * noise(k - 1) + 2 * randn;
end
demand = 100 + 0.01 * t + profile .* (1 - 0.35 * weekend) + 1.5 * abs(temp - 15) + noise;

%% Turn the series into a table of features
% For the value at hour t, everything must be known at midnight of the
% day before: lags of at least 24 hours, calendar terms and the temperature
% forecast (here the true temperature).
first = 7 * H + 1;                         % one week of history for lag 168
rows = (first:T)';
X = [sin(2 * pi * hour(rows) / H), cos(2 * pi * hour(rows) / H), ...
    dow(rows), weekend(rows), temp(rows), t(rows), ...
    demand(rows - H), demand(rows - 2 * H), demand(rows - 7 * H)];
y = demand(rows);

% last week for testing, the 7 weeks before for "fitting"
test = rows > T - 7 * H;
Xtr = X(~test, :);
ytr = y(~test);
Xte = X(test, :);
yte = y(test);

%% Forecast with quantiles
R = tabicl_load('regressor');
reg = tabicl_fit_regressor(Xtr, ytr, 'Model', R, 'NumEstimators', 4);
[yhat, out] = tabicl_predict_regressor(reg, Xte, 'Alphas', [0.05 0.5 0.95]);
lo = out.quantiles(:, 1);
hi = out.quantiles(:, 3);

naive = Xte(:, end);                       % seasonal naive: same hour last week
fprintf('day-ahead demand forecast over %d hours\n', numel(yte));
fprintf('  MAE TabICL (mean) : %.2f\n', mean(abs(yhat - yte)));
fprintf('  MAE TabICL (median): %.2f\n', mean(abs(out.median - yte)));
fprintf('  MAE seasonal naive : %.2f\n', mean(abs(naive - yte)));
fprintf('  90%% interval coverage: %.1f%% (mean width %.1f)\n', ...
    100 * mean(yte >= lo & yte <= hi), mean(hi - lo));

%% Plot the test week
th = (0:numel(yte)-1)' / H;
if exist('figure') > 0                     % skipped without graphics (nelson-cli)
    figure;
    fill([th; flipud(th)], [lo; flipud(hi)], [0.8 0.85 1], 'EdgeColor', 'none');
    hold on;
    plot(th, yte, 'k');
    plot(th, out.median, 'b');
    plot(th, naive, ':', 'Color', [0.6 0.6 0.6]);
    hold off;
    legend('90% interval', 'actual', 'TabICL median', 'seasonal naive');
    xlabel('day of the test week');
    ylabel('demand');
    title('Day-ahead forecast');
end
