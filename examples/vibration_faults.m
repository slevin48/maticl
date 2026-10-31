%% Machine fault diagnosis from vibration signals
% Classify rotating-machine faults from accelerometer recordings: healthy,
% imbalance, misalignment and an outer-race bearing defect. Each recording
% is reduced to a handful of classic condition-monitoring features
% (statistics, order amplitudes, envelope spectrum), and TabICL classifies
% the feature table without any training. No toolbox is needed: spectra and
% the analytic signal are computed with fft.

here = fileparts(mfilename('fullpath'));
if isempty(which('tabicl_fit'))              % not installed: use this clone
    addpath(fullfile(here, '..', 'src'));
end
rng(1);

%% Simulate the recordings
% 1 s at 2 kHz. The shaft speed varies between recordings (20-30 Hz) and
% is known from a tachometer, as in most monitoring setups. Fault severity
% and the noise level vary too, so the mild cases are hard.
fs = 2000;
t = (0:fs-1)' / fs;
classes = {'healthy', 'imbalance', 'misalignment', 'bearing'};
per_class = 100;
n = per_class * numel(classes);
signals = zeros(numel(t), n);
speed = zeros(n, 1);
label = zeros(n, 1);
for i = 1:n
    c = ceil(i / per_class);
    f0 = 20 + 10 * rand;
    sev = 0.3 + 1.2 * rand;                         % fault severity
    a = [0.5, 0.1, 0.05] .* (0.7 + 0.6 * rand(1, 3));   % 1x, 2x, 3x orders
    if c == 2
        a(1) = a(1) + sev;
    elseif c == 3
        a(2) = a(2) + 0.8 * sev;
        a(3) = a(3) + 0.3 * sev;
    end
    x = zeros(size(t));
    for k = 1:3
        x = x + a(k) * sin(2 * pi * k * f0 * t + 2 * pi * rand);
    end
    if c == 4
        % impacts at the ball-pass frequency of the outer race (3.58x),
        % each ringing a 600 Hz structural resonance
        bpfo = 3.58 * f0;
        for t0 = rand / bpfo : 1 / bpfo : t(end)
            dt = t - t0;
            on = dt >= 0;
            x(on) = x(on) + 0.6 * sev * exp(-dt(on) / 0.002) .* sin(2 * pi * 600 * dt(on));
        end
    end
    x = x + (0.2 + 0.3 * rand) * randn(size(t));
    signals(:, i) = x;
    speed(i) = f0;
    label(i) = c;
end

%% Extract features
% Per recording: RMS, crest factor, kurtosis, skewness; amplitudes of the
% first three shaft orders relative to the RMS; the envelope spectrum at
% the bearing's ball-pass frequency (band 400-800 Hz around the resonance);
% and the shaft speed.
names = {'rms', 'crest', 'kurtosis', 'skewness', 'order1', 'order2', ...
    'order3', 'env_bpfo', 'env_2bpfo', 'speed'};
X = zeros(n, numel(names));
f = (0:numel(t)-1)' * fs / numel(t);
for i = 1:n
    x = signals(:, i);
    xc = x - mean(x);
    r = sqrt(mean(xc .^ 2));
    S = abs(fft(xc)) / numel(x) * 2;
    orders = zeros(1, 3);
    for k = 1:3
        orders(k) = max(S(abs(f - k * speed(i)) <= 1)) / r;
    end
    % envelope: band-pass by masking the spectrum, analytic signal by
    % zeroing negative frequencies, then the spectrum of its magnitude
    Z = fft(xc);
    band = (f >= 400 & f <= 800);
    Za = zeros(size(Z));
    Za(band) = 2 * Z(band);
    env = abs(ifft(Za));
    E = abs(fft(env - mean(env))) / numel(env) * 2;
    bpfo = 3.58 * speed(i);
    e1 = max(E(abs(f - bpfo) <= 1.5));
    e2 = max(E(abs(f - 2 * bpfo) <= 3));
    X(i, :) = [r, max(abs(xc)) / r, mean(xc .^ 4) / r ^ 4, mean(xc .^ 3) / r ^ 3, ...
        orders, e1 / mean(env), e2 / mean(env), speed(i)];
end
y = classes(label)';

%% Fit on 60% of the recordings, test on the rest
idx = randperm(n);
ntr = round(0.6 * n);
tr = idx(1:ntr);
te = idx(ntr+1:end);

M = tabicl_load();
clf = tabicl_fit(X(tr, :), y(tr), 'Model', M);
yhat = tabicl_predict(clf, X(te, :));
acc = mean(strcmp(string(yhat), string(y(te))));
fprintf('vibration fault diagnosis: %d train / %d test recordings, accuracy %.3f\n', ...
    numel(tr), numel(te), acc);

% confusion matrix (rows: true class, columns: predicted)
C = zeros(numel(classes));
for k = 1:numel(te)
    C(strcmp(classes, y(te(k))), strcmp(classes, yhat(k))) = ...
        C(strcmp(classes, y(te(k))), strcmp(classes, yhat(k))) + 1;
end
disp(array2table(C, 'VariableNames', classes, 'RowNames', classes));

%% Plot one recording per class and the confusion matrix
if exist('figure') > 0                     % skipped without graphics (nelson-cli)
    figure;
    for c = 1:numel(classes)
        subplot(3, 2, c);
        i = find(label == c, 1);
        plot(t(1:400) * 1000, signals(1:400, i));
        title(sprintf('%s (%.1f Hz shaft)', classes{c}, speed(i)));
        xlabel('time (ms)');
    end
    subplot(3, 2, [5 6]);
    imagesc(C);
    colorbar;
    set(gca, 'XTick', 1:numel(classes), 'XTickLabel', classes, ...
        'YTick', 1:numel(classes), 'YTickLabel', classes);
    title(sprintf('confusion matrix, accuracy %.3f', acc));
    xlabel('predicted');
    ylabel('true');
end
