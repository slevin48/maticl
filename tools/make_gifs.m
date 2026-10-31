%MAKE_GIFS Render the README animations of the examples (MATLAB only).
%   Runs each example, then writes docs/img/two_moons.gif,
%   docs/img/vibration_faults.gif and docs/img/load_forecasting.gif.
%   Frames go through exportgraphics, so it also works without a display
%   (CI: .github/workflows/gifs.yml).

root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'src'));
outdir = fullfile(root, 'docs', 'img');
if ~exist(outdir, 'dir')
    mkdir(outdir);
end
set(groot, 'defaultFigureVisible', 'off');

%% Two moons: the decision boundary as the context grows
run(fullfile(root, 'examples', 'two_moons.m'));
close all;
[g1, g2] = meshgrid(linspace(-1.3, 2.3, 60), linspace(-1, 1.5, 45));
G = [g1(:), g2(:)];
gif = fullfile(outdir, 'two_moons.gif');
fig = figure('Position', [0 0 560 420], 'Color', 'w');
sizes = [4 8 16 32 64 128 200];
for k = 1:numel(sizes)
    ctx = tr(1:sizes(k));
    if numel(unique(y(ctx))) < 2
        continue;
    end
    c = tabicl_fit(X(ctx, :), y(ctx), 'Model', M);
    [~, P] = tabicl_predict(c, G);
    delete(allchild(fig));
    contourf(g1, g2, reshape(P(:, 2), size(g1)), 0:0.1:1, 'LineColor', 'none');
    colormap(fig, interp1([0 0.5 1], [0.85 0.33 0.10; 1 1 1; 0 0.45 0.74], linspace(0, 1, 64)));
    clim([0 1]);
    hold on;
    contour(g1, g2, reshape(P(:, 2), size(g1)), [0.5 0.5], 'k', 'LineWidth', 1.5);
    scatter(X(ctx, 1), X(ctx, 2), 36, y(ctx), 'filled', 'MarkerEdgeColor', 'k');
    hold off;
    axis([-1.3 2.3 -1 1.5]);
    set(gca, 'XTick', [], 'YTick', []);
    title(sprintf('TabICL in context: %d training points, no training', sizes(k)));
    hold_frames = 1 + 2 * (k == numel(sizes));
    for r = 1:hold_frames
        write_frame(fig, gif, k == 1 && r == 1, 1.0);
    end
end
close(fig);

%% Vibration faults: one test recording per class, with the prediction
run(fullfile(root, 'examples', 'vibration_faults.m'));
close all;
[~, P] = tabicl_predict(clf, X(te, :));
gif = fullfile(outdir, 'vibration_faults.gif');
fig = figure('Position', [0 0 720 420], 'Color', 'w');
first = true;
win = round(0.25 * fs);                   % 250 ms window
f = (0:numel(t)-1)' * fs / numel(t);
for c = 1:numel(classes)
    k = find(strcmp(y(te), classes{c}) & strcmp(string(yhat(:)), classes{c}), 1);
    i = te(k);
    x = signals(:, i);
    S = abs(fft(x - mean(x))) / numel(x) * 2;
    p = P(k, :);
    for s = 0:5
        seg = s * round(win / 3) + (1:win);
        delete(allchild(fig));
        subplot(2, 2, [1 2]);
        plot(t(seg) * 1000, x(seg), 'Color', [0 0.45 0.74]);
        ylim([-1 1] * max(abs(x)) * 1.05);
        xlabel('time (ms)');
        title(sprintf('recording %d: %s, shaft %.1f Hz', i, classes{c}, speed(i)));
        subplot(2, 2, 3);
        lim = f <= 400;
        plot(f(lim), S(lim), 'k');
        xlabel('frequency (Hz)');
        title('spectrum');
        subplot(2, 2, 4);
        barh(categorical(clf.classes, clf.classes), p, 'FaceColor', [0.85 0.33 0.10]);
        xlim([0 1]);
        title(sprintf('TabICL: %s (%.2f)', char(yhat(k)), max(p)));
        write_frame(fig, gif, first, 0.25);
        first = false;
    end
end
close(fig);

%% Load forecasting: the test week revealed hour by hour
run(fullfile(root, 'examples', 'load_forecasting.m'));
close all;
gif = fullfile(outdir, 'load_forecasting.gif');
fig = figure('Position', [0 0 720 360], 'Color', 'w');
steps = [6:6:numel(yte), numel(yte) * ones(1, 4)];
for k = 1:numel(steps)
    m = 1:steps(k);
    delete(allchild(fig));
    fill([th(m); flipud(th(m))], [lo(m); flipud(hi(m))], [0.8 0.85 1], 'EdgeColor', 'none');
    hold on;
    plot(th(m), yte(m), 'k', 'LineWidth', 1);
    plot(th(m), out.median(m), 'Color', [0 0.45 0.74], 'LineWidth', 1.5);
    hold off;
    xlim([0 7]);
    ylim([min(lo) - 5, max(hi) + 5]);
    legend('90% interval', 'actual', 'TabICL median', 'Location', 'northwest');
    xlabel('day of the test week');
    ylabel('demand');
    title(sprintf('Day-ahead forecast: MAE %.2f (seasonal naive %.2f)', ...
        mean(abs(out.median(m) - yte(m))), mean(abs(naive(m) - yte(m)))));
    write_frame(fig, gif, k == 1, 0.12);
end
close(fig);
fprintf('GIFs written to %s\n', outdir);

function write_frame(fig, file, first, delay)
% Append the figure to an animated GIF (exportgraphics works headless).
% Every frame is padded or cropped to the size of the first one, since GIF
% frames of different sizes do not line up.
persistent sz
png = [tempname '.png'];
exportgraphics(fig, png, 'Resolution', 96, 'Padding', 'figure');
img = imread(png);
delete(png);
if first
    sz = size(img, [1 2]);
end
out = 255 * ones(sz(1), sz(2), 3, 'uint8');
h = min(sz(1), size(img, 1));
w = min(sz(2), size(img, 2));
out(1:h, 1:w, :) = img(1:h, 1:w, :);
[A, map] = rgb2ind(out, 128, 'nodither');
if first
    imwrite(A, map, file, 'gif', 'LoopCount', Inf, 'DelayTime', delay);
else
    imwrite(A, map, file, 'gif', 'WriteMode', 'append', 'DelayTime', delay);
end
end
