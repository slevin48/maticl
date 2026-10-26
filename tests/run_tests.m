% RUN_TESTS Run the MATLAB TabICL test-suite against the PyTorch reference.
%   Works in MATLAB, Nelson and RunMat:  run('tests/run_tests.m')
here = fileparts(mfilename('fullpath'));
root = fileparts(here);
addpath(fullfile(root, 'src'));
addpath(here);

t0 = tic;
M = tabicl_load();
fprintf('loaded TabICL v2 weights in %.1fs\n', toc(t0));

results = [test_forward(M), ...
    test_classifier(M, 'clf_iris'), ...
    test_classifier(M, 'clf_wine'), ...
    test_classifier(M, 'clf_breast_cancer'), ...
    test_forward(M, 'forward_many'), ...
    test_classifier(M, 'clf_many')];
Mreg = tabicl_load('regressor');
results = cat(2, results, test_regressor(Mreg, 'reg_diabetes'));
fprintf('%d/%d tests passed\n', sum(results), numel(results));
if ~all(results)
    error('tabicl:tests', 'Some tests failed');
end
