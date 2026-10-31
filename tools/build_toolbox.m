%BUILD_TOOLBOX Package maticl as a toolbox and as an MPM package (MATLAB).
%   Stages a clean package folder build/maticl (resources/mpackage.json,
%   src, assets, examples, README.md, LICENSE) and writes
%     build/maticl-<version>.mltbx  the toolbox (Add-On Manager, File Exchange)
%     build/maticl-<version>.zip    the package folder, for a folder-based
%                                   MATLAB Package Manager repository
%   Tests, tools, fixtures and weights are left out. The version comes from
%   resources/mpackage.json; if $MATICL_VERSION is set (CI sets it from the
%   tag, v0.1.0 -> 0.1.0) it must match.
%
%   If resources/mpackage.json does not exist yet, it is created with
%   mpmcreate (R2024b+), with src/ as the only member folder.
%   CI: .github/workflows/package.yml.

root = fileparts(fileparts(mfilename('fullpath')));
id = "c1250ebe-6bf0-492c-a465-18ef1ac70bbc";   % fixed across versions
summary = "TabICL v2 tabular foundation model (classification and regression) in pure MATLAB";

def = fullfile(root, 'resources', 'mpackage.json');
if ~isfile(def)
    pkg = mpmcreate("maticl", root, ID=id, Version="0.1.0", DisplayName="maticl", ...
        Summary=summary, IncludeSubfolders=false, Install=false);
    addFolder(pkg, fullfile(root, 'src'));
    fprintf('Created %s:\n%s\n', def, fileread(def));
end
meta = jsondecode(fileread(def));
version = string(meta.version);
tag = string(getenv('MATICL_VERSION'));
if strlength(tag) > 0 && tag ~= version
    error('build_toolbox:version', 'Tag version %s does not match mpackage.json (%s).', ...
        tag, version);
end

% stage the package folder
build = fullfile(root, 'build');
stage = fullfile(build, 'maticl');
if isfolder(stage)
    rmdir(stage, 's');
end
mkdir(stage);
items = {'resources', 'src', 'assets', 'examples', 'README.md', 'LICENSE'};
for k = 1:numel(items)
    copyfile(fullfile(root, items{k}), fullfile(stage, items{k}));
end
zip_file = fullfile(build, sprintf('maticl-%s.zip', version));
zip(zip_file, 'maticl', build);

opts = matlab.addons.toolbox.ToolboxOptions(stage, id);
opts.ToolboxName = "maticl";
opts.ToolboxVersion = version;
opts.Summary = summary;
opts.AuthorName = "maticl contributors";
opts.ToolboxMatlabPath = fullfile(stage, 'src');
opts.OutputFile = fullfile(build, sprintf('maticl-%s.mltbx', version));
if isprop(opts, 'UseLicenseBSD')                % R2026b+, required by File Exchange
    opts.UseLicenseBSD = true;
end
matlab.addons.toolbox.packageToolbox(opts);
fprintf('Built %s and %s\n', opts.OutputFile, zip_file);
