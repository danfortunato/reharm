% Reference outputs of the point-cloud spherical conformal pipeline, for
% validating the sh-fit TypeScript port. Binary format per array:
% uint32 rows, uint32 cols, float64 data (row-major).
function make_pc_refs(outdir)
repo = '/Users/dfortunato/Research/sphere-surf/pointcloudsphericalconformalmap';
addpath(fullfile(repo, 'mfile'));
addpath(fullfile(repo, 'spherical_delaunay'));

% small deterministic cloud: golden-spiral sphere with a radial bump
N = 800;
i = (0:N-1)';
z = 1 - 2*(i+0.5)/N;
th = acos(z);
ph = mod(i * 2.399963229728653, 2*pi);
r = 1 + 0.1*sin(3*th).*cos(4*ph);
bumpy = [r.*sin(th).*cos(ph), r.*sin(th).*sin(ph), r.*cos(th)];

S = load(fullfile(repo, 'david.mat'));
david = S.vertex;
fprintf('david: %d points; bumpy: %d points\n', size(david,1), size(bumpy,1));

emit(outdir, 'bumpy', bumpy);
emit(outdir, 'david', david);
end

function emit(outdir, name, vertex)
k = 25;
knnInd = knnsearch(vertex, vertex, 'K', k);
[L, rings] = calc_pc_laplacian(vertex, k);
[li, lj, lv] = find(L);
t0 = tic;
map = pc_spherical_conformal_map(vertex);
tmap = toc(t0);
[~, face] = sphere_delaunay([], map');
face = face';
d = angle_distortion(vertex, face, map);
close all
fprintf('%s: map %.2fs, faces %d, |angle distortion| mean %.4f\n', name, tmap, size(face,1), mean(abs(d)));

writebin(fullfile(outdir, [name '_cloud.bin']), vertex);
writebin(fullfile(outdir, [name '_knn.bin']), knnInd - 1);          % 0-based
writebin(fullfile(outdir, [name '_L.bin']), [li - 1, lj - 1, lv]);  % triplets, 0-based
writebin(fullfile(outdir, [name '_map.bin']), map);
writebin(fullfile(outdir, [name '_faces.bin']), face - 1);          % 0-based
% one-ring triangle sets of the extreme-z map points (the balancing step's
% inputs), flattened; and every local ring triangle for the regularity triple
[~, id] = sort(map(:,3), 'descend');
writebin(fullfile(outdir, [name '_poles.bin']), [id(1) - 1, id(end) - 1]);
ringsAll = cell2mat(rings);
writebin(fullfile(outdir, [name '_rings.bin']), ringsAll - 1);      % 0-based
end

function writebin(path, A)
fid = fopen(path, 'w');
fwrite(fid, size(A), 'uint32');
fwrite(fid, A.', 'double');   % row-major
fclose(fid);
end
