% Stage-by-stage outputs of pc_spherical_conformal_map for debugging the port
% (the pipeline inlined from the MATLAB, with intermediates written out).
function make_pc_stages(outdir)
repo = '/Users/dfortunato/Research/sphere-surf/pointcloudsphericalconformalmap';
addpath(fullfile(repo, 'mfile'));

N = 800;
i = (0:N-1)';
z = 1 - 2*(i+0.5)/N;
th = acos(z);
ph = mod(i * 2.399963229728653, 2*pi);
r = 1 + 0.1*sin(3*th).*cos(4*ph);
vertex = [r.*sin(th).*cos(ph), r.*sin(th).*sin(ph), r.*cos(th)];

% also emit david's boundary triple (its final map is already in david_map.bin)
D = load('/Users/dfortunato/Research/sphere-surf/pointcloudsphericalconformalmap/david.mat');
[~, drings] = calc_pc_laplacian(D.vertex, 25);
dface = cell2mat(drings);
dtemp = D.vertex(reshape(dface',1,length(dface)*3),1:3);
de1 = sqrt(sum((dtemp(2:3:end,1:3) - dtemp(3:3:end,1:3))'.^2))';
de2 = sqrt(sum((dtemp(1:3:end,1:3) - dtemp(3:3:end,1:3))'.^2))';
de3 = sqrt(sum((dtemp(1:3:end,1:3) - dtemp(2:3:end,1:3))'.^2))';
dreg = abs(de1./(de1+de2+de3)-1/3)+abs(de2./(de1+de2+de3)-1/3)+abs(de3./(de1+de2+de3)-1/3);
[~,dbig] = min(dreg);
writebin(fullfile(outdir,'david_bd.bin'), dface(dbig,:));

[L, rings] = calc_pc_laplacian(vertex, 25);
numofv = length(vertex);
L = L(1:numofv,1:numofv);
face_temp = cell2mat(rings);
temp = vertex(reshape(face_temp',1,length(face_temp)*3),1:3);
e1 = sqrt(sum((temp(2:3:end,1:3) - temp(3:3:end,1:3))'.^2))';
e2 = sqrt(sum((temp(1:3:end,1:3) - temp(3:3:end,1:3))'.^2))';
e3 = sqrt(sum((temp(1:3:end,1:3) - temp(2:3:end,1:3))'.^2))';
regularity = abs(e1./(e1+e2+e3)-1/3)+abs(e2./(e1+e2+e3)-1/3)+abs(e3./(e1+e2+e3)-1/3);
[~,bigtri] = min(regularity);
bd = face_temp(bigtri,:);
writebin(fullfile(outdir,'bumpy_bd.bin'), bd);

v1 = vertex(bd(1),:); v2 = vertex(bd(2),:); v3 = vertex(bd(3),:);
l12 = norm(v1-v2); l23 = norm(v2-v3); l31 = norm(v1-v3);
angle2 = (l12^2+l23^2-l31^2)/2/l12/l23;
d = vertex(:,1)*0;
d(bd) = [0 l12 -l23*exp(1i*acos(angle2))+l12]/l12*2.6562^2;
d(bd) = d(bd) - mean(d(bd));
writebin(fullfile(outdir,'st_d.bin'), [real(d(bd)), imag(d(bd))]);
[ii,jj,kk] = find(L(bd,:));
L2 = L - sparse(bd(ii),jj,kk,numofv,numofv) + sparse(bd,bd,[1 1 1],numofv,numofv);
z = L2\d;
z = z - mean(z);
map = [2*real(z)./(1+real(z).^2+imag(z).^2), 2*imag(z)./(1+real(z).^2+imag(z).^2), (-1+real(z).^2+imag(z).^2)./(1+real(z).^2+imag(z).^2)];
writebin(fullfile(outdir,'st_map_north.bin'), map);

ratio = 0.2;
[~, index] = sort(map(:,3));
fixed = index(1:floor(length(vertex)*ratio));
w = complex(map(:,1)./(1+map(:,3)) , map(:,2)./(1+map(:,3)));
d = w*0; d(fixed) = w(fixed);
[ii,jj,kk] = find(L(fixed,:));
L2 = L - sparse(fixed(ii),jj,kk,numofv,numofv) + sparse(fixed,fixed,ones(length(fixed),1),numofv,numofv);
w = L2\d;
map = [2*real(w)./(1+abs(w).^2), 2*imag(w)./(1+abs(w).^2), -(abs(w).^2-1)./(1+abs(w).^2)];
writebin(fullfile(outdir,'st_map_south.bin'), map);

[~,id] = sort(map(:,3),'descend');
north = id(1); south = id(end);
north_ring = setdiff(unique((rings{north})),north);
south_ring = setdiff(unique((rings{south})),south);
z = complex(map(:,1)./(1-map(:,3)) , map(:,2)./(1-map(:,3)));
w = complex(map(:,1)./(1+map(:,3)) , map(:,2)./(1+map(:,3)));
NorthTriSide = mean(abs(z(north_ring)-z(north)));
SouthTriSide = mean(abs(w(south_ring)-w(south)));
w = w*(sqrt(NorthTriSide*SouthTriSide))/(SouthTriSide);
map = [2*real(w)./(1+abs(w).^2), 2*imag(w)./(1+abs(w).^2), -(abs(w).^2-1)./(1+abs(w).^2)];
writebin(fullfile(outdir,'st_map_balance.bin'), map);
end

function writebin(path, A)
fid = fopen(path, 'w');
fwrite(fid, size(A), 'uint32');
fwrite(fid, A.', 'double');
fclose(fid);
end
