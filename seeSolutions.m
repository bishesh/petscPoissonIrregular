%% Create results: At ../../results/
clear all;clc;
addpath(genpath('/home/bkhanal/Documents/softwares/petsc/bin/matlab/'));
addpath(genpath('/home/bkhanal/Documents/softwares/matlabTools/imshow3D/'));
res_path = '/user/bkhanal/home/works/poissonIrregular/';

fname = 'sol';
dimension = 2;
sysFname = 'sysFile';
% A = full(PetscBinaryRead([res_path sysFname]));
petscObj = PetscReadBinaryMatlab([res_path fname]);

% Solution, rhs, residual
vx = petscObj.x.vx;
bx = petscObj.b.vx;
rx = petscObj.res.vx;

vy = petscObj.x.vy;
by = petscObj.b.vy;
ry = petscObj.res.vy;

% Plot them
if (dimension == 2)
    subplot(121), imagesc(vx), title('solution vx');
    axis image ij;
    subplot(122), imagesc(bx), title('rhs bx');
    axis image ij;
    
    figure,
    subplot(121), imagesc(vy), title('solution vy');
    axis image ij;
    subplot(122), imagesc(by), title('rhs by');
    axis image ij;
end

if(dimension == 3)
    vz = petscObj.x.vz;
    bz = petscObj.b.vz;
    rz = petscObj.res.vz;
    
    figure,
    imshow3D(vx), title('solution vx');
    figure,
    imshow3D(vy), title('solution vy');
    figure,
    imshow3D(vz), title('solution vz');
end
