% training FBPConvNet
% modified from MatconvNet (ver.23)
% 22 June 2017
% contact : Kyong Jin (kyonghwan.jin@gmail.com)

clear
reset(gpuDevice(1))
restoredefaultpath
run ./matconvnet-1.0-beta23/matlab/vl_setupnn

%load preproc_x20_ellipse_fullfbp.mat
load dft_mask_75.mat
W   = 512; % size of patch
Nimg= 500; % # of train + test set
Nimg_test= fix(Nimg*0.05);

train_opts.channel_in = 1;
train_opts.channel_out=1;

id_tmp  = ones(Nimg,1);
id_tmp(Nimg-Nimg_test+1:end)=2;

imdb.images.set=id_tmp;             % train set : 1 , test set : 2
imdb.images.noisy=single(rs_images);    % input  : H x W x C x N (X,Y,channel,batch)
imdb.images.orig=single(lab_n);     % output : H x W x C x N (X,Y,channel,batch)
%% check error
means = zeros(500,1);
for ii=1:500
    means(ii) = mean2((lab_n(:,:,1,ii)-rs_images(:,:,1,ii)).^2);
end
display(['Error = ', num2str(mean(means))]);

%%
outpath = ['./training_result/',[num2str(date) '_fbpconvnet_ellipse_RS_75/']]
if (~exist(outpath,'dir')); mkdir(outpath); end

%% check for results.txt
num_results = dir([outpath,'results*.txt']);
numel(num_results)
result_name = [outpath,'results_',num2str(numel(num_results)),'.txt'];
fclose(fopen(result_name,'w+'));
%%
diary(result_name)
diary on
training_start =tic;
opt='none';
train_opts.useGpu = 'true'; %'false'
train_opts.gpus = 1 ;       % []
train_opts.patchSize = 512;
train_opts.batchSize = 1;
train_opts.gradMax = 1e-2;
train_opts.numEpochs = 151 ;
train_opts.momentum = 0.99 ;
train_opts.imdb=imdb;
train_opts.plotStatistics = false;
%train_opts.learningRate=1e-3;

%train_opts.expDir = fullfile('./training_result',[num2str(date) '_fbpconvent_ellipse_fullfbp_'],[opt '_x' num2str(dsr)] ,'/');
train_opts.expDir = fullfile(outpath);

%train_opts.expDir = fullfile('./training_result',[num2str(date) '_fbpconvnet_ellipse_RS_75'] ,'/');

[net, info] = cnn_fbpconvnet(train_opts);
toc(training_start)
diary off


