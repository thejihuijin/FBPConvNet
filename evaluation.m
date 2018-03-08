% evaluation - FBPConvNet
% modified from MatconvNet (ver.23)
% 22 June 2017
% contact : Kyong Jin (kyonghwan.jin@gmail.com)

clear
restoredefaultpath
reset(gpuDevice(1))
run ./matconvnet-1.0-beta23/matlab/vl_setupnn
%%
%load preproc_x20_ellipse_fullfbp.mat
load dft_mask_75.mat
%load('./training_result/14-Feb-2018_fbpconvnet_ellipse_fullfbp/net-epoch-151.mat');
load('./training_result/19-Feb-2018_fbpconvnet_ellipse_RS_75/net-epoch-151.mat')
%%
%outpath = '12_Feb_2018_results/rsNet_';
cmode='gpu'; % 'cpu'
if strcmp(cmode,'gpu')
    net = vl_simplenn_move(net, 'gpu') ;
else
    net = vl_simplenn_move(net, 'cpu') ;
end
output_imgs = zeros(512,512,1,25);

%%

avg_psnr_m=zeros(25,1);
avg_psnr_rec=zeros(25,1);
for iter=476:500
    gt=lab_n(:,:,1,iter);
    m=rs_images(:,:,1,iter);
    %m=rs_images(:,:,1,iter);
    if strcmp(cmode,'gpu')
        res=vl_simplenn_fbpconvnet_eval(net,gpuArray((single(m))));
        rec=gather(res(end-1).x)+m;
    else
        res=vl_simplenn_fbpconvnet_eval(net,((single(m))));
        rec=(res(end-1).x)+m;
    end
    
    snr_m=computeRegressedSNR(m,gt);
    snr_rec=computeRegressedSNR(rec,gt);
    %figure(1), 
    %subplot(131), imagesc(m),axis equal tight, title({'fbp';num2str(snr_m)})
    %subplot(132), imagesc(rec),axis equal tight, title({'fbpconvnet';num2str(snr_rec)})
    %subplot(133), imagesc(gt),axis equal tight, title(['gt ' num2str(iter)])
    %pause(0.1)
    
    output_imgs(:,:,1,iter-475)=rec;
    avg_psnr_m(iter-475)=snr_m;
    avg_psnr_rec(iter-475)=snr_rec;
end

display(['avg SNR (FBP) : ' num2str(mean(avg_psnr_m))])
display(['avg SNR (FBPconvNet) : ' num2str(mean(avg_psnr_rec))])

%%
save('2-19-18_rs_75_net_151.mat','output_imgs');
%%
figure; 
i=1;
subplot(121)
imagesc(output_imgs(:,:,1,i))
subplot(122)
imagesc(lab_n(:,:,1,i))

