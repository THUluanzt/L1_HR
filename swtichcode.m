dbstop if error
% L1-norm code，无多信道
% 录入服务器时修改:
%存储路径datafile_count
%实验次数M 活跃用户个数ka 采样间隔delta_t 信噪比离散序列snr 并行设置parpool&parfor 算法终止迭代条件tol
%张量规模I1&I2&I3及构成设置multiplier
%异常值是否存在outlier_exist 离散异常值个数outlier_amount 自适应分离是否存在semi_seperate_tag
%% 修改 datafile_count存储路径
clear all;clc;
addpath("/home/ztl/tensor_toolbox/");
addpath("/home/ztl/svdandpca");
datafile_count=2;
addpath(['/home/ztl/data' num2str(datafile_count) '/']);

%% 基本参数:实验次数M 载波总数K1 活跃用户数ka 样本容量n（采样间隔为1）
M=10000;%重复生成噪声进行实验的次数
K=64;b=0.64;K1=K/b;%K'载波总数
ka=30; %活跃用户数
Q=100;%Q>=ka
n=K-1;%样本容量
delta_t=1;%采样间隔
huber_gamma=0.001;
tau=10^(-2);
w0=exp(1i*2*pi*delta_t/K1);%元单位根*r(指数上带采样间隔）
w=w0.^((1:(K1-1)).');%表示各分信号的频率%per sgnal带采样间隔

outlier_exist=0;%1表示有outlier，0表示没有,在多信道下的异常值代码未写完，待修改
outlier_amount=10;%异常值个数
semi_seperate_tag=1;%1表示载波自适应分离，0表示不分离
tol=0.002;%l1算法终止条件参数
multiplier=0;%表示构成hankel张量的边乘积为n，若=0则和为n+2
%% 生成星座点的所有可能，存入starset（ka×(4^ka)矩阵）作为其列向量， 用于枚举估计ck
%
% starset_pre=zeros(ka,4^ka);
% x=ones(ka,1);
% count=1;
% upbound=4;
% while count<=upbound^ka
%     flag=-1;
%     starset_pre(:,count)=x;
%     if x(ka)<upbound
%         x(ka)=x(ka)+1;
%     else
%         for former=(ka-1):-1:1
%             if x(former)<upbound
%                 flag=former;
%                 break;
%             end
%         end
%         if flag~=-1
%             x(flag)=x(flag)+1;
%             for later=(flag+1):ka
%                 x(later)=1;
%             end
%         end
%     end
%     count=count+1;
% end
% starlist=[pi/4 pi*3/4 pi*5/4 pi*7/4].';
% starset_value_pre=exp(starlist(starset_pre)*1i);

%% 信噪比设定、参与对比的算法种类及记录数据的数组
% snr=[-9:2:3, 5:1:29];%实验用离散信噪比
snr=0:2:22;%实验用离散信噪比
snr_totaltype=length(snr);

algorithm_totalnumber=4;%算法种数 hosvd  l1_hosvd  hooi  l1_hooi

%每次实验设定参数的记录，包括无噪声信号y_pure 带噪信号y 活跃频道 活跃频道对应的phi
y_pure_record=zeros(n,Q,M,snr_totaltype);%(:,q,j,k)表示在第k种噪声下的第j次实验中生成的第q信道的纯净信号列（n个）
y_record=zeros(n,Q,M,snr_totaltype);%(:,q,j,k)表示在第k种噪声下的第j次实验中生成的第q信道的带噪声信号列（n个）
active_user_record=zeros(ka,M,snr_totaltype);%(:,j,k)表示在第k种噪声下的第j次实验中设定好的活跃用户（取值1-K1）
c_record=zeros(ka,Q,M,snr_totaltype);%(:,q,j,k)表示在第k种噪声下的第j次实验中活跃用户对应的第q信道的ck

%通过hankel算法估计出的参数

%包括matrix和hooi算法得到的活跃频道的zk以及对应的ck，皆为硬判决后的结果，记录之
z_est_hard_record=zeros(ka,M,snr_totaltype,algorithm_totalnumber);%硬判决（频率从小到大排序）
c_est_hard_record=zeros(ka,Q,M,snr_totaltype,algorithm_totalnumber);%硬判决（同zk排序）

%根据估计结果，计算zk的mse和ck的average ser
exper_times=zeros(K1-1,snr_totaltype);%第i行第j列表示在第j种信噪比的M次实验中，用户i被选中的次数
z_mse_hard=zeros(K1-1,snr_totaltype,algorithm_totalnumber);
ser1_hard=zeros(K1-1,Q,snr_totaltype,algorithm_totalnumber);%不考虑zk的正确性
ser2_hard=zeros(K1-1,Q,snr_totaltype,algorithm_totalnumber);%考虑zk的正确性
% %并行开始
MyPar = parpool(12);
parfor ser_count=1:snr_totaltype % 各种信噪比下进行实验，并行
    ser=snr(ser_count);%当前线程的信噪比参数
    e=0.5*10^(-ser/10);%wgn 方差， 由eb/n0确定
    power_wgn=10*log10(e);%生成白噪声用的参数（分贝）
    I1=floor(n/2+1);
    I2=n+1-I1;
    I3=Q;
    V1 = bsxfun(@power,w.',(0:1:I1-1)')/sqrt(I1);
    V2 = bsxfun(@power,w.',(0:1:I2-1)')/sqrt(I2);
    %     starset=starset_pre;
    %     starset_value=starset_value_pre;
    for exper_count=1:M %进行数万次实验，以观察ck和zk的恢复效果
        y_pure=zeros(n,Q);%无噪声信号值初始化
        y=zeros(n,Q); %信号观测值样本点初始化,表示y0到y(n-1)
        noise=wgn(n,Q,power_wgn,'complex');%生成WGN
        a_nonzero_nonsort=randperm(K1-1,ka);%随机选择ka个用户为活跃用户
        a_nonzero=sort(a_nonzero_nonsort).';%按照频率从低到高对这些活跃用户排序，i对应w0^(i)频率的用户
        if semi_seperate_tag==1
            a_nonzero=semi_separate(a_nonzero,K1);%自适应分离，使得它们没有连续三个在一起
        end
        active_user_record(:,exper_count,ser_count)=exp((a_nonzero)*1i*2*pi/K1);%记录当前实验的活跃用户
        
        if outlier_exist==1 %生成异常值
            outlier_nonzero=randperm(n*Q,outlier_amount);
            %             outlier_nonzero=semi_separate(sort(outlier_nonzero).',n*Q);
            Out_random_generate=sqrt(10)*(randn(n,Q)+1i*randn(n,Q));
            out_active=zeros(n,Q);
            out_active(outlier_nonzero)=1;
            tmp_outlier=out_active.*Out_random_generate;
        end
        
        tmp_nonzero_current_exper=zeros(K1-1,1);
        tmp_nonzero_current_exper(a_nonzero)=1;
        exper_times(:,ser_count)=exper_times(:,ser_count)+tmp_nonzero_current_exper;%活跃用户参与实验次数+1
        a=zeros(K1-1,Q);%记录所有用户振幅
        a(a_nonzero,:)=1;%活跃用户振幅为1，非活跃的为0
        phi_count=randsrc(K1-1,Q,randperm(4));%随机赋予各分信号相位phi所在象限 赋值1-4
        angle_or=[pi*5/4 pi*7/4 pi/4 pi*3/4].';%4PSK 的四个星座点对应1-4象限
        phi=angle_or(phi_count);%将赋予的象限变成对应的星座点
        phi=exp(1i*phi);%再变成exp（角）
        tmp=a(a_nonzero,:).*phi(a_nonzero,:);
        c_record(:,:,exper_count,ser_count)=tmp;%记录当前实验活跃用户设定好的ck
        
        for ttime=1:n
            for q=1:Q
                y_pure(ttime,q)=sum(a(:,q).*phi(:,q).*(w.^(ttime-1)));%生成纯净信号
            end
            if outlier_exist==1
                y(ttime,:)=y_pure(ttime,:)+noise(ttime,:)+tmp_outlier(ttime,:);%混入白噪声和异常值
            else
                y(ttime,:)=y_pure(ttime,:)+noise(ttime,:);%混入白噪声
            end
        end
        y_pure_record(:,:,exper_count,ser_count)=y_pure;%记录
        y_record(:,:,exper_count,ser_count)=y;%记录
        
        
        %% 分离信号开始 matrix method(1) & tensor methods - hosvd(2) / huber_hosvd(3) / hooi(4) / huber_hooi(5)
        i1=floor(n/2);
        i2=n+1-i1;%hankel matrix维度
        matrix_combine=cell(1,Q);%记录Q个信道各自的hankel矩阵
        for q=1:Q
            matrix_combine{q}=zeros(i1,i2);
            for j=1:i1
                for k=1:i2
                    matrix_combine{q}(j,k)=y(j+k-1,q);
                end
            end %生成各信道的hankle矩阵H
        end
        H_matrix=[matrix_combine{:}]%将Q个不同信道的hankel矩阵从左到右依次排列，构成一个hankel矩阵构成的矩阵
        
        
        if multiplier==0
            
            H=zeros(I1,I2,I3);%初始化hankel张量
            for j=1:I1
                for k=1:I2
                    for r=1:I3
                        H(j,k,r)=y(j+k-1,r);
                    end
                end
            end
            H_tensor=tensor(H);%生成hankle张量H
        else
            I3=2;
            I2=2;
            I1=n/I3/I2;
            H=reshape(y,[I1 I2 I3]);
            H_tensor=tensor(H);%生成hankle张量H
        end
        %         H_fft_tensor=fft(H,[],3);
        %         H_fft_tensor=tensor(H_fft_tensor);%生成hankle张量H
        
        for algorithm_type=1:4
            switch algorithm_type+1
                case 2
                    %                     disp(['snr=' num2str(ser) ' 第' num2str(exper_count) '次实验 HOSVD'])
                    %                     U=hosvd_k(H_tensor,1,'ranks',[ka ka ka]);
%                     disp(['snr=' num2str(ser) ' 第' num2str(exper_count) '次实验 L1-HOSVD'])
%                     U=l1hosvd_k(H_tensor,1,tol,'ranks',[ka ka ka]);
                    disp(['snr=' num2str(ser) ' 第' num2str(exper_count) '次实验 HUBER-HOSVD'])
                    U=colwise_l1_hosvd_k(H_tensor,1,ka,V1,0.01,'ranks',[ka ka ka]);
                case 3
                    
                    disp(['snr=' num2str(ser) ' 第' num2str(exper_count) '次实验 HUBER-HOSVD'])
                    U=colwise_l1_hosvd_k(H_tensor,2,ka,V1,0.01,'ranks',[ka ka ka]);
                    %                     U=huber_hosvd_k(H_tensor,1,huber_gamma,tau,'ranks',[ka ka ka]);
                case 4
                    %                     disp(['snr=' num2str(ser) ' 第' num2str(exper_count) '次实验 HOOI'])
                    %                     U=tucker_als_1stmode_matrix(H_tensor,ka,'tol',0.01,'init','nvecs');
%                     disp(['snr=' num2str(ser) ' 第' num2str(exper_count) '次实验 L1-HOOI'])
%                     U=L1_tucker_als_1stmode_matrix(H_tensor,ka,'tol',0.002,'init','nvecs');                    
                    disp(['snr=' num2str(ser) ' 第' num2str(exper_count) '次实验 HUBER-HOOI'])
                    U=colwise_l1_tucker_als_1stmode_matrix(H_tensor,1,ka,{V1,V2},'tol',0.002,'init','nvecs');
                    
                case 5
                    disp(['snr=' num2str(ser) ' 第' num2str(exper_count) '次实验 HUBER-HOOI'])
                    U=colwise_l1_tucker_als_1stmode_matrix(H_tensor,2,ka,{V1,V2},'tol',0.002,'init','nvecs');
                    %                     U=huber_hooi_als_1stmode_matrix(H_tensor,ka,huber_gamma,tau,'tol',0.01,'init','nvecs');
            end
            
            %             Z=U(1:end-1,:)\U(2:end,:);% 求解超定方程 得到ka*ka阶方阵Z
            
            U_combine=[U(1:end-1,:) U(2:end,:)];
            [~,~,W]=svd(U_combine);
            W12=W(1:end/2,(end/2+1):end);
            W22=W((end/2+1):end,(end/2+1):end);
            Z=-W12/W22;
            
            
            [~,z_diag]=eig(Z);%求特征值
            z_est=diag(z_diag);
            z_est=z_est./abs(z_est);%标准化，模型中z_k模长均为1
            z_est=z_est.^(1/delta_t);% z_est即为活跃用户的z_k的估计值
            z_angle_est=angle(z_est);%求幅角，取值在-pi到pi之间
            for i=1:ka
                if z_angle_est(i)<0
                    z_angle_est(i)=z_angle_est(i)+2*pi;
                end
                if z_angle_est(i)<=pi/K1
                    z_angle_est(i)=2*pi/K1;
                end
                if z_angle_est(i)>=2*pi*(K1-0.5)/K1
                    z_angle_est(i)=2*pi*(K1-1)/K1;
                end
            end %将所有负幅角+2pi变为正常频率, 0-1.5归到1，48.5-50归到50
            
            [sort_z,~]=sort(z_angle_est);%从小到大对幅角估计值排序，sort_z为排序结果，index_z为在原序列中的序号们
            z_angle_est_hard=exp(1i*2*pi*rem(round(sort_z*K1/2/pi),K1)/K1);%硬判决到某个频道，表示对应频道0-K1中的一个
            z_angle_est_hard_with_sampleinterval=exp(1i*2*pi*delta_t*rem(round(sort_z*K1/2/pi),K1)/K1);%带采样间隔的
            z_est_hard_record(:,exper_count,ser_count,algorithm_type)=z_angle_est_hard;%记录zk硬判决估计结果，按频率从小到大排序
            
            tmp_z_mse_hard_current_exper=zeros(K1-1,1);%同上，用于记录硬判决后的误差
            for i=1:ka
                tmp_z_mse_hard_current_exper(a_nonzero(i))=abs(z_angle_est_hard(i)-w(a_nonzero(i)).^(1/delta_t))^2;
            end
            z_mse_hard(:,ser_count,algorithm_type)=z_mse_hard(:,ser_count,algorithm_type)+tmp_z_mse_hard_current_exper;%加总zk的误差平方，用于最后求mse
            
            
            %         %无判决的ck估计和误码率（最小二乘法）
            %         V_hosvd = bsxfun(@power,z_est_hosvd(index_z_hosvd).',(0:1:n-1)');
            %         c_est_hosvd=V_hosvd\y;% 求活跃用户ck的估计值，顺序对应上方zk频率从小到大排序
            %         c_est_hosvd_record(:,exper_count,ser_count)=c_est_hosvd;%记录ck估计值，按对应zk的频率从小到大排序
            %         c_est_quadrant_hosvd=zeros(ka,1);
            %         for i=1:ka
            %             tmp=angle(c_est_hosvd(i));
            %             if tmp<0
            %                 tmp=2*pi+tmp;
            %             end
            %             c_est_quadrant_hosvd(i)=floor(tmp*2/pi)+1;%计算ck估计值所在象限
            %         end
            %         tmp_c_ser_hosvd_current_exper=zeros(K1-1,1);%记录这次实验中的误码情况，此变量设置是因为parfor要求
            %         for i=1:ka
            %             if c_est_quadrant_hosvd(i)~=phi_count(a_nonzero(i))
            %                 tmp_c_ser_hosvd_current_exper(a_nonzero(i))=tmp_c_ser_hosvd_current_exper(a_nonzero(i))+1;
            %             end
            %         end%误码的活跃用户在tmp变量中0变1
            %         ser_hosvd(:,ser_count)=ser_hosvd(:,ser_count)+tmp_c_ser_hosvd_current_exper;%计入总的误码次数
            
            %硬判决的ck的估计和误码率（枚举法）
            
            V_hard = bsxfun(@power,z_angle_est_hard_with_sampleinterval.',(0:1:n-1)');%Y=V_hard*C, 要求C， V_hard是zk构成的vandermonde矩阵
            c_est_hard=zeros(ka,Q);
            for q=1:Q
                c_est=V_hard\y(:,q);
                for j=1:ka
                    angle_tmp=floor(angle(c_est(j))*2/pi)+3;
                    c_est_hard(j,q)=exp(1i*angle_or(angle_tmp));
                end
            end
            c_est_hard_record(:,:,exper_count,ser_count,algorithm_type)=c_est_hard;
            %             c_est_index=ones(1,Q);
            %             for channel=1:Q
            %                 c_est_dif=norm(y(:,channel)-V_hard*starset_value(:,c_est_index(channel)));
            %                 for index=2:(4^ka)
            %                     dif_tmp=norm(y(:,channel)-V_hard*starset_value(:,index));
            %                     if dif_tmp<c_est_dif
            %                         c_est_index(channel)=index;
            %                         c_est_dif=dif_tmp;
            %                     end
            %                 end
            %             end
            %         c_est_hard_hosvd=V_hard_hosvd\y;% 求活跃用户ck的估计值，顺序对应上方zk频率从小到大排序
            %             c_est_hard_record(:,:,exper_count,ser_count,algorithm_type)=starset_value(:,c_est_index);%记录ck估计值，按对应zk的频率从小到大排序
            %         c_est_hard_quadrant_hosvd=zeros(ka,1);
            %         for i=1:ka
            %             tmp=angle(c_est_hard_hosvd(i));
            %             if tmp<0
            %                 tmp=2*pi+tmp;
            %             end
            %             c_est_hard_quadrant_hosvd(i)=floor(tmp*2/pi)+1;%计算ck估计值所在象限
            %         end
            check=(z_est_hard_record(:,exper_count,ser_count,algorithm_type)==active_user_record(:,exper_count,ser_count));
            tmp_c_ser1_hard_current_exper=zeros(K1-1,Q);%记录这次实验中的误码情况，此变量设置是因为parfor要求
            tmp_c_ser2_hard_current_exper=zeros(K1-1,Q);%记录这次实验中的误码情况，此变量设置是因为parfor要求
            for i=1:ka
                for q=1:Q
                    if exp(1i*angle_or(phi_count(a_nonzero(i),q)))~=c_est_hard(i,q)
                        tmp_c_ser1_hard_current_exper(a_nonzero(i),q)=tmp_c_ser1_hard_current_exper(a_nonzero(i),q)+1;
                        tmp_c_ser2_hard_current_exper(a_nonzero(i),q)=tmp_c_ser2_hard_current_exper(a_nonzero(i),q)+1;
                    else
                        if check
                        else
                            tmp_c_ser2_hard_current_exper(a_nonzero(i),q)=tmp_c_ser2_hard_current_exper(a_nonzero(i),q)+1;
                        end
                    end
                end
            end %有错误就误码次数+1
            ser1_hard(:,:,ser_count,algorithm_type)=ser1_hard(:,:,ser_count,algorithm_type)+tmp_c_ser1_hard_current_exper;%计入总的误码次数
            ser2_hard(:,:,ser_count,algorithm_type)=ser2_hard(:,:,ser_count,algorithm_type)+tmp_c_ser2_hard_current_exper;%计入总的误码次数
            %计算平均误符号率ser_hard_matrix(1,ser_count)
            
        end
    end
end
%% 作图，保存数据

% disp('ck的估计值')
% abs(cc_nonzero).'
% disp('ck的误差')
% (abs(cc_nonzero)-sorta(1:rk)).'
% disp('zk的估计值的幅角按照对应的估计幅度从大到小排序');
% argz_nonzero.'
% % disp('真实的zk的幅角从小到大排序');
% % angle(w(a_nonzero)).'
% disp('估计的信号值与真实的无噪声信号值的误差')
% norm(V_nonzero*cc_nonzero-y)
% disp('hankel矩阵秩')
% rank(H)

% Z_MSE_HARD_MATRIX=log10(sum(z_mse_hard(:,:,1))/M);
Z_MSE_HARD_HOSVD=log10(sum(z_mse_hard(:,:,1))/M);
Z_MSE_HARD_HUBER_HOSVD=log10(sum(z_mse_hard(:,:,2))/M);
Z_MSE_HARD_HOOI=log10(sum(z_mse_hard(:,:,3))/M);
Z_MSE_HARD_HUBER_HOOI=log10(sum(z_mse_hard(:,:,4))/M);


% C_SER_HARD_MATRIX=log10(sum(ser_hard(:,:,1))/ka/M);
C_SER1_HARD_HOSVD=log10(sum(sum(ser1_hard(:,:,:,1)))/ka/M/Q);
C_SER1_HARD_HUBER_HOSVD=log10(sum(sum(ser1_hard(:,:,:,2)))/ka/M/Q);
C_SER1_HARD_HOOI=log10(sum(sum(ser1_hard(:,:,:,3)))/ka/M/Q);
C_SER1_HARD_HUBER_HOOI=log10(sum(sum(ser1_hard(:,:,:,4)))/ka/M/Q);

% C_SER_HARD_MATRIX=log10(sum(ser_hard(:,:,1))/ka/M);
C_SER2_HARD_HOSVD=log10(sum(sum(ser2_hard(:,:,:,1)))/ka/M/Q);
C_SER2_HARD_HUBER_HOSVD=log10(sum(sum(ser2_hard(:,:,:,2)))/ka/M/Q);
C_SER2_HARD_HOOI=log10(sum(sum(ser2_hard(:,:,:,3)))/ka/M/Q);
C_SER2_HARD_HUBER_HOOI=log10(sum(sum(ser2_hard(:,:,:,4)))/ka/M/Q);

figure(1)
% plot(snr,Z_MSE_HARD_MATRIX,'--o')
% hold on
plot(snr,Z_MSE_HARD_HOSVD,'g:d')
hold on
plot(snr,Z_MSE_HARD_HUBER_HOSVD,'r-.*')
hold on
plot(snr,Z_MSE_HARD_HOOI,'b:+')
hold on
plot(snr,Z_MSE_HARD_HUBER_HOOI,'c--x')
xlim([snr(1) snr(end)])

legend('HOSVD算法','L1-TOTD算法','HOOI算法','L1-TOOI算法')

xlabel('SNR','FontName','Times New Roman','FontSize',15);
ylabel('log10(zk.MSE)','FontName','Times New Roman','FontSize',15);
title(['各算法的信噪比-log10(zk.MSE)曲线， ka=' num2str(ka)],'FontName','宋体','FontSize',18)

figure(2)
% plot(snr,C_SER_HARD_MATRIX,'--o')
% hold on
plot(snr,reshape(C_SER1_HARD_HOSVD,1,[]),'g:d')
hold on
plot(snr,reshape(C_SER1_HARD_HUBER_HOSVD,1,[]),'r-.*')
hold on
plot(snr,reshape(C_SER1_HARD_HOOI,1,[]),'b:+')
hold on
plot(snr,reshape(C_SER1_HARD_HUBER_HOOI,1,[]),'c--x')
xlim([snr(1) snr(end)])
legend('HOSVD算法','L1-TOTD算法','HOOI算法','L1-TOOI算法')

xlabel('SNR','FontName','Times New Roman','FontSize',15);
ylabel('log10(ck.SER)','FontName','Times New Roman','FontSize',15);
title(['各算法的信噪比-log10(ck.SER)曲线， ka=' num2str(ka)],'FontName','宋体','FontSize',18)

figure(3)
% plot(snr,C_SER_HARD_MATRIX,'--o')
% hold on
plot(snr,reshape(C_SER2_HARD_HOSVD,1,[]),'g:d')
hold on
plot(snr,reshape(C_SER2_HARD_HUBER_HOSVD,1,[]),'r-.*')
hold on
plot(snr,reshape(C_SER2_HARD_HOOI,1,[]),'b:+')
hold on
plot(snr,reshape(C_SER2_HARD_HUBER_HOOI,1,[]),'c--x')
xlim([snr(1) snr(end)])
legend('HOSVD算法','L1-TOTD算法','HOOI算法','L1-TOOI算法')

xlabel('SNR','FontName','Times New Roman','FontSize',15);
ylabel('log10(ck.SER)','FontName','Times New Roman','FontSize',15);
title(['各算法的信噪比-log10(ck.SER)曲线， ka=' num2str(ka)],'FontName','宋体','FontSize',18)

% 已有总体平均均方误差ZK_MSE图像, 总体误码率CK_SER图像，但二者算法对比有差异
% 一个zk如果估计偏离严重，也会对ZK_MSE产生影响，考虑从zk的估计错误个数出发，即ZK_SER
% 先算一个有错误就计1的，表示实验出错次数；
wrong_time=zeros(snr_totaltype,algorithm_totalnumber);
wrong_distribution=zeros(ka,snr_totaltype,algorithm_totalnumber);
for algorithm_type=1:algorithm_totalnumber
    for snr_type=1:snr_totaltype
        for exper_counting=1:M
            tmpwrong=length(find(active_user_record(:,exper_counting,snr_type)-z_est_hard_record(:,exper_counting,snr_type,algorithm_type)~=0));
            %matlab的if array 只要array有0就判定为false
            if tmpwrong~=0
                wrong_time(snr_type,algorithm_type)=wrong_time(snr_type,algorithm_type)+1;
                wrong_distribution(tmpwrong,snr_type,algorithm_type)=wrong_distribution(tmpwrong,snr_type,algorithm_type)+1;
            end
        end
    end
end
% 再算一个错误详细个数的分布情况，包括0-ka个错误，记录于wrong_distribution
figure(4)
% plot(snr,wrong_time(:,1),'--o')
% hold on
plot(snr,wrong_time(:,1),'g:d')
hold on
plot(snr,wrong_time(:,2),'r-.*')
hold on
plot(snr,wrong_time(:,3),'b:+')
hold on
plot(snr,wrong_time(:,4),'c--x')
legend('HOSVD算法','L1-TOTD算法','HOOI算法','L1-TOOI算法')
xlabel('SNR','FontName','Times New Roman','FontSize',15);
ylabel('zk估计有误的实验次数','FontName','宋体','FontSize',15);
title(['SNR-zk估计有误的实验次数曲线,ka=' num2str(ka) ',总次数=' num2str(M)],'FontName','宋体','FontSize',18)

figure(5)
% plot(snr,wrong_time(:,1),'--o')
% hold on
plot(snr,sum(wrong_distribution(:,:,1))/ka/M,'g:d')
hold on
plot(snr,sum(wrong_distribution(:,:,2))/ka/M,'r-.*')
hold on
plot(snr,sum(wrong_distribution(:,:,3))/ka/M,'b:+')
hold on
plot(snr,sum(wrong_distribution(:,:,4))/ka/M,'c--x')
legend('HOSVD算法','L1-TOTD算法','HOOI算法','L1-TOOI算法')
xlabel('SNR','FontName','Times New Roman','FontSize',15);
ylabel('zk估计有误的实验次数','FontName','宋体','FontSize',15);
title(['zk恢复错误的次数曲线,ka=' num2str(ka) ',总次数=' num2str(M)],'FontName','宋体','FontSize',18)
% 服务器上的存储路径

delete(MyPar);
saveas(figure(1),['/home/ztl/data' num2str(datafile_count) '/ZK_MSE'],'fig');
saveas(figure(2),['/home/ztl/data' num2str(datafile_count) '/CK_SER_no_zk_confirm'],'fig');
saveas(figure(3),['/home/ztl/data' num2str(datafile_count) '/CK_SER_zk_confirm'],'fig');
saveas(figure(4),['/home/ztl/data' num2str(datafile_count) '/ZK_SER'],'fig');
saveas(figure(5),['/home/ztl/data' num2str(datafile_count) '/ZK_BER'],'fig');
save(['/home/ztl/data' num2str(datafile_count) '/ka-partial-users'],'-v7.3');

%本机存储方式
% saveas(figure(1),'ZK_MSE','fig');
% saveas(figure(2),'CK_SER_no_zk_confirm','fig');
% saveas(figure(3),'CK_SER_zk_confirm','fig');
% saveas(figure(4),'ZK_SER','fig');
% saveas(figure(5),'ZK_BER','fig');
% save('ka-partial-users','-v7.3');

% plot(1:1:50,(z_mse_matrix(:,8).')./exper_times(:,8).')
% xlabel('频率序号k=1~50，对应频率exp(j2\pi*(k-1)*\beta/K)','FontName','宋体','FontSize',15);
% ylabel('各频率的均方误差','FontName','宋体','FontSize',15);
% title(['当10log10(Eb/N0)=37时，频道在M=50000次实验后的zk的均方误差'],'FontName','宋体','FontSize',18)
