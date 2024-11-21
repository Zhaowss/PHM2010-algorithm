function output= rawdataDisplay(datastore, fs)

 %% 数据展示
 % =====================fx 数据展示 ===============
 fontsize=14;
 reset(datastore)
 % the following code display the fx vs time
 tstart=0;
 hold on;
figure(1)
i=1;
Downsampling=10000;
% Downsampling=10;
while hasdata(datastore)
    data=read(datastore);
    fx=data.fx{1};
    t=tstart+(1:length(fx))/length(fx); 
    plot(t(1:Downsampling:end), fx(1:Downsampling:end), 'color', 'k');
    tstart=t(end);
    i=i+1;
end
hold off
box on;

set(gca,'fontsize',fontsize,'fontweight','bold')
ylabel('x方向切削力/N');
xlabel( '时间/(切削次数)');

% ==================fy 数据展示=============
reset(datastore)
tstart = 0;
figure
hold on
i=1;
figure(2)
while hasdata(datastore)
    i=i+1;
    data = read(datastore);
    fy = data.fy{1};
    t = tstart + (1:length(fy))/length(fy);
    plot(t(1:Downsampling:end), fy(1:Downsampling:end), 'color', 'k')
    tstart = t(end);
end
hold off
box on;
set(gca,'fontsize',fontsize,'fontweight','bold')
xlabel('时间/(切削次数)');
ylabel('y方向切削力/N');

% ===============fz数据展示================
reset(datastore)
tstart = 0;
figure
hold on
i=1;
figure(3)
while hasdata(datastore)
    i=i+1;
    data = read(datastore);
    fz = data.fz{1};
    t = tstart + (1:length(fz))/length(fz);
    plot(t(1:Downsampling:end), fz(1:Downsampling:end), 'color', 'k')
    tstart = t(end);
end
hold off
box on;
set(gca,'fontsize',fontsize,'fontweight','bold')
xlabel('时间/(切削次数)')
ylabel('z方向切削力/N');

% =================vx数据展示============
reset(datastore)
tstart = 0;
figure
hold on
i=1;
figure(4)
while hasdata(datastore)
    i=i+1;
    data = read(datastore);
    vx = data.vx{1};
    t = tstart + (1:length(vx))/length(vx);
    plot(t(1:Downsampling:end), vx(1:Downsampling:end), 'color', 'k')
    tstart = t(end);
end
hold off
box on
set(gca,'fontsize',fontsize,'fontweight','bold')
xlabel('时间/(切削次数)')
ylabel('x方向加速度/g')

% ================vy数据展示===============
reset(datastore)
tstart = 0;
figure
hold on
i=1;
figure(5)
while hasdata(datastore)
    i=i+1;
    data = read(datastore);
    vy = data.vy{1};
    t = tstart + (1:length(vy))/length(vy);
    plot(t(1:Downsampling:end), vy(1:Downsampling:end), 'color', 'k')
    tstart = t(end);
end
hold off
box on
set(gca,'fontsize',fontsize,'fontweight','bold')
xlabel('时间/(切削次数)')
ylabel('y方向加速度/g');

% ===============vz数据展示===============
reset(datastore)
tstart = 0;
figure
hold on
i=1;
figure(6)
while hasdata(datastore)
    i=i+1;
    data = read(datastore);
    vz = data.vz{1};
    t = tstart + (1:length(vz))/length(vz);
    plot(t(1:Downsampling:end), vz(1:Downsampling:end), 'color', 'k')
    tstart = t(end);
end
hold off
box on
set(gca,'fontsize',fontsize,'fontweight','bold')
xlabel('时间/(切削次数)')
ylabel('z方向加速度/g');

% ==============ae数据展示==============
reset(datastore)
tstart = 0;
figure
hold on
i=1;
figure(7)
while hasdata(datastore)
    i=i+1;
    data = read(datastore);
    ae = data.ae{1};
    t = tstart + (1:length(ae))/length(ae);
    plot(t(1:Downsampling:end), ae(1:Downsampling:end), 'color', 'k')
    tstart = t(end);
end
hold off
box on
set(gca,'fontsize',fontsize,'fontweight','bold')
xlabel('时间/(切削次数)')
ylabel('声发射信号')
output=data;
end

