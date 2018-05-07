% EKF SLAM
close all;
clear;
clc;
% I. Initialize

% System noise
q = [0.09;0.009];
Q = diag(q.^2);
% Measurement noise
m = [0.15; 1*pi/180];
M = diag(m.^2);
% randn('seed',1);

%
%   1. Simulator
%       R: robot pose u: control
R = [0;-2.5;0];
u = [0.1;0.05];
xmin = -10;
xmax = 10;
ymin = -10;
ymax = 10;
W = cloister(xmin, xmax, ymin, ymax, 9);
y = zeros(2,size(W,2));

Rshape0 = 0.2*[2 -1 -1 2;
               0  1 -1 0];
Rshape = fromFrame2D(Rshape0, R);
% 2. Estimator

x = zeros(numel(R) + numel(W), 1);
P = zeros(numel(x), numel(x));
% P(1,1) = 1e4;
% P(2,2) = 1e4;
% P(3,3) = 1e2;
mapspace = 1:numel(x);
landmarks = zeros(size(W));

r = 1:numel(R);
mapspace(r) = 0; % blocking space for robot
x(r) = R;
P(r, r) = 0;

% Promt to ask user if we want to use the measurement model or not.
disp('user input is case-sensitive: True!=true');
useLandmarks = input('Do you want to map the landmarks?[Enter true for Yes and false for No:]');

 
% 3. Graphics

mapFig = figure(1);
cla;
axis([xmin-3 xmax+3 ymin-3 ymax+3]);
axis square;
grid;
xlabel('X');
ylabel('Y');
title('X-Y plane, Red: Truth, Blue: Estimation');
WG = line('parent', gca,...
     'Linestyle', 'none',...
     'marker', 'o',...
     'color', 'r',...
     'xdata', W(1,:),...
     'ydata', W(2,:));
 
RG = line('parent', gca,...
     'marker', '.',...
     'color', 'r',...
     'xdata', R(1,:),...
     'ydata', R(2,:));
 
RGshape = line('parent', gca,...
     'Linestyle', '-',...
     'marker', '.',...
     'color', 'r',...
     'xdata', Rshape(1,:),...
     'ydata', Rshape(2,:));
 
rGshape = line('parent', gca,...
     'Linestyle', '-',...
     'marker', '.',...
     'color', 'b',...
     'xdata', Rshape(1,:),...
     'ydata', Rshape(2,:));
 
rG = line('parent', gca,...
     'Linestyle', 'none',...
     'marker', '+',...
     'color', 'b',...
     'xdata', x(r(1)),...
     'ydata', x(r(2)));
 
lG = line('parent', gca,...
     'Linestyle', 'none',...
     'marker', '+',...
     'color', 'b',...
     'xdata', [],...
     'ydata', []);
 
 eG = zeros(1, size(W,2));
 for i = 1:numel(eG)
   eG(i) = line('parent', gca,...
     'color', 'g',...
     'xdata', [],...
     'ydata', []);    
 end
 
reG = line('parent', gca,...
     'color', 'b',...
     'xdata', [],...
     'ydata', []);

rangeCirc = line('parent', gca,...
     'Linestyle', '-.',...
     'color', 'r',...
     'xdata', [],...
     'ydata', []);
 
 
 % Sensor Params
 range = 10;
 tht_range = pi/3;

 % Temporal Loop
 lid_mapped = [];

 X_estimated = [];
 X_true = [];
 P_ = [];
 end_time = 1000;
 for t = 1 : end_time
     % 1. Simulator
     n = q.*randn(2, 1);
     R = motion_model(R, u, n);
     lid_observed = [];
     X_true = [X_true; R'];
     %Observe
     for lid = 1:size(W, 2)
         distance1 = norm(W(:,lid) - R(1:2));
         theta1 = atan2(W(2,lid) - R(2), W(1,lid) - R(1)) - atan2(sin(R(3)), cos(R(3)));
         if distance1 < range && theta1 >= -tht_range && theta1 <= tht_range
         v = m.*randn(2, 1);
         y(:,lid) = observe(W(:,lid), R) + v;
         lid_observed = [lid_observed lid];
         end
     end
     
     % 2. Filter
     % a. Prediction
     mapids = find(mapspace(numel(r)+1:end)==0) + numel(r);
     [x(r), R_r, R_n] = motion_model(x(r), u, zeros(2,1));
     P_rr = P(r,r);
     P(r,mapids) = R_r * P(r, mapids);
     P(mapids,r) = P(r,mapids)';
     P(r,r) = R_r * P_rr * R_r' + R_n * Q * R_n';
     if (useLandmarks)  

     
     % Discover New Landmarks
     if(~isempty(lid_observed))
        lids = lid_observed(randi(numel(lid_observed)));
        if ismember(lids, lid_mapped)
          lids = [];
        end
     else
         lids = [];
     end

     if(~isempty(lids))
         s = find(mapspace, 2);
         if(~isempty(s))
          mapspace(s) = 0;
          landmarks(:,lids) = s';
          %mapid = [mapid s];
          lid_mapped = [lid_mapped lids];
          % measurement
          yi = y(:,lids);
          [Li, L_y, L_r] = invobserve(yi, x(r));
          x(landmarks(:,lids)) = Li;
          P(s,:) = L_r * P(r,:);
          P(:,s) = P(s,:)';
          P(s,s) = L_r * P(r, r) * L_r' + L_y * M * L_y';
         end
     end

     % b. Correction
     % Observe Known Landmarks 
     % Data Association is done here
     for lid = lid_mapped
      distance2 = norm(x(landmarks(:,lid)) - x(r(1:2)));
      theta2 = atan2(x(landmarks(2,lid)) - x(r(2)), x(landmarks(1,lid)) - x(r(1))) - atan2(sin(x(r(3))), cos(x(r(3))));
      if distance2 < range && ismember(lid, lid_observed) && theta2 >= -tht_range && theta2 <= tht_range
      
      % Expectation
      [e, E_l, E_r] = observe(x(landmarks(:,lid)), x(r));
      E_rl = [E_r, E_l];
      rl = [r landmarks(:,lid)'];
      E = E_rl * P(rl, rl) * E_rl';
      
      % Measurement
      yi = y(:,lid);
      
      % Innovation
      z = yi - e;
      z(2) = atan2(sin(z(2)), cos(z(2)));
      Z = E + M;
   
      % Kalman Filter
      %rm = [r mapids];
      K = P(:,rl) * E_rl' * Z^(-1);
      
      % update
      x = x + K*z;
      P = P - K * Z * K';
      end
     end

     %3. Graphics
     

     lx = x(landmarks(1,lid_mapped));
     ly = x(landmarks(2,lid_mapped));
     set(lG, 'xdata', lx, 'ydata', ly);
     
     for lid = lid_mapped
         le = x(landmarks(:,lid));
         LE = P(landmarks(:,lid), landmarks(:,lid));
         [X, Y] = cov2elli(le, LE, 3, 16);
         set(eG(lid), 'xdata', X, 'ydata', Y);
     end
     end
     X_estimated = [X_estimated; x(1:3)'];
     P_ = [P_; [P(1,1), P(2,2), P(3,3)]];
     set(RG, 'xdata', R(1), 'ydata', R(2));
     set(rG, 'xdata', x(r(1)), 'ydata', x(r(2)));
     range_ = fromFrame2D([0, range*cos(-tht_range:0.01:tht_range), 0;0, range*sin(-tht_range:0.01:tht_range), 0], R);
     set(rangeCirc, 'xdata', range_(1,:), 'ydata', range_(2,:));
     if t > 1
         re = x(r(1:2));
         RE = P(r(1:2), r(1:2));
         [X, Y] = cov2elli(re, RE, 3, 16);
         set(reG, 'xdata', X, 'ydata', Y);
     end
     Rshape = fromFrame2D(Rshape0, R);
     set(RGshape, 'xdata', Rshape(1,:), 'ydata', Rshape(2,:));
     Rshape = fromFrame2D(Rshape0, x(r));
     set(rGshape, 'xdata', Rshape(1,:), 'ydata', Rshape(2,:));
     drawnow;
     
 end
 %%
 t = 1:end_time;
 figure(2)
 subplot(3,1,1)
 plot(t, X_true(:,1),'-r',...
    'LineWidth',3);
 hold on;
 plot(t, X_estimated(:,1),'-g',...
    'LineWidth',2)
 hold on;
 plot(t, X_true(:,1)-X_estimated(:,1),'-b',...
    'LineWidth',2);
 title('X');
  xlabel('time->(in units)');
 grid;
 legend('x_{true}','x_{estimated}','e_x')
 subplot(3,1,2)
 plot(t, X_true(:,2),'-r',...
    'LineWidth',3);
 hold on;
 plot(t, X_estimated(:,2),'-g',...
    'LineWidth',2)
 hold on;
 plot(t, X_true(:,2)-X_estimated(:,2),'-b',...
    'LineWidth',2);
 title('Y');
 xlabel('time->(in units)');
 grid;
 legend('y_{true}','y_{estimated}','e_y')
 subplot(3,1,3)
 plot(t, X_true(:,3),'-r',...
    'LineWidth',3);
 hold on;
 plot(t, X_estimated(:,3),'-g',...
    'LineWidth',2)
 hold on;
 plot(t, X_true(:,3)-X_estimated(:,3),'-b',...
    'LineWidth',2);
 title('\phi');
 xlabel('time->(in units)');
 grid;
 legend('\phi_{true}','\phi_{estimated}','e_{\phi}')
 hold off;
 
 figure(3)
 plot(t, P_(:,1),'-r',...
    'LineWidth',3);
 hold on;
 plot(t, P_(:,2),'-g',...
    'LineWidth',3);
 hold on;
 plot(t, P_(:,3),'-b',...
    'LineWidth',3);
 hold off;
 grid;
 legend('\sigma_{xx}','\sigma_{yy}','\sigma_{\phi \phi}');
 xlabel('time->(in units)');
 ylabel('State Covariances');
 title('Robot State Covariances vs Time');
 
 mean_avg_error_x = sum(abs(X_true(:,1)-X_estimated(:,1)))/length(t)
 mean_avg_error_y = sum(abs(X_true(:,2)-X_estimated(:,2)))/length(t)
 mean_avg_error_z = sum(abs(X_true(:,3)-X_estimated(:,3)))/length(t)