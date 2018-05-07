% EKF SLAM
clear;
clc;
% I. Initialize

% System noise
q = [0.01;0.005];
Q = diag(q.^2);
% Measurement noise
m = [.15; 1*pi/180];
M = diag(m.^2);
% randn('seed',1);

%
%   1. Simulator
%       R: robot pose u: control
R = [0;-2.5;0];
u = [0.1;0.05];
xmin = -6;
xmax = 6;
ymin = -6;
ymax = 6;
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

% 3. Graphics

mapFig = figure(1);
cla;
axis([xmin-3 xmax+3 ymin-3 ymax+3]);
axis square;
grid;

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
 
 useLandmarks = true;
 % II. Temporal Loop
 
 for t = 1 : 6000
     % 1. Simulator
     n = q.*randn(2, 1);
     R = motion_model(R, u, n);
     
     for lid = 1:size(W, 2)
         v = m.*randn(2, 1);
         y(:,lid) = observe(W(:,lid), R) + v;
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
     % b. Correction
     % i. Known Landmarks   
     lids = [];
     for i = 1:size(W,2)
         if (landmarks(1,i) ~= 0 || landmarks(2,i) ~= 0)
             lids = [lids, i];
         end
     end

     for lid = lids
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
      rm = [r mapids];
      K = P(rm, rl) * E_rl' * Z^(-1);
      
      % update
      x(rm,:) = x(rm,:) + K*z;
      P(rm,rm) = P(rm,rm) - K * Z * K';
     end
     
     % ii. New Landmarks
     lids = [];
     for i = 1:size(W,2)
         if (landmarks(1,i) == 0 && landmarks(2,i) == 0)
             lids = [lids, i];
         end
     end

     if(~isempty(lids))
         lid = lids(randi(numel(lids)));
         s = find(mapspace, 2);
         if(~isempty(s))
          mapspace(s) = 0;
          landmarks(:,lid) = s';
          
          % measurement
          yi = y(:,lid);
          [Li, L_y, L_r] = invobserve(yi, x(r));
          x(landmarks(:,lid)) = Li;
          xids = [r mapids];
          P(s,xids) = L_r * P(r,xids);
          P(xids,s) = P(s,xids)';
          P(s,s) = L_r * P(r, r) * L_r' + L_y * M * L_y';
         end
     end

     %3. Graphics
     lids = [];
     for i = 1:size(W,2)
         if (landmarks(1,i) ~= 0 || landmarks(2,i) ~= 0)
             lids = [lids, i];
         end
     end

     lx = x(landmarks(1,lids));
     ly = x(landmarks(2,lids));
     set(lG, 'xdata', lx, 'ydata', ly);
     
     for lid = lids
         le = x(landmarks(:,lid));
         LE = P(landmarks(:,lid), landmarks(:,lid));
         [X, Y] = cov2elli(le, LE, 3, 16);
         set(eG(lid), 'xdata', X, 'ydata', Y);
     end
     end
     set(RG, 'xdata', R(1), 'ydata', R(2));
     set(rG, 'xdata', x(r(1)), 'ydata', x(r(2)));
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
 
 