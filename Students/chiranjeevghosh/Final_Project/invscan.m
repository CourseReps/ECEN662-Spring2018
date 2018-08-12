function [p, P_y] = invscan(y)

% input : y = [range, bearing]'
% output: p   = [px, py] landmark location (in robot frame)
%         P_y = jacobian of p wrt y

d = y(1);
tht = atan2(sin(y(2)), cos(y(2)));

p = [ d * cos(tht);
      d * sin(tht)];

P_y = [ cos(tht), -d*sin(tht);
        sin(tht),  d*cos(tht)];
end