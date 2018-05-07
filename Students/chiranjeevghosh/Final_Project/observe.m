function [y, Y_p, Y_r] = observe(p, r)
% input : p - point in global frame
%         r - 2D-robot pose in world frame

% output : y   - range and bearing measurement (in robot frame)
%          Y_p - Jacobian of y with respect to p
%          Y_r - Jacobian of y with respect to r

[q, Q_p, Q_r] = toFrame2D(p, r);
[y, Y_q] = scan(q);

if nargout > 1
% using chain rule to find Jacobians
Y_p = Y_q * Q_p;
Y_r = Y_q * Q_r;
end
end