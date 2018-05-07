function [ro, RO_r, RO_n] = motion_model(r, u, n)

% implements a simple motion model
% input:  r : current robot pose, r = [x, y, phi]'
%         u : control input,      u = [dx, dphi]'
%         n : gaussian noise input
% output: ro   : robot pose post application of control
%         RO_r : jacobian of robot pose wrt r
%         RO_n : jacobian of robot pose wrt n

dx =  u(1) + n(1);
dphi = u(2) + n(2);

phi = r(3);
phio = phi + dphi;
phio = atan2(sin(phio), cos(phio));

dp  = [dx; 0];

[to, TO_dp, TO_r] = fromFrame2D(dp, r);

PHIO_phi = 1;
PHIO_dphi = 1;

ro = [to; phio];

RO_r = [TO_r; 0 0 PHIO_phi];
RO_n = [TO_dp(:,1) zeros(2, 1); 0 PHIO_dphi];

end