function [p, P_q, P_r] = fromFrame2D(q, r)

% p = R*q + t
% p is the co-ordinate in frame1
% q is the co-ordinate in frame2
% t is the origin of the frame2 in frame1

% inputs: q: 2D-position in frame2
%         r: 2D-pose of the frame1 r = [x, y, phi]

% outputs: p  : transformed coordinates in from frame2 to frame1
%          P_q: Jacobian of p wrt q
%          P_r: Jacobian of p wrt r (= [x y phi])

t = [r(1), r(2)]';
phi = atan2(sin(r(3)), cos(r(3)));


R = [cos(phi) -sin(phi);
     sin(phi)  cos(phi)];
 
p = R * q + repmat(t, 1, size(q,2)); 

if nargout > 1
 P_q = R;
 qx = q(1);
 qy = q(2);

 P_r = [1 0 -sin(phi) * qx - cos(phi) * qy;
        0 1  cos(phi) * qx - sin(phi) * qy];
end
end