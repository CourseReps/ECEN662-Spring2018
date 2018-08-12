function [q, Q_p, Q_r] = toFrame2D(p, r)

% q = R' * p - R' * t
% p is the co-ordinate in frame1
% q is the co-ordinate in frame2
% t is the origin of the frame2 in frame1

% inputs: p: 2D-position in frame1
%         r: 2D-pose of the frame1 r = [x, y, phi]

% outputs: q  : transformed coordinates in from frame1 to frame2
%          Q_p: Jacobian of q wrt p
%          Q_r: Jacobian of q wrt r (= [x y phi])

t = [r(1), r(2)]';
phi = atan2(sin(r(3)), cos(r(3)));
R = [cos(phi) -sin(phi);
     sin(phi)  cos(phi)];
 
q = R' * p - R'*repmat(t, 1, size(p,2)) ;
if nargout > 1
Q_p = R';

x = r(1); y = r(2);
px = p(1); py = p(2);

Q_r = [ -cos(phi), -sin(phi),   cos(phi)*(py - y) - sin(phi)*(px - x);
         sin(phi), -cos(phi), - cos(phi)*(px - x) - sin(phi)*(py - y)];
end
end
 