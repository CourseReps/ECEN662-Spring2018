function [y, Y_p] = scan(p)

% outputs range and bearing (in robot frame)
% landmark observation function
% input : p = [x y]'     - landmark position (in robot frame)
% output: y = [d theta]' - landmark range and bearing respectively 
%                          (in robot frame)
%         Y_p = Jacobian of y wrt p


px = p(1);
py = p(2);

d = sqrt(px^2 + py^2);
a = atan2(py, px);
% a = atan(py/ px); % use this only for symbolic Jacobian computation

y = [d;a];

if nargout > 1
    
    Y_p =[...
[     px/(px^2 + py^2)^(1/2), py/(px^2 + py^2)^(1/2)]
[ -py/(px^2*(py^2/px^2 + 1)), 1/(px*(py^2/px^2 + 1))]];

    
end
end