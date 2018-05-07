C = sym('c', [3 3]);
P = sym('p', [3 1]);
syms alpha beta rho;

H = C*[alpha; beta; 1]+rho*P;

h1 = H(1);
h2 = H(2);
h3 = H(3);

F = 1/h3*[h1; h2];

J=jacobian(F,[alpha, beta, rho])