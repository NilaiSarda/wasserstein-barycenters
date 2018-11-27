getd = @(p)path(p,path); % scilab users must *not* execute this
getd('toolbox_signal/');
getd('toolbox_general/');
mynorm = @(a)norm(a(:));
sum3 = @(a)sum(a(:));
n = 100;
p = 20;
[Y,X] = meshgrid(linspace(0,1,n), linspace(0,1,n));
gaussian = @(a,b,sigma)exp( -((X-a).^2+(Y-b).^2)/(2*sigma^2) );
normalize = @(u)u/sum(u(:));
sigma = .1;
gg = 0.1;
rho = .05; % minimum density value
f0 = normalize( rho + gaussian(.2,.3,sigma) );
f1 = normalize( rho + gaussian(.6,.5,sigma*.5) + .5*gaussian(.7,.4,sigma*.7) );
dx = @(u)u([2:end 1],:,:)-u;
dy = @(u)u(:,[2:end 1],:)-u;
dxS = @(u)-u+u([end 1:end-1],:,:);
dyS = @(u)-u+u(:,[end 1:end-1],:);
grad = @(f)cat(4, dx(f), dy(f));
div  = @(u)-dxS(u(:,:,:,1)) - dyS(u(:,:,:,2));
dt  = @(f)cat(3, f(:,:,2:end)-f(:,:,1:end-1), zeros(size(f,1),size(f,2)) );
dtS = @(u)cat(3, -u(:,:,1), u(:,:,1:end-2)-u(:,:,2:end-1), u(:,:,end-1));
A = @(w)cat( 3, div(w(:,:,:,1:2))+dt(w(:,:,:,3))-gg*div(grad(w(:,:,:,1:2))), w(:,:,1,3), w(:,:,end,3) );
U = @(r0,r1)cat(3, r0, zeros(n,n,p-2), r1);
AS = @(s)cat(4, -grad(s(:,:,1:p)), dtS(s(:,:,1:p)) - gg*div(grad(w(:,:,:,1:2))) + U(s(:,:,end-1),s(:,:,end)) );
r0 = cat(3, zeros(n,n,p), f0, f1);
J = @(w)sum3(  sum(w(:,:,:,1:2).^2,4) ./ w(:,:,:,3)   );
PolyCoef = @(m0,f0,lambda)[ones(length(f0),1), 4*lambda-f0, 4*lambda^2-4*f0, -lambda*sum(m0.^2,2) - 4*lambda^2*f0];
extract = @(A)A(:,1);
CubicReal = @(P)real( extract(poly_root(P')') );
Proxj0 = @(m0,f, lambda)cat(2, m0 ./ repmat( 1+2*lambda./f, [1 2]), f );
Proxj  = @(m0,f0,lambda)Proxj0( m0, CubicReal(PolyCoef(m0,f0,lambda)), lambda );
ProxJ = @(w,lambda)reshape( Proxj( ...
                   reshape(w(:,:,:,1:2), [n*n*p 2]), ...
                   reshape(w(:,:,:,3  ), [n*n*p 1]), lambda ), [n n p 3] );
opts.epsilon = 1e-9;
opts.niter_max = 150;
flat = @(x)x(:);
resh = @(x)reshape(x, [n n p+2]);
mycg = @(B,y)resh( perform_cg(@(r)flat(B(resh(r))),y(:),opts) );
pA = @(r)mycg(@(s)A(AS(s)),r);
ProxG = @(w,lambda)w + AS( pA(r0-A(w)) );
mu = 1;
gamma = 1;
rProxJ = @(w,tau)2*ProxJ(w,tau)-w;
rProxG = @(w,tau)2*ProxG(w,tau)-w;
niter = 200;
t = repmat( reshape(linspace(0,1,p), [1 1 p]), [n n 1]);
f = (1-t) .* repmat(f0, [1 1 p]) + t .* repmat(f1, [1 1 p]);
m = zeros(n,n,p,2);
w0 = cat(4, m,f);
energy = [];
constr = [];
tw = w0;
for i=1:niter
    tw_old = tw;
    w = ProxG(tw,gamma);
    rw = 2*w-tw_old;
    tw = (1-mu/2)*tw + mu/2*rProxJ( rw, gamma );
    % 
    energy(i) = J(w);
    constr(i) = mynorm( A(w)-r0 ) / mynorm(r0); 
end
clf;
h = plot(min(energy, energy(1)));
set(h, 'LineWidth', 2);
title('J(w)');
axis tight;
sel = round(linspace(1,p,6));
clf;
imageplot( mat2cell(w(:,:,sel,3), n, n, ones(6,1)) , '', 2,3);