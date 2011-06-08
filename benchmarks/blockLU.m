%Blocked LU factorization

% block lu

SIZE = 4000;
BS = 200;
Nblock = SIZE / BS;
Nlocal = BS;
N      = Nlocal;

% matrix to factorize
A = zeros(SIZE,SIZE);

for row=1:Nblock
    for col=1:Nblock
        for n=0:N-1
            for m=0:N-1
                r = (row-1)*N + n;
                c = (col-1)*N + m;
                A((row-1)*BS+n+1,(col-1)*BS+m+1) = r*c / (Nblock*N*N*Nblock);
            end
        end
        if(row==col)
            for n=0:N-1
                r = (row-1)*N + n;
                c = (col-1)*N + m;
                A((row-1)*BS+n+1,(col-1)*BS+n+1) = A((row-1)*BS+n+1,(col-1)*BS+n+1) + 10;
            end
        end
    end
end

Aorig = A;

% multiplier matrix
L = zeros(SIZE,SIZE);

% extract U from the modified A
U = zeros(SIZE,SIZE);

% loop over columns
for k=0:Nblock-1
    kb = k*BS;
    % check to see if the pivot is zero
    diagA = A(kb+1:kb+BS,kb+1:kb+BS);
    [diagL,diagU] = lu(diagA);
    if(abs(det(diagL))<1e-10)
        disp('warning')
        break; % Gaussian Elimination breaks down
    end
    
    L(kb+1:kb+BS,kb+1:kb+BS) = diagL;
    U(kb+1:kb+BS,kb+1:kb+BS) = diagU;
    % compute multipliers
    for i=k+1:Nblock-1
        ib = i*BS;
        L(ib+1:ib+BS,kb+1:kb+BS) = A(ib+1:ib+BS,kb+1:kb+BS)/diagU;
        U(kb+1:kb+BS,ib+1:ib+BS) = diagL \ A(kb+1:kb+BS,ib+1:ib+BS);
    end
    
    % apply M to remaining submatrix
    for j=k+1:Nblock-1
        for i=k+1:Nblock-1
            ib = i*BS;
            jb = j*BS;
            A(ib+1:ib+BS,jb+1:jb+BS) = A(ib+1:ib+BS,jb+1:jb+BS) - L(ib+1:ib+BS,kb+1:kb+BS)*U(kb+1:kb+BS,jb+1:jb+BS);
        end
    end
end


% back solve L
y = zeros(Nlocal, Nblock);
b = zeros(Nlocal, Nblock);
for i=1:N*Nblock
  b(i) = (i-1)/(N*Nblock);
end

for i=1:Nblock
    ib = (i-1)*BS;
    y(:,i) = b(:,i);
    for j=1:i-1
        jb = (j-1)*BS;
        y(:,i) = y(:,i) - L(ib+1:ib+BS,jb+1:jb+BS) * y(:,j);
    end
    y(:,i) = L(ib+1:ib+BS,ib+1:ib+BS) \ y(:,i);
end


x = zeros(Nlocal, Nblock);
for i=Nblock:-1:1
    ib = (i-1)*BS;
    x(:,i) = y(:,i);
    for j=i+1:Nblock
        jb = (j-1)*BS;
        x(:,i) = x(:,i) - U(ib+1:ib+BS,jb+1:jb+BS) * x(:,j);
    end
    x(:,i) = U(ib+1:ib+BS,ib+1:ib+BS) \ x(:,i);  
end


% test answer
Ax = zeros(Nlocal, Nblock);

for i=1:Nblock
    ib = (i-1)*BS;
    for j=1:Nblock
        jb = (j-1)*BS;
        Ax(:,i) = Ax(:,i) + Aorig(ib+1:ib+BS,jb+1:jb+BS)*x(:,j);
    end
end

max(max(abs(Ax(:)-b(:)))) %Should be zero



% compute L*U
LU = zeros(SIZE,SIZE);

for i=1:Nblock
    for j=1:Nblock
        ib = (i-1)*BS;
        jb = (j-1)*BS;
        LU(ib+1:ib+BS,jb+1:jb+BS) = zeros(Nlocal,Nlocal);
        for k=1:Nblock
            kb = (k-1)*BS;
            if( (i>=k) & (j>=k)) 
                LU(ib+1:ib+BS,jb+1:jb+BS) = LU(ib+1:ib+BS,jb+1:jb+BS) + L(ib+1:ib+BS,kb+1:kb+BS)*U(kb+1:kb+BS,jb+1:jb+BS);
            end
        end
    end
end

error = zeros(Nblock);
for i=1:Nblock
    ib = (i-1)*BS;
    for j=1:Nblock
        jb = (j-1)*BS;
        error(i,j) = max(max(abs(LU(ib+1:ib+BS,jb+1:jb+BS)-Aorig(ib+1:ib+BS,jb+1:jb+BS))));
    end
end
error
