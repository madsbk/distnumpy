%Blocked LU factorization

% block lu

SIZE = 4;
BS = 4;
Nblock = SIZE / BS;
Nlocal = BS;
N      = Nlocal;

% matrix to factorize
A = zeros(SIZE,SIZE);

for row=1:BS:SIZE
    rbs = min(BS,SIZE - (row-1));%Current block size
    for col=1:BS:SIZE
        cbs = min(BS,SIZE - (col-1));
        for n=0:rbs-1
            for m=0:cbs-1
                r = (row-1) + n;
                c = (col-1) + m;
                A(row+n,col+m) = r*c / (Nblock*BS*BS*Nblock);
            end
        end
        if(row==col)
            for n=0:rbs-1
                A(row+n,col+n) = A(row+n,col+n) + 10;
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
for col=1:BS:SIZE
    bs = min(BS,SIZE - (col-1));%Current block size
    
    % check to see if the pivot is zero
    diagA = A(col:col+bs-1,col:col+bs-1);
    [diagL,diagU] = lu(diagA);
    if(abs(det(diagL))<1e-10)
        disp('warning')
        break; % Gaussian Elimination breaks down
    end

    L(col:col+bs-1,col:col+bs-1) = diagL;
    L(col:col+bs-1,col:col+bs-1)
    U(col:col+bs-1,col:col+bs-1) = diagU;
    
    % compute multipliers
    for i=col+bs:BS:SIZE
        tbs = min(BS,SIZE - (i-1));%Current block size
        L(i:i+tbs-1,col:col+bs-1) = A(i:i+tbs-1,col:col+bs-1) / diagU;
        U(col:col+bs-1,i:i+tbs-1) = diagL \ A(col:col+bs-1,i:i+tbs-1);
    end
    
    % apply M to remaining submatrix
    if(col+bs <= SIZE)
        A(col+bs:SIZE, col+bs:SIZE) = A(col+bs:SIZE, col+bs:SIZE) - L(col+bs:SIZE,col:col+bs-1) * U(col:col+bs-1,col+bs:SIZE);
    end
    
end
A = L + U - eye(size(A)) %Merge L and U into A.


% 
% 
% % back solve L
% y = zeros(Nlocal, Nblock);
% b = zeros(Nlocal, Nblock);
% for i=1:N*Nblock
%   b(i) = (i-1)/(N*Nblock);
% end
% 
% for i=1:Nblock
%     ib = (i-1)*BS;
%     y(:,i) = b(:,i);
%     for j=1:i-1
%         jb = (j-1)*BS;
%         y(:,i) = y(:,i) - L(ib+1:ib+BS,jb+1:jb+BS) * y(:,j);
%     end
%     y(:,i) = L(ib+1:ib+BS,ib+1:ib+BS) \ y(:,i);
% end
% 
% 
% x = zeros(Nlocal, Nblock);
% for i=Nblock:-1:1
%     ib = (i-1)*BS;
%     x(:,i) = y(:,i);
%     for j=i+1:Nblock
%         jb = (j-1)*BS;
%         x(:,i) = x(:,i) - U(ib+1:ib+BS,jb+1:jb+BS) * x(:,j);
%     end
%     x(:,i) = U(ib+1:ib+BS,ib+1:ib+BS) \ x(:,i);  
% end
% 
% 
% % test answer
% Ax = zeros(Nlocal, Nblock);
% 
% for i=1:Nblock
%     ib = (i-1)*BS;
%     for j=1:Nblock
%         jb = (j-1)*BS;
%         Ax(:,i) = Ax(:,i) + Aorig(ib+1:ib+BS,jb+1:jb+BS)*x(:,j);
%     end
% end
% 
% max(max(abs(Ax(:)-b(:)))) %Should be zero
% 
% 
% 
% % compute L*U
% LU = zeros(SIZE,SIZE);
% 
% for i=1:Nblock
%     for j=1:Nblock
%         ib = (i-1)*BS;
%         jb = (j-1)*BS;
%         LU(ib+1:ib+BS,jb+1:jb+BS) = zeros(Nlocal,Nlocal);
%         for k=1:Nblock
%             kb = (k-1)*BS;
%             if( (i>=k) & (j>=k)) 
%                 LU(ib+1:ib+BS,jb+1:jb+BS) = LU(ib+1:ib+BS,jb+1:jb+BS) + L(ib+1:ib+BS,kb+1:kb+BS)*U(kb+1:kb+BS,jb+1:jb+BS);
%             end
%         end
%     end
% end
% 
% error = zeros(Nblock);
% for i=1:Nblock
%     ib = (i-1)*BS;
%     for j=1:Nblock
%         jb = (j-1)*BS;
%         error(i,j) = max(max(abs(LU(ib+1:ib+BS,jb+1:jb+BS)-Aorig(ib+1:ib+BS,jb+1:jb+BS))));
%     end
% end
% error
