function [y_idx, Tobj] = AGLMF(Ss,Fs,opts,r2Temp)

k = opts.k; % clusters

v = length(Ss);

[n, ~] = size(Ss{1});

for idx = 1:v
    Rs{idx} = Fs{idx}; 
    p(idx) = 1;
    M{idx} = Ss{idx};
    A0{idx} = M{idx}-diag(diag(M{idx}));
end

for i=1:n
   u{i} = ones(1,length(1:n)); 
end

    % Update H（Y）
    T = zeros(n,k);
    for idx = 1:v
        Rt = Rs{idx};
        St = M{idx};
        temp = St*Rt;
        pt = p(idx)/sum(p);
        T = T + pt*temp;
    end
    Y = max(T/v,0);
    
    % Update F（R）
    for idx = 1:v
        St = M{idx};
        temp = St'*Y;
        [Ur, ~, Vr] = svds(temp,k);
        Rs{idx} = Ur*Vr';   
    end   
    
    %Update M
    for idx=1:v
        temp=Rs{idx}*Y';
        M{idx}=zeros(n);
        for i=1:n
            %u{i} = ones(1,n);
            ai = A0{idx}(i,:);
            di = temp(i,:);
            si = M{idx}(i,:);
            e=eye(n);
            ei=e(i,:);
            lambda = r2Temp;
            for ii = 1:1
                ad = u{i}.*ai+lambda*di;
                si = EProjSimplexdiag(ad, u{i}+(lambda/2).*ei);
                u{i} = 1./(2*sqrt((si-ai).^2+eps));     
            end
            M{idx}(i,:) = si;
        end
    end

max_iter = 60;

for iter = 1:max_iter
    
    % Update H（Y）
    T = zeros(n,k);
    for idx = 1:v
        Rt = Rs{idx};
        St = M{idx};
        temp = St*Rt;
        pt = p(idx)/sum(p);
        T = T + pt*temp;
    end
    Y = max(T/v,0);
    
    % Update F（R）
    for idx = 1:v
        St = M{idx};
        temp = St'*Y;
        [Ur, ~, Vr] = svds(temp,k);
        Rs{idx} = Ur*Vr';   
    end   
    
    %Update M
    for idx=1:v
        temp=Rs{idx}*Y';
        M{idx}=zeros(n);
        for i=1:n
            %u{i} = ones(1,n);
            ai = A0{idx}(i,:);
            di = temp(i,:);
            si = M{idx}(i,:);
            e=eye(n);
            ei=e(i,:);
            lambda = r2Temp;
            for ii = 1:1
                ad = u{i}.*ai+lambda*di;
                si = EProjSimplexdiag(ad, u{i}+(lambda/2).*ei);
                u{i} = 1./(2*sqrt((si-ai).^2+eps));    
            end
            M{idx}(i,:) = si;
        end
    end
    
  % objective function
    obj = 0;
    for idx = 1:v
        Rt = Rs{idx};
        St = M{idx};
        temp = St - Y*Rt';
        p(idx) = 1/norm(temp,'fro');
        obj = obj + norm(temp,'fro')^2+norm(Ss{idx}-St,1);
    end
    Tobj(iter) = obj;
    
    % convergence checking
    if iter>1
        temp_obj = Tobj(iter -1);
    else
        temp_obj = 0;
    end
    if abs(obj - temp_obj)/temp_obj <1e-8
        break;
    end
end
[~ ,y_idx] = max(Y,[],2);