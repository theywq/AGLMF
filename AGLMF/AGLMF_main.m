function [res] = AGLMF_main(fea, numClust, knn0, metric, gt,r2Temp)
%%
v = length(fea);   % view number 
k = numClust;  % class number
Fs = cell(1,v); 
Ss = cell(v, 1); 
n = length(gt);

for i = 1 :v
    for  j = 1:n
         X{i}(j,:) = ( fea{i}(j,:) - mean( fea{i}(j,:) ) ) / std( fea{i}(j,:) ) ;
    end
end
%%
for t=1:v
    A0 = constructgraph(X{t}',10);
    A0 = A0-diag(diag(A0));
    A10 = (A0+A0')/2;
    D10 = diag(1./sqrt(sum(A10, 2)));
    St = D10*A10*D10;
    [Ft,~,~] = svds(St,k);
    Ss{t} = St;
    Fs{t} = Ft;
end
opts.k = numClust;
[label, ~] = AGLMF(Ss,Fs,opts,r2Temp);
res = zeros(2, 8);
res(1,:) = Clustering8Measure(gt, label);
end