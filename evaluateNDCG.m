function [NDCG] = evaluateNDCG(test_users, m_U, m_V, M)
m_num_users = size(m_U,1);
m_num_items = size(m_V,1);

batch_size = 100;
n = ceil(1.0*m_num_users/batch_size);
num_hit = zeros(m_num_users,M);
ideal = zeros(m_num_users,m_num_items);
num_total = zeros(m_num_users,1);
for i=1:n
   ind = (i-1)*batch_size+1:min(i*batch_size, m_num_users);
   u_tmp = m_U(ind,:);
   score = u_tmp * m_V';
   if (i==1)
        %disp(size(score))
   end
   [~,I] = sort(score, 2, 'descend');
   
   bs = length(ind);
   gt = zeros(bs, m_num_items);
   for j=1:bs
       idx = (i-1)*batch_size + j;
       u = test_users{idx};
       gt(j, u(2:end)) = 1;
   end
   re = zeros(bs, m_num_items);
   for j=1:bs
       re(j,:) = gt(j, I(j,:));
   end
   num_hit(ind, :) = re(:, 1:M);
   ideal(ind, :) = re(:,:);
   num_total(ind, :) = sum(re, 2);
end

ideal = sort(ideal, 2, 'descend');
ideal = ideal(:,1:M);
k=1;
A=repmat(num_total, 1, M);
num_nozero=0;
for j=1:m_num_users
 if A(j,1)==0
     continue;
 end
 num_nozero=num_nozero+1;
end
ideal2 = zeros(num_nozero,M);
num_hit2 = zeros(num_nozero,M);
A2=zeros(num_nozero,M);
for j=1:m_num_users
 if A(j,1)==0
     continue;
 end
 num_hit2(k,:)=num_hit(j,:);
 ideal2(k,:) = ideal(j,:);
 A2(k,:)=A(j,:);
 k=k+1;
end

nums = ones(num_nozero,M);
cumnums = cumsum(nums,2);
DCG = cumsum((2.^num_hit2 - 1)./(log2(cumnums + 1)),2);
IDCG = cumsum((2.^ideal2 - 1)./(log2(cumnums + 1)),2);

NDCG = mean(DCG./IDCG,1);






