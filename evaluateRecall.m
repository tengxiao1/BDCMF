function [recall] = evaluatePaper(test_users, m_U, m_V, M)
m_num_users = size(m_U,1);
m_num_items = size(m_V,1);

batch_size = 100;
n = ceil(1.0*m_num_users/batch_size);
num_hit = zeros(m_num_users,M);
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
   num_total(ind, :) = sum(re, 2);
end

k=1;
A=repmat(num_total, 1, M);
num_nozero=0;
for j=1:m_num_users
 if A(j,1)==0
     continue;
 end
 num_nozero=num_nozero+1;
end
num_hit2=zeros(num_nozero,M);
A2=zeros(num_nozero,M);
for j=1:m_num_users
 if A(j,1)==0
     continue;
 end
 num_hit2(k,:)=num_hit(j,:);
 A2(k,:)=A(j,:);
 k=k+1;
end

recall = mean(cumsum(num_hit2, 2)./A2, 1);

