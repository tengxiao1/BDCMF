M = 300;
m_num_users = 1892;
m_num_items = 17632;

test_users = cell(m_num_users,1);
fid=fopen('data\lastfm\test_users_45.dat','r'); % user test file
for i=1:m_num_users
    tline = fgetl(fid);
    if ~ischar(tline), break, end
    liked = str2num(tline);
    liked(2:end) = liked(2:end)+1;
    test_users{i} = liked;
end
fclose(fid);

NDCG10 = [];
NDCG20 = [];
NDCG30 = [];
NDCG40 = [];
NDCG50 = [];

for i = 1:100
    disp(i);
    S = load(['model\matrix\bdcmf_',num2str(i-1)]);
    m_U = S.m_U;
    m_V = S.m_V;
    [NDCG_bdcmf] = evaluateNDCG(test_users, m_U, m_V, M);
    
    NDCG10(end+1) = NDCG_bdcmf(10);
    NDCG20(end+1) = NDCG_bdcmf(20);
    NDCG30(end+1) = NDCG_bdcmf(30);
    NDCG40(end+1) = NDCG_bdcmf(40);
    NDCG50(end+1) = NDCG_bdcmf(50);
end

save('model/NDCG.mat','NDCG10','NDCG20','NDCG30','NDCG40','NDCG50');
