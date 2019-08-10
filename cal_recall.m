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

recall50 = [];
recall100 = [];
recall150 = [];
recall200 = [];
recall250 = [];
recall300 = [];

for i = 1:100

    x = 50:50:M;
    S = load(['model\matrix\bdcmf_',num2str(i-1)]);
    m_U = S.m_U;
    m_V = S.m_V;
    [recall_bdcmf] = evaluateRecall(test_users, m_U, m_V, M);
    recall50(end+1) = recall_bdcmf(50);
    recall100(end+1) = recall_bdcmf(100);
    recall150(end+1) = recall_bdcmf(150);
    recall200(end+1) = recall_bdcmf(200);
    recall250(end+1) = recall_bdcmf(250);
    recall300(end+1) = recall_bdcmf(300);
end

save('model/recall.mat','recall50','recall100','recall150','recall200','recall250','recall300');

