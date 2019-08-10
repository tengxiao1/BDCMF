import scipy.io
import numpy as np
pairs_train = []
pairs_test = []
pairs_social = []
num_train_per_user = 45
np.random.seed(123)
artists = {}

# get artist id
with open("artists.dat",'rb') as f:
    lines = f.readlines()
    artist_id = 0
    for line in lines[1:]:
        line = line.decode('utf-8')
        arr = line.strip().split('\t')
        artist = arr[0]
        artists[artist] = artist_id
        artist_id += 1
    print("num of artists: ",artist_id)

# get user id
users = {}
with open("user_artists.dat") as f:
    lines = f.readlines()
    user_id = 0
    for line in lines[1:]:
        arr = line.strip().split()
        user = arr[0]
        if(user not in users.keys()):
            users[user] = user_id
            user_id += 1
    print("num of users: ",user_id)

    cur_user = 0
    cur_item = 0
    user_item_list = []
    user_item = []

    for line in lines[1:]:
        arr = line.strip().split()
        user = arr[0]
        item = arr[1]
        if(user != cur_user):
            user_item_list.append(user_item)
            user_item = []
            cur_user = user
            cur_item = item
            user_item.append(artists[item])
        elif(artists[item] not in user_item):
            user_item.append(artists[item])
            
    user_item_list.append(user_item)

print('len(user_item_list): ',len(user_item_list))

user_id = 0
for arr in user_item_list[1:]:
    n = len(arr)
    idx = np.random.permutation(n)
    # assert(n > num_train_per_user)
    for i in range(min(num_train_per_user, n)):
        pairs_train.append([user_id, arr[idx[i]]])
    if n > num_train_per_user:
        for i in range(num_train_per_user, n):
            pairs_test.append([user_id, arr[idx[i]]])
    user_id += 1
num_users = user_id
pairs_train = np.asarray(pairs_train).astype(np.int)
pairs_test = np.asarray(pairs_test).astype(np.int)
print('pairs_train.shape: ',pairs_train.shape)
print('pairs_test.shape: ',pairs_test.shape)
num_items = np.maximum(np.max(pairs_train[:, 1]), np.max(pairs_test[:, 1]))+1
print("num_users=%d, num_items=%d" % (num_users, num_items))



# social communication
with open("user_friends.dat") as f:
    lines = f.readlines()
    for line in lines[1:]:
        arr = line.strip().split()
        user = arr[0]
        friend = arr[1]
        if((user not in users.keys()) or (friend not in users.keys())):
            pass
        else:
            pairs_social.append([users[user], users[friend]])

pairs_social = np.array(pairs_social).astype(np.int)      
print('pairs_social.shape: ',pairs_social.shape)


tags = {}
with open('tags1.dat','rb') as f:
    lines = f.readlines()
    tag_id = 0
    for line in lines[1:]:
        line = line.decode('utf-8')
        arr = line.strip().split()
        tagid = arr[0]
        tags[tagid] = tag_id
        tag_id += 1
    print('num_tags: ',tag_id)

content = np.zeros((artist_id,tag_id))
with open('user_taggedartists-timestamps.dat') as f:
    lines = f.readlines()
    cnt = 0
    for line in lines[1:]:
        arr = line.strip().split()
        if(arr[1] not in artists.keys()):
            cnt += 1
            pass
        else:
            content[artists[arr[1]], tags[arr[2]]] = 1
    print('mistagged: ',cnt)

print('content.shape: ',content.shape)

item_user_matrix = np.zeros((num_items,num_users))


with open("train_users_{}.dat".format(num_train_per_user), "w") as fid:
    for user_id in range(num_users):
        this_user_items = pairs_train[pairs_train[:, 0]==user_id, 1]
        items_str = " ".join(str(x) for x in this_user_items)
        fid.write("%d %s\n" % (len(this_user_items), items_str))

with open("train_items_{}.dat".format(num_train_per_user), "w") as fid:
    for item_id in range(num_items):
        this_item_users = pairs_train[pairs_train[:, 1]==item_id, 0]
        for x in this_item_users:
            item_user_matrix[item_id,x] = 1
        users_str = " ".join(str(x) for x in this_item_users)
        fid.write("%d %s\n" % (len(this_item_users), users_str))

with open("test_users_{}.dat".format(num_train_per_user), "w") as fid:
    for user_id in range(num_users):
        this_user_items = pairs_test[pairs_test[:, 0]==user_id, 1]
        items_str = " ".join(str(x) for x in this_user_items)
        fid.write("%d %s\n" % (len(this_user_items), items_str))

with open("test_items_{}.dat".format(num_train_per_user), "w") as fid:
    for item_id in range(num_items):
        this_item_users = pairs_test[pairs_test[:, 1]==item_id, 0]
        users_str = " ".join(str(x) for x in this_item_users)
        fid.write("%d %s\n" % (len(this_item_users), users_str))


with open("social.dat", "w") as fid:
    for user_id in range(num_users):
        this_user_friends = pairs_social[pairs_social[:, 0]==user_id, 1]
        friends_str = " ".join(str(x) for x in this_user_friends)
        fid.write("%d %s\n" % (len(this_user_friends), friends_str))

print('item_user_matrix.shape: ',item_user_matrix.shape)

scipy.io.savemat('content.mat',{'X0':content, 'X1':item_user_matrix},do_compression = True)