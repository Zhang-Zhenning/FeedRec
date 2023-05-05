
from network import *
import random
from tqdm import tqdm
import pickle as pkl

# the logged data contains the following information:
# (i,f,d) where i is the item we recommend, f is the user feedback, d is the user dwell time
# along with f, we also have the following reward-related information:
# mc: the number of clicks for the feedback f, md: the number of scans for the feedback f, mr: the revisit time between 2 visits
# in the silumation, we generate mc in a range of (0,5), md in a range of (0,5), mr in a range of (0,10)

BETA = 15  # the weight of the mr term in the reward function
MC_WEIGHT = 1.0  # the weight of the mc term in the reward function
MD_WEIGHT = 1.0  # the weight of the md term in the reward function
MR_WEIGHT = 1.0  # the weight of the mr term in the reward function

# the reward function
def Reward(mc, md, mr):
    return MC_WEIGHT * mc + MD_WEIGHT * md + MR_WEIGHT * BETA / mr

# generate logged data M
def GenerateData(n=1000, length=10000):
    M = []
    U = []
    R = []
    for uidx in tqdm(range(n)):
        cur_m = []
        cur_r = []
        cur_user = random.randint(0, TOTAL_NUM_USERS-1)
        U.append(cur_user)
        for idx in range(length):
            cur_item = random.randint(0, TOTAL_NUM_ITEMS-1)
            cur_feedback = random.randint(0, 2)
            cur_dwell_time = random.uniform(0, 10)
            cur_mc = random.randint(0, 5)
            cur_md = random.randint(0, 5)
            cur_mr = random.uniform(0, 10)
            cur_m.append([cur_item, cur_feedback, cur_dwell_time])
            cur_r.append([cur_mc,cur_md,cur_mr])
        M.append(cur_m)
        R.append(cur_r)
    
    return M,U,R


if __name__ == '__main__':
    # generate logged data
    
    # if data.pkl exists, load it
    try:
        with open('../data.pkl','rb') as f:
            M = pkl.load(f)
    except:
        M = GenerateData()
        with open('../data.pkl','wb') as f:
            pkl.dump(M,f)
    
    Trajs,Users,Res = M

    test_user = Users[0]
    test_traj = Trajs[0][:100]
    test_res_temp = Res[0][:100]
    test_res = [k[2] for k in test_res_temp]

    # initialize the network
    sn = S_NetWork()
    qn = Q_NetWork()

    sn.to(device)
    qn.to(device)
    
    print(S_loss(qn,sn,State(test_user,test_traj,test_res)))






