
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
def GenerateData(n=1000, length=2000):
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
        with open('../data_final.pkl','rb') as f:
            M = pkl.load(f)
    except:
        M = GenerateData()
        with open('../data_final.pkl','wb') as f:
            pkl.dump(M,f)
    
    Trajs,Users,Res = M

    # divide the dataset into training set and test set
    train_trajectories = Trajs[:800]
    train_users = Users[:800]
    train_rewardparas = Res[:800]

    test_trajectories = Trajs[800:]
    test_users = Users[800:]
    test_rewardparas = Res[800:]

    train_revisits = [k[2] for k in train_rewardparas]
    test_revisits = [k[2] for k in test_rewardparas]

    # setup the training dataset buffer
    train_buffer = []
    for traj, user, revisit, rewardparas in zip(train_trajectories, train_users, train_revisits, train_rewardparas):
        for end_idx in range(400,1000):
            s1 = State(user,traj[:end_idx],revisit[:end_idx])
            s2 = State(user,traj[:end_idx+1],revisit[:end_idx+1])
            r1 = Reward(*rewardparas[end_idx])
            i1 = traj[end_idx][0]
            train_buffer.append([s1,i1,r1,s2])

    # initialize the network
    sn = S_NetWork()
    qn = Q_NetWork()

    sn.to(device)
    qn.to(device)
    
    # define the optimizer
    sn_optimizer = torch.optim.Adam(sn.parameters(), lr=0.0001)
    qn_optimizer = torch.optim.Adam(qn.parameters(), lr=0.0001)

    # pretrining the S network
    print(f"Pretraining the S network...")
    batch = random.sample(train_buffer,400,replacement=False)
    for epoch in range(50):
        # random sample 400 pairs from train_buffer
        total_loss = 0
        for cur_pair in batch:
            s1,i1,r1,s2 = cur_pair
            cur_loss = S_loss(qn,sn,s1)
            total_loss += cur_loss.item()

            # update the S network
            sn_optimizer.zero_grad()
            cur_loss.backward()
            sn_optimizer.step()

        print(f"Epoch: {epoch}, loss: {total_loss}")
        

    # iterative training of S-network and Q-network
    print(f"Training the S network and Q network...")
    for i in range(100):
        # generate pairs from train_buffer and S network
        # sample 50 pairs from train_buffer
        cur_buffer = random.sample(train_buffer,50,replacement=False)

        # sample a user sequence by S network
        cur_user = random.randint(0,TOTAL_NUM_USERS-1)
        cur_traj = [train_trajectories[0][0]]
        cur_revisit = [train_revisits[0]]
        leaving = False
        cur_state = State(cur_user,cur_traj,cur_revisit)


        while not leaving:
            # generate next item by Q network

            # using epsilon-greedy strategy
            if random.random() < EPS:
                max_item = random.randint(0,TOTAL_NUM_ITEMS-1)
            else:
                max_score = float('-inf')
                max_item = -1
                with torch.no_grad():
                    for cur_item in range(TOTAL_NUM_ITEMS):
                        cur_score = qn(cur_state.user,cur_state.x,cur_item)
                        if cur_score > max_score:
                            max_score = cur_score
                            max_item = cur_item
            
            # get f_t,d_t,v_t,l_t by S network
            with torch.no_grad():
                cur_feedback, cur_dwell_time, cur_revisit_time, cur_leaving = sn(cur_state.user,cur_state.x,max_item)
            
            # convert cur_feedback, cur_dwell_time, cur_revisit_time, leaving to numpy array
            cur_feedback = cur_feedback.detach().cpu().numpy()
            cur_dwell_time = float(cur_dwell_time.detach().cpu())
            cur_revisit_time = float(cur_revisit_time.detach().cpu())
            cur_leaving = cur_leaving.detach().cpu().numpy()
            
            # get the max feedback from cur_feedback
            real_feedback = np.argmax(cur_feedback)
            leaving = True if cur_leaving[0] > 0.7 else False
            
            cur_mc = 1 if real_feedback == CLICK else 0
            cur_mc = 5 if real_feedback == PURCHASE else cur_mc
            cur_md = cur_dwell_time
            cur_mr = cur_revisit_time
            # get current reward
            cur_reward = Reward(cur_mc,cur_md,cur_mr)

            # generate next state
            cur_traj.append([max_item,real_feedback,cur_dwell_time])
            cur_revisit.append(cur_revisit_time)
            next_state = State(cur_user,cur_traj,cur_revisit)

            # add into cur_buffer
            cur_buffer.append([cur_state,max_item,cur_reward,next_state])

            cur_state = next_state
            if len(cur_traj) > 1000:
                leaving = True
        
        # update the Q network
        for q_pair in cur_buffer:
            cur_loss = Q_loss(qn,*q_pair)

            qn_optimizer.zero_grad()
            cur_loss.backward()
            qn_optimizer.step()
        
        print(f"Epoch: {i}, Q loss: {cur_loss.item()}")

        # update the S network
        s_batch = random.sample(train_buffer,50,replacement=False)
        for epoch in range(50):
            # random sample 400 pairs from train_buffer
            for cur_pair in s_batch:
                s1,i1,r1,s2 = cur_pair
                cur_loss = S_loss(qn,sn,s1)

                # update the S network
                sn_optimizer.zero_grad()
                cur_loss.backward()
                sn_optimizer.step()

            

            
           







