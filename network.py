import torch
import torch.nn as nn
import numpy as np
import random
from copy import copy
from tqdm import tqdm

# ----------------------------define some constants--------------------------

LAMB = 0.9
EPS = 0.1

# f_t (user feedback type) constants
CLICK = 0
PURCHASE = 1
SKIP = 2

# i_t (item type) constants
TOTAL_NUM_ITEMS = 1000
# function to get the one-hot vector of item type
def GetItemOneHot(item_type):
    item_type = int(item_type)
    item_type_vector = np.zeros(TOTAL_NUM_ITEMS)
    item_type_vector[item_type] = 1
    # make the type to int
    item_type_vector = item_type_vector.astype(int)
    return item_type_vector

# u (user id) constants
TOTAL_NUM_USERS = 1000
# function to get the one-hot vector of user id
def GetUserOneHot(user_id):
    user_id = int(user_id)
    user_id_vector = np.zeros(TOTAL_NUM_USERS)
    user_id_vector[user_id] = 1
    # make the type to int
    user_id_vector = user_id_vector.astype(int)
    return user_id_vector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------define the state------------------------------
class State:
    def __init__(self,user,trajectory,revisit_times):
        self.user = user
        self.raw_traj = copy(trajectory) # it,ft,dt
        self.raw_revisit = copy(revisit_times)

        self.x = trajectory
        self.x = np.array(self.x)
        self.x = self.x.reshape((-1,3))

        # produce the revisit time vector
        self.revisit_times = np.array(revisit_times).reshape(1,-1)

        # produce the leave vector (size 2, one-hot,first element is 1 if the user will leave, second element is 1 if the user will stay)
        self.leave = np.zeros((len(trajectory),2))
        for idx in range(len(trajectory)):
            if trajectory[idx][1] == SKIP:
                self.leave[idx][0] = 1
            else:
                self.leave[idx][1] = 1
        
        # produce the feed back vector (one-hot, size 3, first element is 1 if the user clicks, second element is 1 if the user purchases, third element is 1 if the user skips)
        self.feedback = np.zeros((len(trajectory),3))
        for idx in range(len(trajectory)):
            if trajectory[idx][1] == CLICK:
                self.feedback[idx][0] = 1
            elif trajectory[idx][1] == PURCHASE:
                self.feedback[idx][1] = 1
            elif trajectory[idx][1] == SKIP:
                self.feedback[idx][2] = 1



# ----------------------------define the network-----------------------------
class Q_NetWork(nn.Module):
    def __init__(self, Time_LSTM_input_size = 10, item_emb_len = 32, user_emb_len = 32,Time_LSTM_hidden_size = 32, Secondary_LSTM_hidden_size = 40):
        super(Q_NetWork, self).__init__()

        self.timeLstmInputSize = Time_LSTM_input_size
        self.timeLstmHiddenSize = Time_LSTM_hidden_size
        self.secLstmHiddenSize = Secondary_LSTM_hidden_size
        self.itemEmbLen = item_emb_len
        self.userEmbLen = user_emb_len

        # item embedding, convert one-hot item vector into item embedding
        self.itemEmbeddingMat = nn.Parameter(torch.randn(self.itemEmbLen, TOTAL_NUM_ITEMS))
        # user embedding, convert one-hot user vector into user embedding
        self.userEmbeddingMat = nn.Parameter(torch.randn(self.userEmbLen, TOTAL_NUM_USERS))
        
        # FeedBack matrix to convert item embedding into item projection vector to feed into LSTM
        self.clickProjection = nn.Parameter(torch.randn(self.timeLstmInputSize-1, self.itemEmbLen))
        self.purchaseProjection = nn.Parameter(torch.randn(self.timeLstmInputSize-1, self.itemEmbLen))
        self.skipProjection = nn.Parameter(torch.randn(self.timeLstmInputSize-1, self.itemEmbLen))
        
        # raw behavior layer
        self.timeLstm = nn.LSTM(
            input_size=self.timeLstmInputSize, hidden_size=self.timeLstmHiddenSize)
        
        # hierarchical behavior layer
        self.purchaseLstm = nn.LSTM(
            input_size=self.timeLstmHiddenSize, hidden_size=self.secLstmHiddenSize)
        self.clickLstm = nn.LSTM(
            input_size=self.timeLstmHiddenSize, hidden_size=self.secLstmHiddenSize)
        self.skipLstm = nn.LSTM(
            input_size=self.timeLstmHiddenSize, hidden_size=self.secLstmHiddenSize)
        
        # output layer
        self.outputLayer = nn.Linear(self.itemEmbLen+self.userEmbLen+self.timeLstmHiddenSize+3*self.secLstmHiddenSize, 16)
        self.activation = nn.ReLU()
        self.outputLayer2 = nn.Linear(16, 1)

    # sequence should be in the format [...[i_j,f_j,d_j]...] (numpy array)
    def forward(self, user,sequence, nextitem):

        # get feed back type index
        purchaseIdx = torch.from_numpy(sequence[:, 1] == PURCHASE)
        clickIdx = torch.from_numpy(sequence[:, 1] == CLICK)
        skipIdx = torch.from_numpy(sequence[:, 1] == SKIP)

        user_raw = torch.from_numpy(GetUserOneHot(user)).float().view(-1,1).to(device)
        nextitem_raw = torch.from_numpy(GetItemOneHot(nextitem)).float().view(-1,1).to(device)
        user_raw.requires_grad = False
        nextitem_raw.requires_grad = False

        # get user embedding
        userEmbedding = torch.mm(self.userEmbeddingMat,user_raw).view(1, -1)
        # get next item embedding
        nextItemEmbedding = torch.mm(self.itemEmbeddingMat,nextitem_raw).view(1, -1)
        # get dwel time vector
        dwelTimeVector = torch.from_numpy(sequence[:, 2]).float().view(-1, 1).to(device)
        dwelTimeVector.requires_grad = False



        # for each item, get the item embeding then get the item projection vector
        itemProjections = None
        for idx,item in enumerate(sequence[:, 0]):
            item_raw = torch.from_numpy(
                GetItemOneHot(item)).float().view(-1, 1).to(device)
            # item_raw don't need to be a parameter
            item_raw.requires_grad = False

            # convert item to one-hot vector then get item embedding
            itemEmbedding = torch.mm(self.itemEmbeddingMat, item_raw).view(-1, 1)

            if sequence[idx][1] == PURCHASE:
                itemEmbedding = torch.mm(self.purchaseProjection, itemEmbedding).view(1, -1)
            elif sequence[idx][1] == CLICK:
                itemEmbedding = torch.mm(self.clickProjection, itemEmbedding).view(1, -1)
            elif sequence[idx][1] == SKIP:
                itemEmbedding = torch.mm(self.skipProjection, itemEmbedding).view(1, -1)
            else:
                print("Error: unknown feedback type")

            if itemProjections is None:
                itemProjections = itemEmbedding
            else:
                itemProjections = torch.cat([itemProjections, itemEmbedding], dim=0)
        
        # get the d_j vector
        itemProjections = torch.cat([dwelTimeVector,itemProjections], dim=1).unsqueeze(0)


        # get the raw behavior vector
        rawBehaviorVector, _ = self.timeLstm(itemProjections)
        H_r_t = rawBehaviorVector.squeeze(0)[-1].view(1, -1)

        # get the hierarchical behavior vector
        if rawBehaviorVector[:, purchaseIdx, :].shape[1] != 0:
            purchaseBehaviorVector, _ = self.purchaseLstm(
                rawBehaviorVector[:, purchaseIdx, :])
        else:
            purchaseBehaviorVector = torch.zeros(
                1, 1, self.secLstmHiddenSize).to(device)
            
        if rawBehaviorVector[:, clickIdx, :].shape[1] != 0:
            clickBehaviorVector, _ = self.clickLstm(
                rawBehaviorVector[:, clickIdx, :])
        else:
            clickBehaviorVector = torch.zeros(
                1, 1, self.secLstmHiddenSize).to(device)
            
        if rawBehaviorVector[:, skipIdx, :].shape[1] != 0:
            skipBehaviorVector, _ = self.skipLstm(
                rawBehaviorVector[:, skipIdx, :])
        else:
            skipBehaviorVector = torch.zeros(
                1, 1, self.secLstmHiddenSize).to(device)

        H_s_t = skipBehaviorVector.squeeze(0)[-1].view(1, -1)
        H_c_t = clickBehaviorVector.squeeze(0)[-1].view(1, -1)
        H_p_t = purchaseBehaviorVector.squeeze(0)[-1].view(1, -1)

        # concatenate all the vectors
        infoFusion = torch.cat([nextItemEmbedding,userEmbedding, H_r_t, H_s_t, H_c_t, H_p_t], dim=1)

        # feed into the output layer
        output = self.outputLayer2(self.activation(self.outputLayer(infoFusion)))

        return output

class S_NetWork(nn.Module):
    def __init__(self, Time_LSTM_input_size=10, item_emb_len=32, user_emb_len=32, Time_LSTM_hidden_size=32, Secondary_LSTM_hidden_size=40):
        super(S_NetWork, self).__init__()

        self.timeLstmInputSize = Time_LSTM_input_size
        self.timeLstmHiddenSize = Time_LSTM_hidden_size
        self.secLstmHiddenSize = Secondary_LSTM_hidden_size
        self.itemEmbLen = item_emb_len
        self.userEmbLen = user_emb_len

        # item embedding, convert one-hot item vector into item embedding
        self.itemEmbeddingMat = nn.Parameter(
            torch.randn(self.itemEmbLen, TOTAL_NUM_ITEMS))
        # user embedding, convert one-hot user vector into user embedding
        self.userEmbeddingMat = nn.Parameter(
            torch.randn(self.userEmbLen, TOTAL_NUM_USERS))

        # FeedBack matrix to convert item embedding into item projection vector to feed into LSTM
        self.clickProjection = nn.Parameter(torch.randn(
            self.timeLstmInputSize-1, self.itemEmbLen))
        self.purchaseProjection = nn.Parameter(
            torch.randn(self.timeLstmInputSize-1, self.itemEmbLen))
        self.skipProjection = nn.Parameter(torch.randn(
            self.timeLstmInputSize-1, self.itemEmbLen))

        # raw behavior layer
        self.timeLstm = nn.LSTM(
            input_size=self.timeLstmInputSize, hidden_size=self.timeLstmHiddenSize)

        # hierarchical behavior layer
        self.purchaseLstm = nn.LSTM(
            input_size=self.timeLstmHiddenSize, hidden_size=self.secLstmHiddenSize)
        self.clickLstm = nn.LSTM(
            input_size=self.timeLstmHiddenSize, hidden_size=self.secLstmHiddenSize)
        self.skipLstm = nn.LSTM(
            input_size=self.timeLstmHiddenSize, hidden_size=self.secLstmHiddenSize)


        # generate feedback type ft and dwell time dt
        self.outputWay1 = nn.Linear(
            self.itemEmbLen+self.userEmbLen+self.timeLstmHiddenSize+3*self.secLstmHiddenSize,16)
        self.activationWay1 = nn.Tanh()
        self.dwellTimeLayer = nn.Linear(16, 1)
        self.feedbackLayer = nn.Linear(16, 3)
        self.feedbackOutput = nn.Softmax(dim=1)

        # generate leaving probability lt and revisit time vt
        self.outputWay2 = nn.Linear(self.itemEmbLen+self.userEmbLen+self.timeLstmHiddenSize+3*self.secLstmHiddenSize,16)
        self.activationWay2 = nn.Tanh()
        self.revisitLayer = nn.Linear(16, 1)
        self.leaveLayer = nn.Linear(16, 2)
        self.leaveOutput = nn.Softmax(dim=1)


    # sequence should be in the format [...[i_j,f_j,d_j]...] (numpy array)
    def forward(self, user, sequence, nextitem):

        # get feed back type index
        purchaseIdx = torch.from_numpy(sequence[:, 1] == PURCHASE)
        clickIdx = torch.from_numpy(sequence[:, 1] == CLICK)
        skipIdx = torch.from_numpy(sequence[:, 1] == SKIP)

        user_raw = torch.from_numpy(GetUserOneHot(
            user)).float().view(-1, 1).to(device)
        nextitem_raw = torch.from_numpy(GetItemOneHot(
            nextitem)).float().view(-1, 1).to(device)
        user_raw.requires_grad = False
        nextitem_raw.requires_grad = False

        # get user embedding
        userEmbedding = torch.mm(self.userEmbeddingMat, user_raw).view(1, -1)
        # get next item embedding
        nextItemEmbedding = torch.mm(
            self.itemEmbeddingMat, nextitem_raw).view(1, -1)
        # get dwel time vector
        dwelTimeVector = torch.from_numpy(
            sequence[:, 2]).float().view(-1, 1).to(device)
        dwelTimeVector.requires_grad = False

        # for each item, get the item embeding then get the item projection vector
        itemProjections = None
        for idx, item in enumerate(sequence[:, 0]):
            item_raw = torch.from_numpy(
                GetItemOneHot(item)).float().view(-1, 1).to(device)
            # item_raw don't need to be a parameter
            item_raw.requires_grad = False

            # convert item to one-hot vector then get item embedding
            itemEmbedding = torch.mm(
                self.itemEmbeddingMat, item_raw).view(-1, 1)

            if sequence[idx][1] == PURCHASE:
                itemEmbedding = torch.mm(
                    self.purchaseProjection, itemEmbedding).view(1, -1)
            elif sequence[idx][1] == CLICK:
                itemEmbedding = torch.mm(
                    self.clickProjection, itemEmbedding).view(1, -1)
            elif sequence[idx][1] == SKIP:
                itemEmbedding = torch.mm(
                    self.skipProjection, itemEmbedding).view(1, -1)
            else:
                print("Error: unknown feedback type")

            if itemProjections is None:
                itemProjections = itemEmbedding
            else:
                itemProjections = torch.cat(
                    [itemProjections, itemEmbedding], dim=0)

        # get the d_j vector
        itemProjections = torch.cat(
            [dwelTimeVector, itemProjections], dim=1).unsqueeze(0)

        # get the raw behavior vector
        rawBehaviorVector, _ = self.timeLstm(itemProjections)
        H_r_t = rawBehaviorVector.squeeze(0)[-1].view(1, -1)

        # get the hierarchical behavior vector
        if rawBehaviorVector[:, purchaseIdx, :].shape[1] != 0:
            purchaseBehaviorVector, _ = self.purchaseLstm(
                rawBehaviorVector[:, purchaseIdx, :])
        else:
            purchaseBehaviorVector = torch.zeros(
                1, 1, self.secLstmHiddenSize).to(device)
            
        if rawBehaviorVector[:, clickIdx, :].shape[1] != 0:
            clickBehaviorVector, _ = self.clickLstm(
                rawBehaviorVector[:, clickIdx, :])
        else:
            clickBehaviorVector = torch.zeros(
                1, 1, self.secLstmHiddenSize).to(device)
            
        if rawBehaviorVector[:, skipIdx, :].shape[1] != 0:
            skipBehaviorVector, _ = self.skipLstm(
                rawBehaviorVector[:, skipIdx, :])
        else:
            skipBehaviorVector = torch.zeros(
                1, 1, self.secLstmHiddenSize).to(device)
    
        H_s_t = skipBehaviorVector.squeeze(0)[-1].view(1, -1)
        H_c_t = clickBehaviorVector.squeeze(0)[-1].view(1, -1)
        H_p_t = purchaseBehaviorVector.squeeze(0)[-1].view(1, -1)

        # concatenate all the vectors
        infoFusion = torch.cat(
            [nextItemEmbedding, userEmbedding, H_r_t, H_s_t, H_c_t, H_p_t], dim=1)
        
        # get the feedback type
        f_t = self.feedbackOutput(self.feedbackLayer(self.activationWay1(self.outputWay1(infoFusion))))
        # get the dwell time
        d_t = self.dwellTimeLayer(self.activationWay1(self.outputWay1(infoFusion)))
        # get the revisit time
        v_t = self.revisitLayer(self.activationWay2(self.outputWay2(infoFusion)))
        # get the leave probability
        l_t = self.leaveOutput(self.leaveLayer(self.activationWay2(self.outputWay2(infoFusion))))


        return f_t,d_t,v_t,l_t


# pi(ik | sk) based on Q network and eps-greedy
def Policy_Q(q_model,usr,cur_x,ik,eps=EPS):
    # sk is a state
    # ik is a item
    # eps is the probability to choose a random item
    if random.random() < eps:
        return 1/TOTAL_NUM_ITEMS
    else:
        maxi_item = None
        maxi_score = float('-inf')
        
        with torch.no_grad():
            for item in range(TOTAL_NUM_ITEMS):
                score = q_model(usr,cur_x,item)
                if score > maxi_score:
                    maxi_score = score
                    maxi_item = item
            
        if maxi_item == ik:
            return 1
        else:
            return 0

# pb(ik | sk) based on logged data policy
def Policy_B():
    return 1/TOTAL_NUM_ITEMS

# loss function for Q network
def Q_loss(q_model,s1,i1,r1,s2,lamb=LAMB):

    # find the max score of s2
    maxi_score = float('-inf')
    # this part is served as a constant in the loss function, so we don't need to calculate the gradient
    with torch.no_grad():
        for item in range(TOTAL_NUM_ITEMS):
            score = q_model(s2.user,s2.x, item)
            if score > maxi_score:
                maxi_score = score
                maxi_item = item
    
    score_s1_i1 = q_model(s1.user,s1.x,i1)
    loss = (score_s1_i1 - (r1 + lamb * maxi_score)).pow(2).mean()

    return loss

# loss function for S network
def S_loss(q_model,s_model,st,lamb=LAMB):
    lamb_f, lamb_l, lamb_d, lamb_v = 1, 1, 1, 1
    total_loss = None
    is_weight = None
    T = len(st.raw_traj)
    for t in tqdm(range(T-1)):
        # cur_state = State(st.user, st.raw_traj[:t+1], st.raw_revisit[:t+1])
        f_t,d_t,v_t,l_t = s_model(st.user,st.x[:t+1,:], st.raw_traj[t+1][0])
        g_f_t = st.feedback[t+1]
        g_d_t = st.raw_traj[t+1][2]
        g_v_t = st.revisit_times[0][t+1]
        g_l_t = st.leave[t+1]

        # calculate the loss of S network
        cur_loss = lamb_f * torch.nn.functional.kl_div(torch.log(f_t), torch.from_numpy(g_f_t).to(device), reduction='batchmean') + lamb_d * (d_t - g_d_t).pow(2).mean() + lamb_v * (v_t - g_v_t).pow(2).mean() + lamb_l * torch.nn.functional.kl_div(torch.log(l_t), torch.from_numpy(g_l_t).to(device), reduction='batchmean')
        # calculate the importance sampling weight
        # cur_is_weight = Policy_Q(q_model, st.user,st.x[:t+1,:], st.raw_traj[t][0]) / Policy_B()
        # if is_weight is None:
        #     is_weight = cur_is_weight
        # else:
        #     is_weight *= cur_is_weight
        
        if total_loss is None:
            total_loss = (lamb**t) * cur_loss 
        else:
            total_loss = total_loss + (lamb**t) * cur_loss
    
    return total_loss



if __name__ == "__main__":

    model = Q_NetWork()
    model.to(device)

    model1 = S_NetWork()
    model1.to(device)

    test_data = np.array([[1, 1, 1], [2, 2, 2.9], [3, 0, 3.7]])
    s1 = State(1, test_data,None)

    print(model(s1, 10))
    print(model1(s1, 10))
        
        
    
