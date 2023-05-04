import torch
import torch.nn as nn
import numpy as np

# ----------------------------define some constants--------------------------

# f_t (user feedback type) constants
CLICK = 0
PURCHASE = 1
SKIP = 2

# i_t (item type) constants
TOTAL_NUM_ITEMS = 1000
# function to get the one-hot vector of item type
def GetItemOneHot(item_type):
    item_type_vector = np.zeros(TOTAL_NUM_ITEMS)
    item_type_vector[item_type] = 1
    # make the type to int
    item_type_vector = item_type_vector.astype(int)
    return item_type_vector

# u (user id) constants
TOTAL_NUM_USERS = 1000
# function to get the one-hot vector of user id
def GetUserOneHot(user_id):
    user_id_vector = np.zeros(TOTAL_NUM_USERS)
    user_id_vector[user_id] = 1
    # make the type to int
    user_id_vector = user_id_vector.astype(int)
    return user_id_vector


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

        user_raw = torch.from_numpy(GetUserOneHot(user)).float().view(-1,1).to("cuda")
        nextitem_raw = torch.from_numpy(GetItemOneHot(nextitem)).float().view(-1,1).to("cuda")
        user_raw.requires_grad = False
        nextitem_raw.requires_grad = False

        # get user embedding
        userEmbedding = torch.mm(self.userEmbeddingMat,user_raw).view(1, -1)
        # get next item embedding
        nextItemEmbedding = torch.mm(self.itemEmbeddingMat,nextitem_raw).view(1, -1)
        # get dwel time vector
        dwelTimeVector = torch.from_numpy(sequence[:, 2]).float().view(-1, 1).to("cuda")


        # for each item, get the item embeding then get the item projection vector
        itemProjections = None
        for idx,item in enumerate(sequence[:, 0]):
            item_raw = torch.from_numpy(
                GetItemOneHot(item)).float().view(-1, 1).to("cuda")
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
        purchaseBehaviorVector, _ = self.purchaseLstm(rawBehaviorVector[:,purchaseIdx,:])
        clickBehaviorVector, _ = self.clickLstm(rawBehaviorVector[:,clickIdx,:])
        skipBehaviorVector, _ = self.skipLstm(rawBehaviorVector[:,skipIdx,:])

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
        self.leaveLayer = nn.Linear(16, 1)
        self.leaveOutput = nn.Sigmoid()


    # sequence should be in the format [...[i_j,f_j,d_j]...] (numpy array)
    def forward(self, user, sequence, nextitem):
        # get feed back type index
        purchaseIdx = torch.from_numpy(sequence[:, 1] == PURCHASE)
        clickIdx = torch.from_numpy(sequence[:, 1] == CLICK)
        skipIdx = torch.from_numpy(sequence[:, 1] == SKIP)

        user_raw = torch.from_numpy(GetUserOneHot(
            user)).float().view(-1, 1).to("cuda")
        nextitem_raw = torch.from_numpy(GetItemOneHot(
            nextitem)).float().view(-1, 1).to("cuda")
        user_raw.requires_grad = False
        nextitem_raw.requires_grad = False

        # get user embedding
        userEmbedding = torch.mm(self.userEmbeddingMat, user_raw).view(1, -1)
        # get next item embedding
        nextItemEmbedding = torch.mm(
            self.itemEmbeddingMat, nextitem_raw).view(1, -1)
        # get dwel time vector
        dwelTimeVector = torch.from_numpy(
            sequence[:, 2]).float().view(-1, 1).to("cuda")

        # for each item, get the item embeding then get the item projection vector
        itemProjections = None
        for idx, item in enumerate(sequence[:, 0]):
            item_raw = torch.from_numpy(
                GetItemOneHot(item)).float().view(-1, 1).to("cuda")
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
        purchaseBehaviorVector, _ = self.purchaseLstm(
            rawBehaviorVector[:, purchaseIdx, :])
        clickBehaviorVector, _ = self.clickLstm(
            rawBehaviorVector[:, clickIdx, :])
        skipBehaviorVector, _ = self.skipLstm(rawBehaviorVector[:, skipIdx, :])

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


model = Q_NetWork()
model.to("cuda")

model1 = S_NetWork()
model1.to("cuda")

test_data = np.array([[1, 1, 1], [2, 2, 2], [3, 0, 3]])

print(model(1, test_data, 10))
print(model1(1, test_data, 10))
    
        
    
