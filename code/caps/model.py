import torch
import torch.nn as nn

from decoder import construct_decoder
from encoder import construct_encoder
from feature_env import FeatureEvaluator
from modules import *

from torch.distributions import MultivariateNormal, Normal
from torch.distributions import Categorical

SOS_ID = 0
EOS_ID = 0


# gradient based automatic feature selection
class GAINS(nn.Module):
    def __init__(self,
                 fe:FeatureEvaluator,
                 args
                 ):
        super(GAINS, self).__init__()
        self.style = args.method_name
        self.gpu = args.gpu
        self.encoder = construct_encoder(fe, args)
        self.decoder = construct_decoder(fe, args)
        if self.style == 'rnn':
            self.flatten_parameters()

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, target_variable=None):
        encoder_outputs, encoder_hidden, feat_emb, predict_value = self.encoder.forward(input_variable)
        decoder_hidden = (feat_emb.unsqueeze(0), feat_emb.unsqueeze(0))
        decoder_outputs, decoder_hidden, ret = self.decoder.forward(target_variable, decoder_hidden, encoder_outputs)
        decoder_outputs = torch.stack(decoder_outputs, 0).permute(1, 0, 2)
        feat = torch.stack(ret['sequence'], 0).permute(1, 0, 2)
        return predict_value, decoder_outputs, feat

    def generate_new_feature(self, input_variable, predict_lambda=1, direction='-'):
        encoder_outputs, encoder_hidden, feat_emb, predict_value, new_encoder_outputs, new_feat_emb = \
            self.encoder.infer(input_variable, predict_lambda, direction=direction)
        new_encoder_hidden = (new_feat_emb.unsqueeze(0), new_feat_emb.unsqueeze(0))
        decoder_outputs, decoder_hidden, ret = self.decoder.forward(None, new_encoder_hidden, new_encoder_outputs)
        new_feat_seq = torch.stack(ret['sequence'], 0).permute(1, 0, 2)
        return new_feat_seq

class SetTransformer(nn.Module):
    # ==========================================
    # 【类总览：特征空间的炼金炉】
    # 作用：它是一个编码器-解码器架构。
    #      1. 把离散的 0/1 特征选择（废铁），熔炼成高维连续的浮点数向量（金水）。
    #      2. 再把金水倒进模具，重新还原成另一组 0/1 的特征选择概率。
    # ==========================================
    def __init__(self, dim_input, num_outputs, dim_output,args,
            num_inds=32, dim_hidden=128, num_heads=4, ln=True):
        super(SetTransformer, self).__init__()
        self.arch = args.set_tf_arch # 默认是 'ISAB'
        self.dim_hidden = args.set_tf_hidden_size # 默认是 128 维
        # self.dim_hidden = args.set_tf_hidden_size
        # 【1. 搭建熔炉（编码器 Encoder）】
        if self.arch == 'ISAB':
            # ISAB 是高级的注意力机制（带诱导点，算得更快）
            self.enc = nn.Sequential(
                    ISAB(dim_input, self.dim_hidden, num_heads, num_inds, ln=ln),
                    ISAB(self.dim_hidden, self.dim_hidden, num_heads, num_inds, ln=ln))
        elif self.arch == 'SAB':
            # SAB 是标准的自注意力机制（互相看来看去）
            self.enc = nn.Sequential(
                    SAB(dim_input, self.dim_hidden, num_heads, ln=ln),
                    SAB(self.dim_hidden, self.dim_hidden, num_heads, ln=ln))
        # 【2. 搭建模具（解码器 Decoder）】
        self.dec = nn.Sequential(
                PMA(self.dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(self.dim_hidden, self.dim_hidden, num_heads, ln=ln),
                nn.Linear(self.dim_hidden, self.dim_hidden*4),
                nn.ReLU(),
                nn.Linear(self.dim_hidden*4, dim_output)
                )
        self.flatten = nn.Flatten(0,1)
        self.gpu = args.gpu
    # 【3. 炼金全过程（前向传播）】
    def forward(self, X):
        X = X.unsqueeze(1) # 给数据增加一个维度，满足网络输入的格式要求
        feat_emb = self.enc(X) # 魔法发生地！离散变连续！
        # 从连续变回概率预测
        X = self.dec(feat_emb).permute(0,2,1)
        # print(self.dec(feat_emb).shape, X.shape)
        output = self.flatten(X)
        # 同时返回“中间产物（连续向量）”和“最终预测（概率）”
        return feat_emb, output
    # 【4. 推理专用（PPO 冲刺时用的）】
    def infer(self, X):
        X = self.dec(X).permute(0,2,1)
        # torch.max()[1] 就是直接挑出概率最大的那个选项作为答
        output = torch.max(X,dim=-1)[1]
        return output


################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")
################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(ActorCritic, self).__init__()

        self.action_dim = action_dim
        # actor

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # mean head
        self.mean_head = nn.Linear(hidden_size, action_dim)

        # log std head
        self.std_head = nn.Linear(hidden_size, action_dim)

    def forward(self):
        raise NotImplementedError

    def act(self, state):

        mean = self.mean_head(self.actor(state))
        log_std = self.std_head(self.actor(state))
        std = torch.exp(log_std)

        dist = Normal(mean, std)

        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        state_val = self.critic(state)

        return action, action_logprob, state_val

    def evaluate(self, state, action):
        mean = self.mean_head(self.actor(state))
        log_std = self.std_head(self.actor(state))
        std = torch.exp(log_std)

        dist = Normal(mean, std)

        action_logprobs = dist.log_prob(action).sum(dim=-1)
        state_values = self.critic(state)

        return action_logprobs, state_values


class PPO(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip):
        super(PPO, self).__init__()
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(device)

        self.policy_old = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.policy.critic.parameters(), lr=lr_critic)
        self.MseLoss = nn.MSELoss()


    def select_action(self, state):

        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action


    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward in reversed(self.buffer.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward.tolist())
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        # critic_rewards = rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack([self.buffer.states[0]], dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack([self.buffer.actions[0]], dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack([self.buffer.logprobs[0]], dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack([self.buffer.state_values[0]], dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for search_step in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            policy_loss = -torch.min(surr1, surr2)
            value_loss = self.MseLoss(state_values, rewards)

            # actor gradient step
            self.actor_optimizer.zero_grad()
            policy_loss.mean().backward()
            self.actor_optimizer.step()

            # critic gradient step
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()

            print('Search Step: {}/{} Policy loss: {} Value loss: {}'.format(search_step,self.K_epochs,policy_loss.mean().item(), value_loss.item()))
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))