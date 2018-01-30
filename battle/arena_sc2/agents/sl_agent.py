from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from pysc2.lib import actions
from pysc2.lib import features
from pysc2.agents import base_agent
from pysc2.lib.actions import FUNCTIONS
from pysc2.lib.actions import TYPES
from pysc2.lib.features import SCREEN_FEATURES
from pysc2.lib.features import MINIMAP_FEATURES
from pysc2.lib.features import FeatureType

UNITTYPE_WHITELIST = [0, 5, 6, 11, 18, 19, 20, 21, 22, 23,
                      24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                      34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                      44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                      54, 55, 56, 57, 58, 130, 132, 134, 146, 147,
                      149, 268, 341, 342, 343, 365, 472, 473, 474, 483,
                      484, 490, 498, 500, 561, 609, 638, 639, 640, 641,
                      662, 665, 666, 689, 691, 692, 734, 830, 880, 1879,
                      1883]


class SLAgent(base_agent.BaseAgent):

    def __init__(self, model_path, use_gpu=False, enable_batchnorm=False):
        super(SLAgent, self).__init__()
        self._model_path = model_path
        self._use_gpu = use_gpu
        self._enable_batchnorm = enable_batchnorm

    def setup(self, obs_spec, action_spec):
        super(SLAgent, self).setup(obs_spec, action_spec)
        assert obs_spec["screen"][2] == obs_spec["minimap"][2]
        self._resolution = obs_spec["screen"][2]
        self._unittype_map, self._observation_filter, c_screen, c_minimap = \
            self._init_observation_transformer(
                unittype_whitelist=UNITTYPE_WHITELIST)
        self._valid_action_ids, self._action_dims, self._action_args_map = \
            self._init_action_transformer(
                resolution=self._resolution, action_filter=[])
        self._actor_critic = self._load_model(
            self._action_dims, c_screen, c_minimap, self._resolution,
            self._model_path, self._use_gpu, self._enable_batchnorm)
            
    def step(self, obs):
        super(SLAgent, self).step(obs)
        ob, _, _, info = self._transform_observation(obs)
        action = self._make_step(ob, info)
        function_id, function_args = self._transform_action(action)
        return actions.FunctionCall(function_id, function_args)
        
    def _make_step(self, ob, info, greedy=False):
        self._actor_critic.eval()
        info.remove(0) # remove no_op action
        screen_feature = torch.from_numpy(np.expand_dims(ob[0], 0))
        minimap_feature = torch.from_numpy(np.expand_dims(ob[1], 0))
        player_feature = torch.from_numpy(np.expand_dims(ob[2], 0))
        mask = np.ones((1, self._action_dims[0]), dtype=np.float32) * 1e30
        mask[0, info] = 0
        mask = torch.from_numpy(mask)
        if self._use_gpu:
            screen_feature = screen_feature.cuda()
            minimap_feature = minimap_feature.cuda()
            player_feature = player_feature.cuda()
            mask = mask.cuda()
        policy_logprob, value = self._actor_critic(
            screen=Variable(screen_feature, volatile=True),
            minimap=Variable(minimap_feature, volatile=True),
            player=Variable(player_feature, volatile=True),
            mask=Variable(mask, volatile=True))
        # value
        victory_prob = value.data[0, 0]
        # control - function id
        if greedy:
            function_id = torch.max(
                policy_logprob[:, :self._action_dims[0]], 1)[1].data[0]
        else:
            function_id = torch.exp(policy_logprob[:, :self._action_dims[0]])\
                .multinomial(1).data[0, 0]
        # control - function arguments
        arguments = []
        for arg_id in self._action_args_map[function_id]:
            l = sum(self._action_dims[:arg_id+1])
            r = sum(self._action_dims[:arg_id+2])
            if greedy:
                arg_val = torch.max(policy_logprob[:, l:r], 1)[1].data[0]
            else:
                arg_val = torch.exp(
                    policy_logprob[:, l:r]).multinomial(1).data[0, 0]
            arguments.append(arg_val)
        return [function_id] + arguments

    def _transform_action(self, action):
        function_id = self._valid_action_ids[action[0]]
        function_args = []
        for arg_val, arg_info in zip(action[1:], FUNCTIONS[function_id].args):
            if len(arg_info.sizes) == 2:
                coords = np.unravel_index(arg_val, (self._resolution,) * 2)
                function_args.append(coords[::-1])
            elif len(arg_info.sizes) == 1:
                function_args.append([arg_val])
        return function_id, function_args

    def _transform_observation(self, timestep):
        obs_screen = self._transform_spatial_features(
            timestep.observation["screen"], SCREEN_FEATURES)
        obs_minimap = self._transform_spatial_features(
            timestep.observation["minimap"], MINIMAP_FEATURES)
        obs_player = self._transform_player_feature(
            timestep.observation["player"])
        obs = (obs_screen, obs_minimap, obs_player)
        done = timestep.last()
        info = timestep.observation["available_actions"]
        info = [self._valid_action_ids.index(fid)
                for fid in info if fid in self._valid_action_ids]
        return obs, timestep.reward, done, info

    def _transform_player_feature(self, obs):
        return np.log10(obs[1:].astype(np.float32) + 1)

    def _transform_spatial_features(self, obs, specs):
        features = []
        for ob, spec in zip(obs, specs):
            if spec.name in self._observation_filter:
                continue
            scale = spec.scale
            if spec.name == "unit_type" and self._unittype_map:
                ob = np.vectorize(lambda k: self._unittype_map.get(k, 0))(ob)
                scale = len(self._unittype_map)
            if spec.type == FeatureType.CATEGORICAL:
                features.append(np.eye(scale, dtype=np.float32)[ob][:, :, 1:])
            else:
                features.append(
                    np.expand_dims(np.log10(ob + 1, dtype=np.float32), axis=2))
        return np.transpose(np.concatenate(features, axis=2), (2, 0, 1))

    def _load_model(self, action_dims, in_channels_screen, in_channels_minimap,
                    resolution, model_path, use_gpu, enable_batchnorm):
        model = FullyConvNet(
            resolution=resolution,
            in_channels_screen=in_channels_screen,
            in_channels_minimap=in_channels_minimap,
            out_channels_spatial=3,
            out_dims_nonspatial=action_dims[0:1] + action_dims[4:],
            enable_batchnorm=enable_batchnorm)
        model.load_state_dict(
            torch.load(model_path, map_location=lambda storage, loc: storage))
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        if use_gpu:
            model.cuda()
        return model

    def _init_action_transformer(self, resolution=64, action_filter=[]):
        valid_action_ids = list(set(range(524)) - set(action_filter))
        action_dims = [len(valid_action_ids)]
        for argument in TYPES:
            if len(argument.sizes) == 2:
                action_dims.append(resolution ** 2)
            elif len(argument.sizes) == 1:
                action_dims.append(argument.sizes[0])
            else:
                raise NotImplementedError
        action_args_map = []
        for func_id in valid_action_ids:
            action_args_map.append([arg.id for arg in FUNCTIONS[func_id].args])
        return valid_action_ids, action_dims, action_args_map
        
    def _init_observation_transformer(self, observation_filter=[],
                                      unittype_whitelist=None):
        unittype_map = None
        if unittype_whitelist:
            unittype_map = {v : i for i, v in enumerate(unittype_whitelist)}
        observation_filter = set(observation_filter) 
        def get_spatial_channels(specs):
            num_channels = 0
            for spec in specs:
                if spec.name in observation_filter:
                    continue
                if spec.name == "unit_type" and unittype_map:
                    num_channels += len(unittype_map) - 1
                    continue
                if spec.type == FeatureType.CATEGORICAL:
                    num_channels += spec.scale - 1
                else:
                    num_channels += 1
            return num_channels
        num_channels_screen = get_spatial_channels(SCREEN_FEATURES)
        num_channels_minimap = get_spatial_channels(MINIMAP_FEATURES)
        return (unittype_map, observation_filter,
                num_channels_screen, num_channels_minimap)


class FullyConvNet(nn.Module):
    def __init__(self,
                 resolution,
                 in_channels_screen,
                 in_channels_minimap,
                 out_channels_spatial,
                 out_dims_nonspatial,
                 enable_batchnorm=False):
        super(FullyConvNet, self).__init__()
        self.screen_conv1 = nn.Conv2d(in_channels=in_channels_screen,
                                      out_channels=16,
                                      kernel_size=5,
                                      stride=1,
                                      padding=2)
        self.screen_conv2 = nn.Conv2d(in_channels=16,
                                      out_channels=32,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)
        self.minimap_conv1 = nn.Conv2d(in_channels=in_channels_minimap,
                                       out_channels=16,
                                       kernel_size=5,
                                       stride=1,
                                       padding=2)
        self.minimap_conv2 = nn.Conv2d(in_channels=16,
                                       out_channels=32,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        self.spatial_policy_conv = nn.Conv2d(in_channels=74,
                                             out_channels=out_channels_spatial,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0)
        if enable_batchnorm:
            self.screen_bn1 = nn.BatchNorm2d(16)
            self.screen_bn2 = nn.BatchNorm2d(32)
            self.minimap_bn1 = nn.BatchNorm2d(16)
            self.minimap_bn2 = nn.BatchNorm2d(32)
            self.player_bn = nn.BatchNorm2d(10)
            self.state_bn = nn.BatchNorm1d(256)
        self.state_fc = nn.Linear(74 * (resolution ** 2), 256)
        self.value_fc = nn.Linear(256, 1)
        self.nonspatial_policy_fc = nn.Linear(256, sum(out_dims_nonspatial))

        self._enable_batchnorm = enable_batchnorm
        self._action_dims = out_dims_nonspatial[0:1] + [resolution ** 2] * 3 \
                            + out_dims_nonspatial[1:]

    def forward(self, screen, minimap, player, mask):
        player = player.clone().repeat(
            screen.size(2), screen.size(3), 1, 1).permute(2, 3, 0, 1)
        if self._enable_batchnorm:
            screen = F.leaky_relu(self.screen_bn1(self.screen_conv1(screen)))
            screen = F.leaky_relu(self.screen_bn2(self.screen_conv2(screen)))
            minimap = F.leaky_relu(self.minimap_bn1(self.minimap_conv1(minimap)))
            minimap = F.leaky_relu(self.minimap_bn2(self.minimap_conv2(minimap)))
            player = self.player_bn(player.contiguous())
        else:
            screen = F.leaky_relu(self.screen_conv1(screen))
            screen = F.leaky_relu(self.screen_conv2(screen))
            minimap = F.leaky_relu(self.minimap_conv1(minimap))
            minimap = F.leaky_relu(self.minimap_conv2(minimap))
        screen_minimap = torch.cat((screen, minimap, player), 1)
        if self._enable_batchnorm:
            state = F.leaky_relu(self.state_bn(self.state_fc(
                screen_minimap.view(screen_minimap.size(0), -1))))
        else:
            state = F.leaky_relu(self.state_fc(
                screen_minimap.view(screen_minimap.size(0), -1)))

        spatial_policy = self.spatial_policy_conv(screen_minimap)
        spatial_policy = spatial_policy.view(spatial_policy.size(0), -1)
        nonspatial_policy = self.nonspatial_policy_fc(state)

        value = F.sigmoid(self.value_fc(state))
        first_dim = self._action_dims[0] 
        policy_logit = torch.cat([nonspatial_policy[:, :first_dim] - mask,
                                  spatial_policy,
                                  nonspatial_policy[:, first_dim:]],
                                 dim=1)
        policy_logprob = self._group_log_softmax(
            policy_logit, self._action_dims)
        return policy_logprob, value

    def _group_log_softmax(self, x, group_dims):
        idx = 0
        output = []
        for dim in group_dims:
            output.append(F.log_softmax(x[:, idx:idx+dim], dim=1))
            idx += dim
        return torch.cat(output, dim=1)
