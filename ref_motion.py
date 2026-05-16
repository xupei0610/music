from typing import Optional, Sequence, Union
import os
import torch
import numpy as np
import yaml
import pickle
from collections import namedtuple

from utils import slerp

Motion = namedtuple("Motion",
    "fps pos orient ang_vel lin_vel joint_pos joint_vel"
)

class ReferenceMotion():
    def __init__(self, motion_file: Union[str, Sequence[str]],
        character_model: str, 
        key_links: Optional[Sequence[int]]=None, 
        device: Optional[torch.device]=None
    ):
        self.device = device

        motions = []
        if type(motion_file) == str:
            motions.extend(self.load_motions(motion_file, key_links))
        else:
            for m in motion_file:
                motions.extend(self.load_motions(m, key_links))
        self.prepare_data(motions)

    def prepare_data(self, motions):
        self.motion_link_pos_tensor = torch.cat([m[0] for m in motions]).to(self.device)
        self.motion_link_orient_tensor = torch.cat([m[1] for m in motions]).to(self.device)
        self.motion_link_vel_tensor = torch.cat((torch.cat([m[2] for m in motions]), torch.cat([m[3] for m in motions])),-1).to(self.device)
        self.motion_joint_pos_tensor = torch.cat([m[4] for m in motions]).to(self.device)
        self.motion_joint_vel_tensor = torch.cat([m[5] for m in motions]).to(self.device)
        self.motion_dt_tensor = torch.tensor([m[6] for m in motions], dtype=torch.float, device=self.device)
        self.motion_n_frames_tensor = torch.tensor([m[0].size(0)-1 for m in motions], dtype=torch.int, device=self.device)
        self.motion_length = np.array([m[6]*(m[0].size(0)-1) for m in motions])
        self.motion_length_tensor = torch.from_numpy(self.motion_length).to(device=self.device, dtype=torch.float)
        self.motion_tensor_offset = torch.cumsum(torch.tensor([0]+[m[0].size(0) for m in motions[:-1]]), 0).to(self.device)

        tot_weights, tot_length = 0, 0
        tot_length_with_weights = 0
        for m in motions:
            w, t = m[7], m[0].size(0)-1
            if w is None or w < 0:
                tot_length += t
            elif w > 0:
                tot_weights += w
                tot_length_with_weights += t
                tot_length += t
        motion_weight = []
        for m in motions:
            w, t = m[7], m[0].size(0)-1
            if tot_length != tot_length_with_weights and (w is None or w < 0):
                if tot_length_with_weights == 0:
                    w = t/tot_length
                else:
                    w = t*tot_weights/tot_length_with_weights
            motion_weight.append(w)
        self.motion_weight = np.array(motion_weight, dtype=float)
        self.motion_weight /= np.sum(self.motion_weight)

        print("Loaded {:d} motions with a total length of {:.3f}s ({} frames).".format(len(motions), sum(self.motion_length), self.motion_link_pos_tensor.size(0)))


    def load_motions(self, motion_file, key_links):
        motions = []
        if os.path.splitext(motion_file)[1] == ".yaml":
            with open(motion_file, 'r') as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)
            dirname = os.path.dirname(motion_file)
            motion_files = []
            motion_weights = []
            for item in motion_config['motions']:
                motion_weights.append(item['weight'])
                motion_files.append(os.path.join(dirname, item['file']))
        else:
            motion_files = [motion_file]
            motion_weights = [None]

        n_motion_files = len(motion_files)
        for f, (w, motion_file) in enumerate(zip(motion_weights, motion_files)):

            if os.path.splitext(motion_file)[1] == ".joblib":
                if "joblib" not in globals():
                    import joblib
                data = joblib.load(motion_file)
                motion_len = 0
                n_frames = 0
                for motion in data:
                    dt = 1.0 / motion.fps
                    w = None
                    motions.append((
                        motion.pos if key_links is None else motion.pos[:,key_links],
                        motion.orient if key_links is None else motion.orient[:,key_links],
                        motion.lin_vel if key_links is None else motion.lin_vel[:,key_links],
                        motion.ang_vel if key_links is None else motion.ang_vel[:,key_links],
                        motion.joint_pos,
                        motion.joint_vel,
                        dt, w
                    ))
                    n_frame = len(motion.pos)
                    n_frames += n_frame
                    motion_len += dt*n_frame
            else:
                with open(motion_file, "rb") as _:
                    motion = pickle.load(_)
                n_frames = len(motion.pos)
                fps = motion.fps
                
                dt = 1.0 / fps
                motion_len = dt * (n_frames - 1)
                motions.append((
                    motion.pos if key_links is None else motion.pos[:,key_links],
                    motion.orient if key_links is None else motion.orient[:,key_links],
                    motion.lin_vel if key_links is None else motion.lin_vel[:,key_links],
                    motion.ang_vel if key_links is None else motion.ang_vel[:,key_links],
                    motion.joint_pos,
                    motion.joint_vel,
                    dt, w
                ))
                print("\t{:.4f}s, {:d} Hz, {:d} frames".format(motion_len, fps, n_frames))
        return motions

    def sample(self, n, truncate_time=None):
        motion_ids = np.random.choice(len(self.motion_weight), size=n, p=self.motion_weight, replace=True)
        phase = np.random.uniform(low=0.0, high=1.0, size=motion_ids.shape)
        motion_len = self.motion_length[motion_ids]
        if truncate_time is not None: motion_len -= truncate_time
        motion_time = phase * motion_len
        return motion_ids, motion_time

    @torch.no_grad
    def state(self, motion_ids, motion_times, with_joint_tensor=False):
        if not torch.is_tensor(motion_ids):
            motion_ids = torch.from_numpy(motion_ids)
        if not torch.is_tensor(motion_times):
            motion_times = torch.from_numpy(motion_times).to(device=self.device, dtype=torch.float)
        
        n_frames = self.motion_n_frames_tensor[motion_ids]
        motion_len = self.motion_length_tensor[motion_ids]
        dt = self.motion_dt_tensor[motion_ids]
        motion_id_offset = self.motion_tensor_offset[motion_ids]

        fid0 = (motion_times / motion_len).clip_(min=0, max=1).mul_(n_frames).to(torch.int)
        fid1 = (fid0+1).clip_(max=n_frames)
        frac = (motion_times - fid0*dt).div_(dt).clip_(max=1).view(-1, 1, 1)

        one_frac = 1.0-frac
        
        fid0.add_(motion_id_offset)
        fid1.add_(motion_id_offset)

        link_pos0 = self.motion_link_pos_tensor[fid0]
        link_pos1 = self.motion_link_pos_tensor[fid1]
        link_orient0 = self.motion_link_orient_tensor[fid0]
        link_orient1 = self.motion_link_orient_tensor[fid1]
        link_vel = self.motion_link_vel_tensor[fid0]

        link_pos = (link_pos0*one_frac).add_(frac*link_pos1)
        link_orient = slerp(link_orient0, link_orient1, frac)
        link_tensor = torch.cat((link_pos, link_orient, link_vel), -1)
        if with_joint_tensor:
            joint_p0 = self.motion_joint_pos_tensor[fid0]
            joint_p1 = self.motion_joint_pos_tensor[fid1]
            joint_vel = self.motion_joint_vel_tensor[fid1]
            frac.squeeze_(-1)
            one_frac.squeeze_(-1)
            joint_pos = (joint_p0*one_frac).add_(frac*joint_p1)
            return link_tensor, (joint_pos, joint_vel)
        else:
            return link_tensor

