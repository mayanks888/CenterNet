from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ctdet import CtdetTrainer
from .ddd import DddTrainer
from .exdet import ExdetTrainer
from .multi_pose import MultiPoseTrainer
from .ctdet_multitask import Ctdet_multitask_Trainer
from .ddd_multitask import ddd_multitask_Trainer

train_factory = {
  'exdet': ExdetTrainer, 
  'ddd': DddTrainer,
  'ctdet': CtdetTrainer,
  'multi_pose': MultiPoseTrainer,
  'cdtet_multitask': Ctdet_multitask_Trainer,
  'ddd_multitask': Ctdet_multitask_Trainer
}
