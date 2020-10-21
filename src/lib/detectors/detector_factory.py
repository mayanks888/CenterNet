from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .exdet import ExdetDetector
from .ddd import DddDetector
from .ddd_multitask import DddDetector_multitask
from .ctdet import CtdetDetector
from .ctdet_multitask import CtdetDetector_multitask
from .multi_pose import MultiPoseDetector

detector_factory = {
  'exdet': ExdetDetector, 
  'ddd': DddDetector,
  'ddd_multitask': DddDetector_multitask,
  'ctdet': CtdetDetector,
  'ctdet_multitask': CtdetDetector_multitask,
  'multi_pose': MultiPoseDetector,
}
