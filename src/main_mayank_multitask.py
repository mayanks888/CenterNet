from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths
import os
import torch
import torch.utils.data
from src.lib.opts_mayank_multitask import opts
from src.lib.models.model import create_model, load_model, save_model,create_model_custom
from src.lib.models.data_parallel import DataParallel
from src.lib.logger import Logger
from src.lib.datasets.dataset_factory import get_dataset
from src.lib.trains.train_factory import train_factory


def main(opt):
  torch.manual_seed(opt.seed)
  # benchmark = True automatically find the most suitable high-efficiency algorithm ' \
  # for the current configuration to achieve the problem of optimizing operating efficiency
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    #Obtain the configuration required for training a specific datasets (pascal in this case)
  Dataset = get_dataset(opt.dataset, opt.task)
  # Update the data and other configurations, and set the model output heads.
  # For example, we need bounding box recognition task, we need to set up three outputs hm, wh, reg
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  print('Creating model...')
  #object detection, 3D bounding box detection,extremedet, multi-person human pose estimation
  # model = create_model(opt.arch, opt.heads, opt.head_conv)
  model = create_model_custom(opt.arch, opt.heads, opt.head_conv)

  # Here, heads = {'hm':3, 'wh':2, 'reg':2} the output channel through the dla model is 5,
  # and the output channel is 256 after the first convolution, and then after the final 1\times 1$ convolution output hm is 3 channels,
  # wh then 2 channels, reg then 2 aisle.
  optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

  ############################################3333
  #freezing backbone and one head
  for param in model.parameters():
    # print(param)
    param.requires_grad = False

  req_grad = ["model.hm_tl", "model.wh_tl", "model.reg_tl"]
  # for hd in model.reg_tl:
  for custom_head in (req_grad):
    for hd in eval(custom_head):
      # print(hd.parameters())
      for wt in hd.parameters():
        print(wt)
        wt.requires_grad = True

  ######################################################
  # select training env of desired architecture ctdet in this case
  Trainer = train_factory[opt.task]
  # Define loss for hm, wh and offset
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'),
      batch_size=1,
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  # if opt.test:
  #   _, preds = trainer.val(0, val_loader)
  #   val_loader.dataset.run_eval(preds, opt.save_dir)
  #   return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )

  print('Starting training...')
  best = 1e10
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      if log_dict_val[opt.metric] < best:
        best = log_dict_val[opt.metric]
        save_model(os.path.join(opt.save_dir, 'model_best.pth'), 
                   epoch, model)
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                 epoch, model, optimizer)
    logger.write('\n')
    if epoch in opt.lr_step:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)