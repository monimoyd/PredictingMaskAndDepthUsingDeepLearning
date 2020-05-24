import torch
from utils.plot_util import draw_and_save
from utils.iou_util import calculateIoU
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from time import time
from tqdm.notebook import tqdm
#import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset,DataLoader
import os
from PIL import Image as PILImage
import torchvision.transforms as transforms

from matplotlib import pyplot as plt




subtract_min = lambda imgs : torch.stack([img-torch.min(img) for img in imgs])  
divide_by_max = lambda imgs : torch.stack([img/ torch.max(img) for img in imgs if torch.max(img) > 0.0 ]) 


def _get_gpu_mem(synchronize=True, empty_cache=True):
    return torch.cuda.memory_allocated(), torch.cuda.memory_cached()


def _generate_mem_hook(handle_ref, mem, idx, hook_type, exp):
    def hook(self, *args):
        if len(mem) == 0 or mem[-1]["exp"] != exp:
            call_idx = 0
        else:
            call_idx = mem[-1]["call_idx"] + 1

        mem_all, mem_cached = _get_gpu_mem()
        torch.cuda.synchronize()
        mem.append({
            'layer_idx': idx,
            'call_idx': call_idx,
            'layer_type': type(self).__name__,
            'exp': exp,
            'hook_type': hook_type,
            'mem_all': mem_all,
            'mem_cached': mem_cached,
        })

    return hook


def _add_memory_hooks(idx, mod, mem_log, exp, hr):
    h = mod.register_forward_pre_hook(_generate_mem_hook(hr, mem_log, idx, 'pre', exp))
    hr.append(h)

    h = mod.register_forward_hook(_generate_mem_hook(hr, mem_log, idx, 'fwd', exp))
    hr.append(h)

    h = mod.register_backward_hook(_generate_mem_hook(hr, mem_log, idx, 'bwd', exp))
    hr.append(h)

def log_mem(model, inp, mem_log=None, exp=None):
    mem_log = mem_log or []
    exp = exp or f'exp_{len(mem_log)}'
    hr = []
    for idx, module in enumerate(model.modules()):
        _add_memory_hooks(idx, module, mem_log, exp, hr)

    try:
        out = model(inp)
        loss = out.sum()
        loss.backward()
    finally:
        [h.remove() for h in hr]

        return mem_log


def train(model,device,criteria1, criteria2, train_loader,optimizer, epoch, writer, MODEL_PATH, TRAIN_PLOT_PATH,  time_measurements = [0.0, 0.0, 0.0]):
    training_time = time_measurements[0]
    data_loading_time = time_measurements[1]
    misc_time =  time_measurements[2] 

    misc_start_time = time()
    mem_log = []
    exp = f'exp_{len(mem_log)}'
    hr = []
    for idx, module in enumerate(model.modules()):
        _add_memory_hooks(idx, module, mem_log, exp, hr)
    

    PATH = "./"
       
    model.train()
    pbar = tqdm(train_loader)
    misc_end_time = time() 
    misc_time += (misc_end_time - misc_start_time)
    
    try:
        for batch_idx, data in enumerate(pbar):
            data_loading_start_time = time()
            data['bg'] = data['bg'].to(device)
            data['fg_bg'] = data['fg_bg'].to(device)
            data['mask_black'] = data['mask_black'].to(device)   
            data['depth_fg_bg'] = data['depth_fg_bg'].to(device)
            data_loading_end_time = time()
            data_loading_time += (data_loading_end_time - data_loading_start_time )
        
            training_start_time = time()   
            optimizer.zero_grad()
            output = model(data['fg_bg'], data['bg'])
            loss1 = criteria1(output[0], data['mask_black'])
            s_min_out = subtract_min(output[1])
            output_1 = divide_by_max(s_min_out)
            loss2 = criteria2(output_1, data['depth_fg_bg'])
            #loss = 2*loss1 + (1 -  loss2)
            loss = 2*loss1 + (1 -  loss2)
            loss.backward()
            optimizer.step()
            training_end_time = time()  
            training_time += (training_end_time - training_start_time) 
        
            misc_start_time = time()
            pbar.set_description(desc= f'epoch={epoch+1} loss={loss.item()}')
            iteration = 40000*epoch + batch_idx
            writer.add_scalar('Loss/train', loss.item(), iteration)
            writer.add_scalar('BatchTrainingTime/train', training_time, iteration)
            writer.add_scalar('BatchDataLoadingTime/train', training_time, iteration)
        
            if batch_idx % 100 == 0:
                torch.save(model.state_dict(), f"{MODEL_PATH}/{epoch+1}_{batch_idx}.pth")
                #writer.add_scalar('BatchTrainingTime/train', training_time, iteration)
                #draw_and_save(data['bg'], f"{TRAIN_PLOT_PATH}/{epoch+1}_{batch_idx}_input_bg.jpg", display=False)
                #draw_and_save(data['fg_bg'], f"{TRAIN_PLOT_PATH}/{epoch+1}_{batch_idx}_input_fg_bg.jpg", display=False)
                iou_value = calculateIoU(output[0], data['mask_black'] )
                writer.add_scalar('IoU/train', iou_value, iteration)
                print(" For epoch: ", (epoch + 1), " after batch: ", batch_idx, " predicted mask IoU: ", iou_value )
                #draw_and_save(output[0], f"{TRAIN_PLOT_PATH}/{epoch+1}_{batch_idx}_output_mask_bg.jpg", display=False)
                #draw_and_save(output_1, f"{TRAIN_PLOT_PATH}/{epoch}_{batch_idx}_depth_fg_bg.jpg", display=False, display_with_plasma=True)

                torch.cuda.empty_cache()
            elif (batch_idx + 1) == 1800 or (batch_idx + 1) == 3600:
                torch.save(model.state_dict(), f"{MODEL_PATH}/{epoch+1}_final.pth")
                writer.add_scalar('BatchTrainingTime/train', training_time, iteration)
                print(" For epoch: ", (epoch +1), "  batch: ", batch_idx, " Background images: " )
                draw_and_save(data['bg'], f"{TRAIN_PLOT_PATH}/{epoch+1}_{batch_idx}_input_bg.jpg")
                print(" For epoch: ", (epoch + 1), " after batch: ", batch_idx, " Foreground superimposed on background images: " )
                draw_and_save(data['fg_bg'], f"{TRAIN_PLOT_PATH}/{epoch+1}_{batch_idx}_input_fg_bg.jpg")
                iou_value = calculateIoU(output[0], data['mask_black'] )
                writer.add_scalar('IoU/train', iou_value, iteration)
                print(" For epoch: ", (epoch + 1), " after batch: ", batch_idx, " predicted mask IoU: ", iou_value )
                print(" For epoch: ", (epoch + 1), " after batch: ", batch_idx, " Predicted mask images: " )
                draw_and_save(output[0], f"{TRAIN_PLOT_PATH}/{epoch+1}_{batch_idx}_output_mask_bg.jpg")
                print(" For epoch: ", (epoch + 1), " after batch: ", batch_idx, " Predicted depth images: " )
                draw_and_save(output_1, f"{TRAIN_PLOT_PATH}/{epoch}_{batch_idx}_depth_fg_bg.jpg", display_with_plasma=True)

            misc_end_time = time()
            misc_time += (misc_end_time - misc_start_time)
    finally:
        [h.remove() for h in hr]

        
    time_measurements[0] = training_time
    time_measurements[1] = data_loading_time
    time_measurements[2] = misc_time
    return mem_log
	
	
def test(model,device,criteria1, criteria2, test_loader, writer, TEST_PLOT_PATH):
    PATH = "./"
    model.eval()
    pbar = tqdm(test_loader)
    with torch.no_grad():
        for batch_idx, data in enumerate(pbar):
            data['bg'] = data['bg'].to(device)
            data['fg_bg'] = data['fg_bg'].to(device)
            data['mask_black'] = data['mask_black'].to(device)   
            data['depth_fg_bg'] = data['depth_fg_bg'].to(device)
        
            test_start_time = time()
            output = model(data['fg_bg'], data['bg'])
            test_end_time = time()
            test_time = test_end_time - test_start_time
            loss1 = criteria1(output[0], data['mask_black'])
            s_min_out = subtract_min(output[1])
            output_1 = divide_by_max(s_min_out)
            loss2 = criteria2(output_1, data['depth_fg_bg'])
            loss = 2*loss1 + (1 -  loss2)
            pbar.set_description(desc= f'loss={loss.item()}')
            writer.add_scalar('Loss/test', loss.item(), batch_idx)
            writer.add_scalar('TestingTime/test', test_time, batch_idx)
            if batch_idx % 100 == 0 or batch_idx==399:
            # pass
                print(" After iteration: ", (batch_idx), " Background images: " )
                draw_and_save(data['bg'], f"{TEST_PLOT_PATH}/{batch_idx}_input_fg_bg.jpg")
                print(" After iteration: ", (batch_idx), " foregound superimposed background images: " )
                draw_and_save(data['fg_bg'], f"{TEST_PLOT_PATH}/{batch_idx}_input_fg_bg.jpg")
                print(" After iteration: ", (batch_idx), " ground truth  mask images:" )
                draw_and_save(data['mask_black'], f"{TEST_PLOT_PATH}/{batch_idx}_input_mask_bg.jpg")
                mask_iou_value = calculateIoU(output[0], data['mask_black'] )
                writer.add_scalar('IoU/test', mask_iou_value, batch_idx)
                print(" After iteration: ", batch_idx, " Mask IoU:", mask_iou_value )
                print(" After iteration: ", (batch_idx), " predicted mask images:" )
                draw_and_save(output[0], f"{TEST_PLOT_PATH}/{batch_idx}_output_mask_bg.jpg")
                print(" After iteration: ", (batch_idx), " ground truth depth images:" )
                draw_and_save(data['depth_fg_bg'], f"{TEST_PLOT_PATH}/{batch_idx}_output_mask_bg.jpg", display_with_plasma=True)
                print(" After iteration: ", batch_idx, " predicted depth images:" )
                draw_and_save(output_1, f"{TEST_PLOT_PATH}/{batch_idx}_output_depth_fg_bg.jpg", display_with_plasma=True)
                          
                
                