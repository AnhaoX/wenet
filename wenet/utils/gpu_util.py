#!/usr/bin/env python3

import logging
import subprocess
import time
import torch


def get_gpu_compute_mode(gpu_id=0):
    cmd = 'nvidia-smi -q -d COMPUTE -i {} | grep "Compute Mode"'.format(gpu_id)
    gpu_info = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return a.stdout.strip().split()[-1]


def select_gpu_device(use_gpu='wait',
                      poll_interval=2,
                      timeout=30) -> (torch.device, int):
    """
    Select one GPU device (or one CPU device if allowed).
    This function requires CUDA being in 'EXCLUSIVE_PROCESS' compute mode,
    so it can poll each GPU on that machine in a round-robin manner till
    finding one that is available.

    Args:
        use_gpu (str): Mode for GPU selection. There are four modes supported
                       for now:
                        1. no (default): Only ask for CPU device.
                           Just return one as is wished.
                        2. yes: Want a GPU device right now.
                           This function polls each GPU at most once for a free
                           GPU device, and will admit failure after acquiring
                           nothing after that round of polling.
                        3. optional: Accept a CPU device if no GPU is available
                           for now.
                           Do as 'yes' mode, but if it fails, back off to CPU.
                        4. wait: Wait for a GPU device to be available.
                           Will do GPU polling every ``poll_interval`` minute(s)
                           until a success or a timeout.
        poll_interval (int): Time interval (in minute) for round-robin GPU
                             polling; only is valid in 'wait' mode and will
                             be at least 1 minute.
        timeout (int): Timeout (in minute) for the operation that repeatedly
                       polls GPUs to acquire one; only valid in 'wait' mode.

    Returns:
        device (torch.device): Will be None if no device is acquired.
        gpu_id (int): Returns this value just in case this is useful.
                      Will be None if no GPU is available.

    """
    if use_gpu == 'no':
        logging.info('Manually selected to compute on CPU.')
        return torch.device('cpu'), None
    elif use_gpu == 'wait':
        # Polling interval should be at least 1 minute in 'wait' mode.
        poll_interval = max(1, poll_interval)
    elif use_gpu in ['yes', 'optional']:
        # This is a hack, which allows only one round of GPU polling
        # by setting timeout to 0.
        timeout = 0
    else:
        logging.error('Please choose : --use_gpu=yes|no|optional|wait,'
            ' passed {}.'.format(use_gpu))
        raise ValueError

    logging.info('Selecting GPU in \'{}\' mode...'.format(use_gpu))
    minutes_elapsed = 0
    while True:
        for i in range(torch.cuda.device_count()):
            try:
                device = torch.device('cuda', i)
                tensor = torch.zeros(1)
                tensor = tensor.to(device)
                return device, i
            except RuntimeError:
                logging.debug('GPU {} is busy or unavailable.'.format(i))
                pass
        logging.info('Failed to acquire any GPU.')
        if mode == 'optional':
            logging.info('Running on CPU since --use_gpu=optional specified.')
            return torch.device('cpu'), None
        minutes_elapsed += poll_interval
        if minutes_elapsed >= timeout:
            break
        logging.warning('Will try again in {} minute(s).'.format(poll_interval))
        time.sleep(poll_interval * 60)
    return None, None
