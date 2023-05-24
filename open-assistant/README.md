# Open-Assistant  

### Set-up Instructions  

**General**  
From the directory ~/projects/open-assistant/dev/AI run following commands   
> bash ../../nvm.sh  

**Local UI installation and run**  
From the directory ~/projects/open-assistant/dev/AI run following commands   
>npm init  
npm run install-chat-ui  
npm run start-mongodb  
npm run start-chat-ui  

**Inference server run locally**  
From the directory ~/projects/open-assistant/dev/AI run following commands   
>Check that the nvidia drivers are installed on your machine with nvidia-smi  
If not run ~/projects/nvidia.sh  

Then  
> npm run start-inference  

### Current issues  

**torch.cuda.OutOfMemoryError: CUDA out of memory**  
npm run start-inference fails with message error:  
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 200.00 MiB (GPU 0; 14.61 GiB total capacity; 13.76 GiB already allocated; 37.12 MiB free; 13.77 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF