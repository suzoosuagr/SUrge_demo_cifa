# SUrge Demo cifa

code modified from [pytorch/tutorial/blitz/cifa10](https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py)

Please feel free to start an issue and sharing your problems : )

## TL;DR (Quickstart)
1. Copy your code and database to Condor machine (e.g `/home/USERNAME/workspace/SHiTNet`)
2. Create your docker image on your machine (let's say with the tag `docker:cifa`)
3. Convert docker image `docker:cifa` to a singularity image:
   ```bash
   docker run --privileged -t --rm -v /var/run/docker.sock:/var/run/docker.sock -v /tmp/singularity_images:/output singularityware/docker2singularity:v2.6 docker:cifa
   ```
   Your singularity image will be created at `/tmp/singularity_images/docker-cifa_DATETIME-HASH.simg`
4. Transfer the singularity image to Condor machine
   ```bash
   scp /tmp/singularity_images/docker-cifa_DATETIME-HASH.simg condor:/home/USERNAME/singularity_images
   ```
5. Login to condor machine and submit the job inside your desired directory. Example job description file:  
   Submit file (my_job.sub):
   ```sub
   executable = /usr/local/bin/singularity
   arguments = exec --nv /home/USERNAME/singularity_images/docker-cifa_DATETIME-HASH.simg python train.py --batch-size=16 --use-adam
   
   error      = logs/singularity.err
   output     = logs/singularity.out
   log        = logs/singularity.log
   
   Requirements = TARGET.vm_name == "its-u18-nfs-20191029_gpu"
   
   +request_gpus = 1
   
   queue
   ```
   How to submit:
   ```bash
   condor_submit my_job.sub
   ```

## FAQ

### How to specify GPU type

By adding a control sentence in .sub file : 
```sub

executable = /usr/local/bin/singularity
arguments = exec --nv docker_cifa.simg python train.py  --batch=16

error      = cifademo.err
output     = cifademo.out
log        = cifademo.log

Requirements = TARGET.vm_name == "its-u18-nfs-20191029_gpu" && CUDADeviceName != "Quadro RTX 5000"

+request_gpus = 1

queue
```

In this exmaple I added `&& CUDADeviceName != "Quadro RTX 5000"` Which means not using RTX 5000 series card, or you can following the grammar to specify training only on `Quadro RTX 5000`, `Quadro RTX 6000`. I recommand to using the given example. 