# SUrge Demo cifa

code modified from [pytorch/tutorial/blitz/cifa10](https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py)

## TL;DR (Quickstart)
1. Copy your code and database to Condor machine (e.g `/home/USERNAME/workspace/SHiTNet`)
2. Create your docker image on your machine (let's say with the tag `my_image:latest`)
3. Convert docker image `my_image:latest` to a singularity image:
   ```bash
   docker run --privileged -t --rm -v /var/run/docker.sock:/var/run/docker.sock -v /tmp/singularity_images:/output singularityware/docker2singularity:v2.6 my_image:latest
   ```
   Your singularity image will be created at `/tmp/singularity_images/my_image-latest_DATETIME-HASH.simg`
4. Transfer the singularity image to Condor machine
   ```bash
   scp /tmp/singularity_images/my_image-latest_DATETIME-HASH.simg condor:/home/USERNAME/singularity_images
   ```
5. Login to condor machine and submit the job inside your desired directory. Example job description file:  
   Submit file (my_job.sub):
   ```sub
   executable = /usr/local/bin/singularity
   arguments = exec --nv /home/USERNAME/singularity_images/my_image-latest_DATETIME-HASH.simg python train.py --batch-size=16 --use-adam
   
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