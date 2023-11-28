# How to set up multi-GPU environment on GCP

## Puropose
Learn the way to run your code:
- on jupyter notebook in GCP;
- with multiple GPUs on a VM in GCP.

Caution: This might not be the only way or the best way, but just the way I passed to meet the purpose above.



## Step 0: (Optional) Request quotas
You need to request quotas for multiple GPUs if you haven't done before.
1. Refer to [this page](https://cloud.google.com/compute/resource-usage) for quotas.
    - Type of GPU and region you request is up to you.
    - E.g. I did four L4, in us-east region. (My quotas request for 8 L4 was rejected.)



## Step 1: Go to Vertex AI and create an instance (then VM will be created automatically)
1. Access [this link](https://cloud.google.com/vertex-ai-notebooks?hl=en), and push the "Go to console" button.
2. Click "CREATE NEW", then "ADVANCED OPTIONS".
3. In Details tab, choose region and zone that you got quotas and allow you to set preferred GPU.
4. In Machine type tab, choose GPU you want to use.
    - T4/V100/... -> "N1 standard"
    - L4 -> "G2 standard"
    - A100 -> "A2 highgpu" : but I wasn't able to create an instance with A100 maybe due to the excessive demand.
5. You can set as you want in other tab.
    - Environment: not sure which is the best, but I chose "Python 3 (with Intel MKL and CUDA 11.8)"
6. Push "CREATE" button at the bottom of the page.



## Step 2: Access the Jupyter lab environment on the instance.
1. If you successfully created an instance, you just click "OPEN JUPYTERLAB" to access Jupyter Lab environment.
2. You can start coding with the default environment, execept for the following error.
    - Google permission error occurred when imporing peft
        -> https://github.com/TimDettmers/bitsandbytes/issues/620
    - Installed transformers with version==4.34.1 since got error when setting all linear layers as target_modules for LoRA.

* If you want to confirm the status of GPU devices, command "nvidia-smi" on terminal or "!nvidia-smi" on notebook. 

