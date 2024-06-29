# A High-Granularity Convolutional Approach to Food Insecurity Prediction
*by Stanley Howard and Zach Hird*
## Submission to NeurIPS high school track 2024: machine learning for social good
The repository containing the work done for Tau Ai's submission into the "Call for High school projects 2024" from NeurIPS.

## Abstract
> Predicting local hunger indices and food insecurity is integral for effective resource allocation and targeted interventions to combat global hunger. Traditional approaches to the prediction of food insecurity are limited by spatiotemporal resolution, accuracy and precision. We suggest that this can be addressed by training high-resolution predictive models as tools that analysts can use in tandem with traditional resources to enhance crisis response. In this paper we demonstrate that a Convolutional Long-Short Term Memory (LSTM) model trained on purely historical land and climate data within a highly localised scope can produce predictions comparable, and of a higher resolution spatiotemporally, to the quarterly manual projection accuracy of the Famine Early Warning System. Our model predicts food insecurity on an unprecedented spatial extent and granularity compared to other food insecurity prediction models. 

## How to reproduce

We developed and trained this model in a TPU V4-8 VM, so first cloning this repository into such an enviroment would be needed if attempting to audit with the least possible friction. Thus, a rough outline to reproduce is as follows:

- Initialise a TPU-VM using the software version `tpu-vm-tf-2.16.1-pjrt`
- SSH into the VM, clone the repository and mount a persistent disk at `/mnt/disks/data` (optional - you could refactor the project to use file system on the TPU-VM)
- Install all neccessary libaries for python and execute `FLDAS_fetcher.py` to download the FLDAS dataset
- To parse both datasets (FEWS is small enough to be contained on the repository), run both `FLDAS_parser.py` and `fews_parser.py` to consolidate them insto .npy files in tensor form.
- Check the TPU is correctly configured with `tpu-test.py`, manually adjusting the `setup.sh` to modify environment variables according to your situation.
- Run `ConvolutionalLSTM.py` to train the model
- Run `validation.py` to verify the model statistically (or indeed load the pretrained weights provided in `Final_pretrained.keras`)
**REMEMBER:** Refactor filepaths before running to suit your filesystem to prevent errors and overwrites

## Acknowledgements

> We would like to thank Jason Stock for continual motivation, advice and support. We would also like to thank Colin Roberts and Waylon Jepsen.  Additionally, this research was supported by Google's TPU Research Cloud with the generous allocation of cloud TPUs for the purpose of training.  Large language models (LLMs) were utilised for purely administrative tasks, such as assisting with rasterisation code. No primary contributions architecturally or textually were made using LLMs.