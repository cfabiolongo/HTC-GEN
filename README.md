# HTC-GEN (Hieralchical Text Classification Generation)
This is the repository of the Python (3.10+) implementation of HTC-GEN, a state-of-the-art zero-shot approach for hieralchical text classification, presented as regular paper at [DATA 2024](https://data.scitevents.org/Home.aspx) conference held in Dijon in 9-11/07/2024.

![Image 1](images/HTC-Inference_generic.jpg)

# Installation

### Pytorch

Follow the instructions reported [here](https://pytorch.org/) for the current system.

### Llama 2 

Download and install Llama 2: https://github.com/meta-llama/llama

### Pandas 

```sh
> pip install pandas
> pip install openpyxl
```


# Code usage

The code includes source files for the following tasks, considering the case study of [Web of Science](https://data.mendeley.com/datasets/9rw3vkcfy4/6) dataset:

* Virtual-leaves generation
* Items generation from virtual-leaves
* Items generation from leaves

with the following correspondences:

* item: abstract
* leaf: area
* virtual-leaf: keywords


## *Virtual leaves* generation

This code was designed to build... 

* filename: [llama_2_ft_dolly_lora.py](https://github.com/cfabiolongo/elicit-meta-llm/blob/master/llama_2_ft_dolly_lora.py)

Relevant parameters:
 
* Training epochs
* Learning rate
* Path fine-tuned model

## *Abstracts* generation from *Virtual leaves*

This code was designed to build... 

* filename: [llama_2_ft_dolly_lora.py](https://github.com/cfabiolongo/elicit-meta-llm/blob/master/llama_2_ft_dolly_lora.py)

Relevant parameters:
 
* Training epochs
* Learning rate
* Path fine-tuned model 

## *Abstracts* generation from *leaves*

This code was designed to build... 

* filename: [llama_2_ft_dolly_lora.py](https://github.com/cfabiolongo/elicit-meta-llm/blob/master/llama_2_ft_dolly_lora.py)

Relevant parameters:
 
* Training epochs
* Learning rate
* Path fine-tuned model 

