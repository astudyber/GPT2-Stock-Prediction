# GPT2-Stock-Prediction

 Predicting stock trends using the time series model GPT-2.



<div align="center">


This is the official repository for **GPT2-Stock-Prediction**.

![made-for-VSCode](https://img.shields.io/badge/Made%20for-VSCode-1f425f.svg)

</div>

</div>





## :bulb: Train based on GPT2

* **Successful implementation**: Transfer the model used for predicting text to predict floating-point numbers!

![LoRA for Qwen3](img/1.png)




##  :hourglass: Environment

* We use python==3.11 and pytorch == 2.3.0  with CUDA version 11.8
* Training with NVIDIA GeForce 4090 GPU is sufficient.
* Create environment:

```python
conda create --name lora python=3.11
conda activate gpt2
```

* Install the corresponding version pytorch:

```python
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
```

* Install other dependency packages:

```bash
Download GPT-2 Model's pth from huggingface into folder /model
```



## 📧 Connecting with Us



If you have any questions, please feel free to send email to `hzcheng@chd.edu.cn`



## 📜 Acknowledgment



 This project is inspired by *Digital Video and HD: Algorithms and Interfaces*, *Deep Learning*( The Flower Book ), *Linear Algebra Done Right*, etc. 