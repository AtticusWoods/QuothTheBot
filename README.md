# QuothTheBot
An AI that converts English into a Shakespearian style (AKA: Early Modern English)

# Prerequisites
pip install transformers    
pip install torch   
pip install transformers[torch]  
pip install sentencepiece  

# Disclaimer
I trained this on a Windows 11 machine, I can't guarantee that it will work on other systems.

# How to
To train the model simply run the train.py file.
If you want to change any of the hyper parameters you can change the relevent variables near the top of the train.py file.
When training, the model will put checkpoints in the checkpoints folder. Make sure to clear those out before running a new training section or it will resume from the latest checkpoint.

If just want to generate responses run the generateResponse.py file and it will ask you for input to translate.

# Project Report
Refer to the Quoth_the_Bot_Project_Report.pdf

