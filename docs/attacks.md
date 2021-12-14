## Attacks on watermarked models

The aim of such attacks is to remove the embedded watermarks. 

This repository provides two types of attacks: fine-tuning and pruning.

### Fine-tuning

Fine-tuning is a technique that comes from transfer learning and aims to use an already trained model for solving a new task. Instead of training a new model from scratch, the pre-trained model is trained on some small set of new data.  Since the weights of the model are changing during this process, it could potentially lead to watermarks removal.

### Pruning

Pruning is a technique that aims to reduce the complexity of the model. Given a certain threshold, all the weights below that threshold are set to 0. If weights responsible for retuning a correct response to the trigger images are zeroed, pruning may remove the corresponding watermarks.


For running an attack, see a guide [here](start.md).